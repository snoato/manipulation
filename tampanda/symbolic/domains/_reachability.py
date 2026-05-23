"""Shared infrastructure for the per-cell executability test.

Each multi-grid domain exposes two functions consumed by
:mod:`examples.check_executability`:

* ``make_setup(scratch_dir, motion=True) -> DomainSetup`` — builds the
  full env (sim + workspace + bridge + executor) and the default
  initial layout.
* ``reachability_spec() -> ReachabilitySpec`` — declares which cells
  the test should treat as the *contract* (must pass) and which are
  diagnostic (full-grid sweep).

The test runner :func:`run_reachability_check` then:

1. Picks the cell list according to the chosen mode.
2. For each cell, parks all movables, places the test object alone at
   the cell, resets the arm + state, and calls
   ``executor.pick(...)`` end-to-end (RRT* + IK + gripper close +
   attach + lift).  Retries up to ``retries`` times with a fresh
   ``np.random`` seed each attempt.  Aborts a cell if the cumulative
   wall time exceeds ``time_cap_s``.
3. Renders a per-region pass/fail matrix at the end.

Layout-mode is the gate (must be 100 % PASS for a domain to be
considered well-formed).  Full-mode is the diagnostic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tampanda.symbolic.workspace import Cell, GridRegion, Workspace


# ---------------------------------------------------------------------------
# Per-domain spec
# ---------------------------------------------------------------------------

@dataclass
class DomainSetup:
    """Live environment + bookkeeping consumed by the reachability test.

    ``object_half_extents`` is a callable rather than a dict because some
    domains use heterogeneous-shape objects whose half-sizes must be
    looked up from the model.
    """
    name: str
    env: Any                                          # FrankaEnvironment
    workspace: Workspace
    object_ids: List[str]
    initial_layout: Dict[str, Cell]
    goal: List[Tuple] = field(default_factory=list)
    executor: Optional[Any] = None                    # PickPlaceExecutor
    place_at_cell: Optional[Callable[[Any, Workspace, str, str], None]] = None
    object_half_extents: Optional[Callable[[str], np.ndarray]] = None
    parked_xyz: Tuple[float, float, float] = (100.0, 0.0, 0.05)
    # Optional per-domain home q.  The default Franka home has q1=0
    # (arm reaches +x).  For +y-mounted shelves this is equidistant
    # from ±π/2 and IK falls into the wrong basin.  Setting q1=+π/2
    # in this domain's home pre-rotates the base so IK seeds favour
    # the correct side.
    home_qpos: Optional[np.ndarray] = None
    # Optional domain-specific pick override.  Receives ``(obj_name,
    # pos, half, quat)`` and returns True iff the simulated pick
    # succeeded.  Used for domains where the generic
    # ``executor.pick`` doesn't cope with the geometry — e.g.,
    # access19 needs a column-aligned approach + row-by-row descent
    # plus gripper-rotation invariance.  When ``None`` (default), the
    # runner falls back to ``setup.executor.pick``.
    pick_fn: Optional[Callable[[str, np.ndarray, np.ndarray, np.ndarray],
                                bool]] = None


@dataclass
class ReachabilitySpec:
    """Per-domain declaration of what the executability test should check.

    Attributes:
        domain_name:        Human-readable label.
        full_regions:       Region names to sweep in full mode.
        layout_test_objects:
                            For each ``(name, cell)`` in
                            ``DomainSetup.initial_layout`` we test by
                            placing **this** specific body at the cell,
                            so heterogeneous-shape domains exercise the
                            actual body that lives there.  Set ``None``
                            (default) to use a single representative
                            body (``layout_proxy``) for every layout
                            cell — appropriate for homogeneous domains.
        layout_proxy:       Body name used for each layout cell when
                            ``layout_test_objects`` is ``None``.
        full_proxy:         Body name used for every cell in full mode
                            (always a single representative — the test
                            is concerned with reachability, not which
                            body fits).  If ``None``, falls back to
                            ``layout_proxy``.
        extra_goal_cells:   Extra cells to require in layout mode beyond
                            the initial-layout cells.  E.g., the
                            specific top-deck cell where the OoI must
                            end up in tabletop_access.
    """
    domain_name: str
    full_regions: Tuple[str, ...]
    layout_test_objects: Optional[Tuple[Tuple[str, Cell], ...]] = None
    layout_proxy: Optional[str] = None
    full_proxy: Optional[str] = None
    extra_goal_cells: Tuple[Cell, ...] = ()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    cell: Cell
    obj_name: str
    success: bool
    elapsed_s: float
    reason: str
    attempts: int


@dataclass
class RegionResult:
    region_name: str
    cells_x: int
    cells_y: int
    cells: List[CellResult]
    excluded_cells: frozenset = frozenset()

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.cells if c.success)

    @property
    def n_total(self) -> int:
        return len(self.cells)

    @property
    def all_pass(self) -> bool:
        return all(c.success for c in self.cells)


@dataclass
class DomainResult:
    domain_name: str
    mode: str
    regions: List[RegionResult]

    @property
    def all_pass(self) -> bool:
        return all(r.all_pass for r in self.regions)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _reset_state(setup: DomainSetup) -> None:
    """Clear stale attachment / collision exceptions, open the gripper,
    park every movable, reset the arm to its home pose."""
    env = setup.env
    if getattr(env, "_attached", None) is not None:
        env.detach_object()
    if hasattr(env, "clear_collision_exceptions"):
        env.clear_collision_exceptions()
    if env.controller is not None:
        env.controller.open_gripper()
    parked = np.asarray(setup.parked_xyz)
    for name in setup.object_ids:
        env.set_object_pose(name, parked)
    env.reset_velocities()
    if setup.home_qpos is not None:
        env.data.qpos[: len(setup.home_qpos)] = setup.home_qpos
    else:
        env.reset_arm_to_home()
    env.forward()


def _place_alone(setup: DomainSetup, obj_name: str, cell: Cell) -> None:
    """Park everyone, then place ``obj_name`` at ``cell``."""
    _reset_state(setup)
    if setup.place_at_cell is not None:
        setup.place_at_cell(setup.env, setup.workspace, obj_name, cell.id)
    else:
        x, y, z = setup.workspace.pose_for(cell)
        setup.env.set_object_pose(obj_name, np.array([x, y, z]))
    setup.env.reset_velocities()
    setup.env.forward()


def _try_pick_with_retries(
    setup: DomainSetup, obj_name: str, cell: Cell,
    retries: int, time_cap_s: float, verbose: bool,
) -> CellResult:
    pos, quat = setup.env.get_object_pose(obj_name)
    half = setup.object_half_extents(obj_name) if setup.object_half_extents \
        else np.asarray(setup.env.get_object_half_size(obj_name))

    t_start = time.perf_counter()
    last_reason = "no attempts"
    for attempt in range(retries):
        np.random.seed(attempt * 1000 + 7)
        if attempt > 0:
            _place_alone(setup, obj_name, cell)
        if verbose:
            q = setup.env.data.qpos[:7].copy()
            print(f"    [attempt {attempt}] qpos[:7] = {q}")
        try:
            if setup.pick_fn is not None:
                ok = setup.pick_fn(
                    obj_name, np.asarray(pos), half, np.asarray(quat),
                )
            else:
                ok = setup.executor.pick(
                    obj_name, np.asarray(pos), half, np.asarray(quat),
                )
        except Exception as exc:
            return CellResult(
                cell=cell, obj_name=obj_name, success=False,
                elapsed_s=time.perf_counter() - t_start,
                reason=f"{type(exc).__name__}: {exc}",
                attempts=attempt + 1,
            )
        elapsed = time.perf_counter() - t_start
        if ok:
            return CellResult(
                cell=cell, obj_name=obj_name, success=True,
                elapsed_s=elapsed,
                reason="OK" if attempt == 0 else f"OK (retry {attempt})",
                attempts=attempt + 1,
            )
        last_reason = "RRT*/IK exhausted"
        if elapsed >= time_cap_s:
            return CellResult(
                cell=cell, obj_name=obj_name, success=False,
                elapsed_s=elapsed,
                reason=f"time cap reached after {attempt + 1} retries",
                attempts=attempt + 1,
            )
    return CellResult(
        cell=cell, obj_name=obj_name, success=False,
        elapsed_s=time.perf_counter() - t_start,
        reason=last_reason, attempts=retries,
    )


def _layout_test_cells(setup: DomainSetup, spec: ReachabilitySpec
                       ) -> List[Tuple[str, Cell]]:
    """Return ``(obj_name, cell)`` pairs for layout-mode testing.

    Resolution order:

    * Explicit ``spec.layout_test_objects`` overrides everything.
    * Otherwise, if the domain has a ``layout_proxy`` body, use that
      single body at every layout cell — appropriate for homogeneous
      domains (Wang's identical cylinders, blocks of one size).
    * Otherwise, exercise each body at its own initial-layout cell —
      the heterogeneous case (Saxena's varied shapes, access's YCB).
    """
    if spec.layout_test_objects is not None:
        out = list(spec.layout_test_objects)
    elif spec.layout_proxy is None:
        out = [(name, cell) for name, cell in setup.initial_layout.items()]
    else:
        out = [(spec.layout_proxy, c) for c in setup.initial_layout.values()]

    # Append extra goal cells, using a representative body.  Prefer the
    # explicit ``layout_proxy``; fall back to the first object whose
    # body the test is going to exercise; final fallback the
    # representative ``ooi`` if present.
    if out:
        fallback_obj = spec.layout_proxy or out[0][0]
    else:
        fallback_obj = (spec.layout_proxy
                        or ("ooi" if "ooi" in setup.object_ids
                            else setup.object_ids[0]))
    seen = {c.id for _, c in out}
    for c in spec.extra_goal_cells:
        if c.id not in seen:
            out.append((fallback_obj, c))
            seen.add(c.id)
    return out


def _full_test_cells(setup: DomainSetup, spec: ReachabilitySpec
                     ) -> List[Tuple[str, Cell]]:
    """Return ``(obj_name, cell)`` pairs sweeping every cell in the
    declared full regions."""
    proxy = spec.full_proxy or spec.layout_proxy or setup.object_ids[0]
    out: List[Tuple[str, Cell]] = []
    for rname in spec.full_regions:
        region = setup.workspace[rname]
        if not isinstance(region, GridRegion):
            continue
        # ``region.cells()`` skips cells the domain has flagged as
        # unreachable — the full sweep tests only reachable cells, so
        # the symbolic planner's domain stays self-consistent with
        # the executability test.
        for cell in region.cells():
            out.append((proxy, cell))
    return out


def run_reachability_check(
    setup: DomainSetup,
    spec: ReachabilitySpec,
    mode: str = "layout",
    retries: int = 5,
    time_cap_s: float = 30.0,
    verbose: bool = False,
) -> DomainResult:
    """Run the executability test against a built ``DomainSetup``.

    Args:
        setup:      Live env + workspace + executor.
        spec:       Per-domain declaration of what to check.
        mode:       ``"layout"`` (test only initial-layout + goal cells —
                    this is the contract) or ``"full"`` (sweep every
                    cell of every full-region — diagnostic).
        retries:    Per-cell retry budget.  Each retry reseeds
                    ``np.random`` so RRT* gets a fresh sampling.
        time_cap_s: Cumulative wall-time cap per cell.  When exceeded
                    we abort that cell and move on (a failure here is
                    *almost always* genuine unreachability — successful
                    picks land in well under 30 s).
        verbose:    Per-cell prints during the run.

    Returns:
        :class:`DomainResult` with one :class:`RegionResult` per region
        that contributed cells to the test.
    """
    if mode not in ("layout", "full"):
        raise ValueError(f"unknown mode {mode!r}; must be 'layout' or 'full'")

    targets = (_layout_test_cells(setup, spec) if mode == "layout"
               else _full_test_cells(setup, spec))

    print(f"\n=== {spec.domain_name} (mode={mode}) ===")
    print(f"  testing {len(targets)} cells "
          f"({retries} retries × {time_cap_s:.0f}s cap each)")

    # Group results by region as they come in.
    cells_by_region: Dict[str, List[CellResult]] = {}
    for obj_name, cell in targets:
        if verbose:
            print(f"  {cell.id} via {obj_name}: ", end="", flush=True)
        _place_alone(setup, obj_name, cell)
        result = _try_pick_with_retries(
            setup, obj_name, cell, retries=retries,
            time_cap_s=time_cap_s, verbose=verbose,
        )
        tag = "PASS" if result.success else "FAIL"
        print(f"  {cell.id:<32} via {obj_name:<14} "
              f"{tag} {result.elapsed_s:5.1f}s  {result.reason}")
        cells_by_region.setdefault(cell.region, []).append(result)

    # Build region summaries.  When in layout mode we may not have
    # touched every cell of a region; cells_x/cells_y still come from
    # the workspace itself.
    regions: List[RegionResult] = []
    for rname, cells in cells_by_region.items():
        region = setup.workspace[rname]
        cells_x = region.cells_x if isinstance(region, GridRegion) else 0
        cells_y = region.cells_y if isinstance(region, GridRegion) else 0
        excluded = (region.excluded_cells
                     if isinstance(region, GridRegion) else frozenset())
        regions.append(RegionResult(
            region_name=rname, cells_x=cells_x, cells_y=cells_y,
            cells=cells, excluded_cells=excluded,
        ))
    return DomainResult(
        domain_name=spec.domain_name, mode=mode, regions=regions,
    )


# ---------------------------------------------------------------------------
# Reporting — matrix / summary
# ---------------------------------------------------------------------------

_PASS_GLYPH = "."
_FAIL_GLYPH = "X"
_UNTESTED_GLYPH = "-"
_EXCLUDED_GLYPH = "#"


def render_matrix(result: DomainResult) -> str:
    """Render a domain's results as one matrix per region."""
    out: List[str] = []
    out.append("")
    out.append("=" * 78)
    out.append(f" {result.domain_name}  ({result.mode} mode)")
    out.append("=" * 78)
    for rr in result.regions:
        rmap: Dict[Tuple[int, int], CellResult] = {
            (c.cell.ix, c.cell.iy): c for c in rr.cells
        }
        out.append("")
        tag = "OK" if rr.all_pass else "FAIL"
        out.append(f"  region {rr.region_name}  ({rr.n_pass}/{rr.n_total} "
                   f"reachable, {tag})")
        if rr.cells_x and rr.cells_y:
            header = "      " + "".join(f"{ix:>3} " for ix in range(rr.cells_x))
            out.append(header)
            for iy in range(rr.cells_y - 1, -1, -1):  # iy=0 (front) at bottom
                row = f"  iy{iy:>2} "
                for ix in range(rr.cells_x):
                    cell_result = rmap.get((ix, iy))
                    if (ix, iy) in rr.excluded_cells:
                        row += f"  {_EXCLUDED_GLYPH} "
                    elif cell_result is None:
                        row += f"  {_UNTESTED_GLYPH} "
                    elif cell_result.success:
                        row += f"  {_PASS_GLYPH} "
                    else:
                        row += f"  {_FAIL_GLYPH} "
                out.append(row)
            out.append(f"   (iy=0 = robot side, '#' = excluded "
                       f"(known unreachable))")
        if not rr.all_pass:
            failing = [c for c in rr.cells if not c.success]
            out.append(f"  FAILS: " + ", ".join(c.cell.id for c in failing[:10])
                       + (f" + {len(failing) - 10} more" if len(failing) > 10 else ""))
    return "\n".join(out)


def summary_line(results: Sequence[DomainResult]) -> str:
    """One-line per-domain summary."""
    out: List[str] = []
    for r in results:
        n_pass = sum(rr.n_pass for rr in r.regions)
        n_total = sum(rr.n_total for rr in r.regions)
        tag = "OK" if r.all_pass else "FAIL"
        out.append(f"  {r.domain_name:<28} {r.mode:<6} "
                   f"{n_pass:>3}/{n_total:<3}  [{tag}]")
    return "\n".join(out)
