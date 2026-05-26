"""confined_shelf dataset generation — monotone (train) + non-monotone (val).

This domain is a **sorting** task (Wang ICAPS-2022 lazy rearrangement): the
goal is to get every same-colour bottle into one column (colour-by-column),
*not* to retrieve a single object.  The monotone↔non-monotone axis is the
train/val split:

* **monotone** (training) — solvable by moving each bottle at most once,
  straight to its goal cell (``planner.solve`` returns ``monotone=True``);
* **non-monotone** (validation, harder/OOD) — needs at least one bottle
  parked in a buffer cell before its real move (``monotone=False``).

Pipeline per instance (mirrors access19/multilevel_blocks, but the planner
is this domain's own monotone/buffer search since the task is sorting):

1. Sample a random initial layout on the even-x sublattice (so a bottle's
   lateral neighbours are always empty — only front-occlusion blocks, the
   paper's core constraint).  Each even column is then a LIFO stack.
2. Build the canonical colour-by-column goal (``default_color_sort_goal``):
   colour k → the k-th non-adjacent even column, filled front-to-back.
3. ``planner.solve`` finds a verified macro-move plan and classifies it.
4. Route monotone→train, non-monotone→val; skip degenerate (already-sorted)
   and unsolvable layouts.
5. Re-validate the flattened pick/put plan in sim via
   ``check_action_sequence`` (FAST — proven to match FULL on every cell).
6. Write the PDDL problem, the reference plan, a meta line, and a
   matplotlib visualisation (initial | goal grid).

Output layout::

    data/confined_shelf/
    ├── domain.pddl
    ├── train/  (monotone)
    │   ├── config_1.pddl
    │   ├── config_1.pddl.plan
    │   └── config_1.png
    └── val/    (non-monotone)

Usage::

    python -m tampanda.symbolic.domains.confined_shelf.generate_data \\
        --num-train 200 --num-val 50 --output-dir data/confined_shelf
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from tampanda.symbolic.domains.confined_shelf import (
    STAGING_HOME_QPOS, ConfinedShelfConfig, apply_runtime_tweaks,
    default_color_sort_goal, make_confined_shelf_builder,
    make_confined_shelf_bridge,
)
from tampanda.symbolic.domains.confined_shelf.env_builder import excluded_cells
from tampanda.symbolic.domains.confined_shelf.feasibility import (
    check_action_sequence,
)
from tampanda.symbolic.domains.confined_shelf.planner import (
    RearrangePlan, _free_cells, pick_feasible, put_feasible, solve,
)
from tampanda.symbolic.domains.confined_shelf.reachability import _build_executor
from tampanda.symbolic.workspace import Cell, GridRegion, Workspace

_HERE = Path(__file__).parent
_DOMAIN = (_HERE / "pddl" / "domain.pddl").resolve()
_DEFAULT_OUTPUT_DIR = Path("data/confined_shelf")

# PDDL colour constants, indexed by colour-group number.
_COLOR_NAMES = ["red", "green", "yellow", "blue", "purple", "orange"]
# Matplotlib RGBA for each colour group (matches env_builder palette).
_COLOR_RGBA = [
    (0.85, 0.20, 0.20), (0.30, 0.70, 0.30), (0.95, 0.78, 0.20),
    (0.20, 0.45, 0.85), (0.65, 0.30, 0.75), (0.95, 0.55, 0.20),
]


Arr = Dict[str, Tuple[int, int]]


# ---------------------------------------------------------------------------
# Layout / goal helpers
# ---------------------------------------------------------------------------


def _even_cells(cfg: ConfinedShelfConfig) -> List[Tuple[int, int]]:
    """Reachable cells on the even-x sublattice (lateral neighbours empty)."""
    excl = excluded_cells(cfg.n_grid_x, cfg.n_grid_y)
    return [(ix, iy) for ix in range(0, cfg.n_grid_x, 2)
            for iy in range(cfg.n_grid_y) if (ix, iy) not in excl]


def _color_groups(cfg: ConfinedShelfConfig) -> List[int]:
    """Balanced colour assignment: bottle i → colour i mod K."""
    return [i % cfg.n_color_groups for i in range(cfg.n_cylinders)]


def _goal_arr(cfg: ConfinedShelfConfig, groups: List[int]) -> Arr:
    arr: Arr = {}
    for _, cid, cyl in default_color_sort_goal(cfg, groups):
        c = Cell.parse(cid)
        arr[cyl] = (c.ix, c.iy)
    return arr


def _color_set_satisfied(init: Arr, groups: List[int]) -> bool:
    """True iff each colour already sits alone in a single distinct column
    (a degenerate 'already sorted' instance — skip it)."""
    cols_by_color: Dict[int, set] = {}
    for cyl, (ix, _) in init.items():
        g = groups[int(cyl.split("_")[1])]
        cols_by_color.setdefault(g, set()).add(ix)
    if any(len(s) != 1 for s in cols_by_color.values()):
        return False
    cols = [next(iter(s)) for s in cols_by_color.values()]
    return len(cols) == len(set(cols))


def _displace(rng: np.random.Generator, arr: Arr, region: GridRegion,
              cyl: str, goal: Arr) -> bool:
    """Pop ``cyl`` from its current cell to a random free even-x cell, in
    place.  Returns True if it moved.  Only used on bottles currently AT
    their goal cell so the reverse (forward) move is provably feasible."""
    if not pick_feasible(arr, cyl, region):
        return False
    free = [c for c in _free_cells(arr, region)
            if c[0] % 2 == 0 and c != goal[cyl]
            and put_feasible(arr, cyl, c, region)]
    if not free:
        return False
    arr[cyl] = free[int(rng.integers(len(free)))]
    return True


def construct_monotone(rng: np.random.Generator, goal: Arr,
                       region: GridRegion, k_displace: int) -> Arr:
    """Initial layout solvable by moving each bottle at most once to goal.

    Start from the goal and apply ``k_displace`` feasible pop-to-buffer
    moves (each on a bottle still at its goal cell).  Reversing that
    sequence is a feasible one-move-per-bottle plan, so the result is
    monotone by construction (no search needed; solve_monotone recovers
    a canonical plan and confirms it)."""
    arr = dict(goal)
    moved = 0
    while moved < k_displace:
        # Re-scan each round: popping a front bottle exposes the one behind
        # it, so deep bottles become displaceable only after their front
        # neighbours leave.  Each displaced bottle's reverse (forward) move
        # stays feasible, so the layout remains monotone.
        pickable = [c for c in goal if arr[c] == goal[c]
                    and pick_feasible(arr, c, region)]
        rng.shuffle(pickable)
        if not any(_displace(rng, arr, region, c, goal) for c in pickable):
            break
        moved += 1
    return arr


def construct_nonmonotone(rng: np.random.Generator, goal: Arr,
                          region: GridRegion, k_extra: int) -> Optional[Arr]:
    """Initial layout that provably needs a buffer (non-monotone).

    Plant a 2-column front-cell cross-block: the front bottle of colour A's
    goal column is placed at the front of colour B's column and vice versa.
    Neither can go straight to its goal (the other occupies it), so no
    monotone plan exists; one buffer relocation breaks the cycle.  Up to
    ``k_extra`` further bottles are displaced monotone-style for variety."""
    by_col: Dict[int, List[Tuple[int, str]]] = {}
    for cyl, (ix, iy) in goal.items():
        by_col.setdefault(ix, []).append((iy, cyl))
    cols = sorted(by_col)
    if len(cols) < 2:
        return None
    ia, ib = rng.choice(len(cols), size=2, replace=False)
    ca, cb = cols[int(ia)], cols[int(ib)]
    iya, a0 = min(by_col[ca])          # front bottle of column A
    iyb, b0 = min(by_col[cb])          # front bottle of column B
    arr = dict(goal)
    arr[a0] = (cb, iyb)                 # A's front bottle → B's front cell
    arr[b0] = (ca, iya)                 # B's front bottle → A's front cell
    locked = {a0, b0}
    moved = 0
    while moved < k_extra:
        pickable = [c for c in goal if c not in locked and arr[c] == goal[c]
                    and pick_feasible(arr, c, region)]
        rng.shuffle(pickable)
        if not any(_displace(rng, arr, region, c, goal) for c in pickable):
            break
        moved += 1
    return arr


# ---------------------------------------------------------------------------
# PDDL writers
# ---------------------------------------------------------------------------


def _adjacency(region: GridRegion) -> List[str]:
    """Static (adjacent north/east c1 c2) edges within the grid.
    north = +iy (depth, away from robot); east = +ix (robot's right)."""
    out: List[str] = []
    nx, ny = region.cells_x, region.cells_y
    for ix in range(nx):
        for iy in range(ny):
            if region.is_excluded(ix, iy):
                continue
            here = Cell(region.name, ix, iy).id
            if iy + 1 < ny and not region.is_excluded(ix, iy + 1):
                out.append(f"(adjacent north {here} "
                           f"{Cell(region.name, ix, iy + 1).id})")
            if ix + 1 < nx and not region.is_excluded(ix + 1, iy):
                out.append(f"(adjacent east {here} "
                           f"{Cell(region.name, ix + 1, iy).id})")
    return out


def write_pddl_problem(init: Arr, groups: List[int], goal: Arr,
                       region: GridRegion, path: Path, name: str) -> None:
    cyl_names = [f"cyl_{i}" for i in range(len(groups))]
    cell_names = [c.id for c in region.cells()]
    color_names = sorted({_COLOR_NAMES[g] for g in groups})
    occ_cell = {f"{region.name}__{ix}_{iy}": cyl
                for cyl, (ix, iy) in init.items()}

    lines = [f"(define (problem {name})", "  (:domain confined-shelf)",
             "  (:objects",
             f"    {' '.join(cyl_names)} - cylinder",
             f"    {' '.join(cell_names)} - cell",
             f"    {' '.join(color_names)} - color", "  )", "  (:init"]
    for e in _adjacency(region):
        lines.append(f"    {e}")
    for i, g in enumerate(groups):
        lines.append(f"    (color-of cyl_{i} {_COLOR_NAMES[g]})")
    for cyl, (ix, iy) in init.items():
        lines.append(f"    (occupied {region.name}__{ix}_{iy} {cyl})")
    for c in cell_names:
        if c not in occ_cell:
            lines.append(f"    (empty {c})")
    lines.append("    (gripper-empty)")
    lines.append("  )")
    lines.append("  (:goal (and")
    for cyl, (ix, iy) in goal.items():
        lines.append(f"    (occupied {region.name}__{ix}_{iy} {cyl})")
    lines.append("  ))")
    lines.append(")")
    path.write_text("\n".join(lines) + "\n")


def write_plan_file(plan: List[Tuple], path: Path, meta: Dict) -> None:
    lines = ["(" + " ".join(str(a) for a in action) + ")" for action in plan]
    lines.append(f"; cost = {len(plan)} (unit cost)")
    for k, v in meta.items():
        lines.append(f"; {k}: {v}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def render_problem(init: Arr, goal: Arr, groups: List[int],
                   cfg: ConfinedShelfConfig, plan: RearrangePlan,
                   path: Path, title: str) -> None:
    """Top-down grid: initial | goal.  Columns along x, depth (row) along y
    with the open face (row 0, robot side) at the bottom.  Bottles = colour
    circles labelled by index; excluded cells hatched grey."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    excl = excluded_cells(cfg.n_grid_x, cfg.n_grid_y)
    cyl_color = {i: _COLOR_RGBA[groups[i] % len(_COLOR_RGBA)]
                 for i in range(cfg.n_cylinders)}

    fig, axes = plt.subplots(1, 2, figsize=(2 + 0.9 * cfg.n_grid_x, 5))
    for ax, layout, sub in ((axes[0], init, "initial"),
                            (axes[1], goal, "goal")):
        for ix in range(cfg.n_grid_x):
            for iy in range(cfg.n_grid_y):
                excluded = (ix, iy) in excl
                ax.add_patch(Rectangle(
                    (ix - 0.5, iy - 0.5), 1, 1, fill=excluded,
                    facecolor="0.8" if excluded else "none",
                    hatch="xx" if excluded else None,
                    edgecolor="0.6", lw=1))
        for cyl, (ix, iy) in layout.items():
            i = int(cyl.split("_")[1])
            ax.add_patch(Circle((ix, iy), 0.34, color=cyl_color[i],
                                 ec="black", lw=1.2, zorder=3))
            ax.text(ix, iy, str(i), ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold", zorder=4)
        ax.set_xlim(-0.7, cfg.n_grid_x - 0.3)
        ax.set_ylim(-0.9, cfg.n_grid_y - 0.3)
        ax.set_xticks(range(cfg.n_grid_x))
        ax.set_yticks(range(cfg.n_grid_y))
        ax.set_xlabel("column (ix)")
        ax.set_ylabel("depth (iy) — row 0 = open face")
        ax.set_aspect("equal")
        ax.set_title(sub)
        ax.text(0.5, -0.8, "↑ front / robot side ↑",
                transform=ax.transData, ha="left", fontsize=8, color="0.4")

    moves = " ".join(f"{c}:{f}->{t}" for c, f, t in plan.moves)
    fig.suptitle(f"{title}\n{moves}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generation driver
# ---------------------------------------------------------------------------


def _validate(env, ws, cfg, executor, init: Arr, plan: RearrangePlan,
              groups: List[int], fast: bool = True) -> bool:
    """Replay the flattened pick/put plan in sim.  ``fast`` selects the
    cheap geometric column-pose check; ``fast=False`` runs the full RRT*
    executor + physics settle (slower, but the true ground truth — FAST is
    lenient at the front-row adjacent case)."""
    cyl_names = [f"cyl_{i}" for i in range(cfg.n_cylinders)]
    state = {("occupied", Cell("shelf_interior", ix, iy).id, cyl): True
             for cyl, (ix, iy) in init.items()}
    actions = plan.to_actions("shelf_interior")
    res = check_action_sequence(env, ws, cfg, state, actions, cyl_names,
                                executor=executor, fast=fast,
                                home_qpos=STAGING_HOME_QPOS,
                                short_circuit=True)
    return bool(res["success"])


def _emit(split: str, num: int, init: Arr, goal: Arr, groups: List[int],
          cfg: ConfinedShelfConfig, region: GridRegion, plan: RearrangePlan,
          output_dir: Path, n_cyl: int, n_col: int) -> None:
    actions = plan.to_actions("shelf_interior")
    stem = output_dir / split / f"config_{num}"
    write_pddl_problem(init, groups, goal, region, stem.with_suffix(".pddl"),
                       name=f"confined-shelf-{split}-{num}")
    write_plan_file(actions, Path(str(stem) + ".pddl.plan"), meta={
        "monotone": plan.monotone, "n_moves": len(plan.moves),
        "n_relocations": plan.n_relocations,
        "n_cylinders": n_cyl, "n_colors": n_col, "split": split,
    })
    render_problem(init, goal, groups, cfg, plan, stem.with_suffix(".png"),
                   title=f"{split} #{num} — "
                   f"{'MONOTONE' if plan.monotone else 'NON-MONOTONE'} "
                   f"({len(plan.moves)} moves, {plan.n_relocations} buffers)")


def _pick_groups(rng: np.random.Generator, n_cyl: int) -> List[int]:
    """Per-instance colour assignment for ``n_cyl`` bottles.

    At least 2 colours (a meaningful sort), at most 4 (goal columns 0,2,4,6),
    and balanced so each colour gets at most 4 bottles — the depth of a goal
    column — so a colour never spills into a second column (which would break
    'same colour, same column')."""
    import math
    lo = max(2, math.ceil(n_cyl / 4))
    hi = min(4, n_cyl)
    lo = min(lo, hi)
    k = int(rng.integers(lo, hi + 1))
    return [i % k for i in range(n_cyl)]


def _build_env(scratch_dir: Path, n_cyl: int):
    """Build env + executor + bridge for an ``n_cyl``-bottle scene.  Colours
    here are cosmetic (feasibility is geometric); the per-instance colour
    assignment is applied at emit time."""
    cfg = ConfinedShelfConfig(n_cylinders=n_cyl,
                              n_color_groups=min(4, max(2, (n_cyl + 3) // 4)))
    b, ws, cfg = make_confined_shelf_builder(scratch_dir, cfg)
    env = b.build_env(rate=10000.0)
    apply_runtime_tweaks(env, cfg)
    tz = ws["shelf_interior"].level_z - cfg.cylinder_half_height
    ex = _build_executor(env, tz)
    make_confined_shelf_bridge(
        env, ws, cfg,
        [_COLOR_NAMES[i % cfg.n_color_groups] for i in range(n_cyl)],
        executor=ex)
    return env, ws, cfg, ex


def generate(num_train: int, num_val: int, output_dir: Path, *,
             mono_n_cyls: List[int], nonmono_n_cyl: int, seed: int,
             buffer_budget: int, validate: bool, validate_fast: bool = True,
             max_attempts_factor: int = 40) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    if _DOMAIN.exists():
        shutil.copy(_DOMAIN, output_dir / "domain.pddl")
    (output_dir / "train").mkdir(exist_ok=True)
    (output_dir / "val").mkdir(exist_ok=True)

    stats: Counter = Counter()
    plan_lens: Dict[str, List[int]] = {"train": [], "val": []}
    n_cyl_hist: Counter = Counter()
    counters = {"train": 0, "val": 0}
    seen: set = set()
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="cs_data_") as td:
        env_cache: Dict[int, tuple] = {}

        def get_env(n_cyl: int):
            if n_cyl not in env_cache:
                sd = Path(td) / f"n{n_cyl}"
                sd.mkdir(parents=True, exist_ok=True)
                env_cache[n_cyl] = _build_env(sd, n_cyl)
            return env_cache[n_cyl]

        def attempt(split: str, n_cyl: int, want_monotone: bool,
                    make_init) -> bool:
            env, ws, cfg, ex = get_env(n_cyl)
            region = ws["shelf_interior"]
            groups = _pick_groups(rng, n_cyl)
            goal = _goal_arr(cfg, groups)
            init = make_init(goal, region, n_cyl)
            if init is None:
                stats[f"{split}_no_init"] += 1
                return False
            key = (n_cyl, frozenset(init.items()), frozenset(goal.items()))
            if key in seen or _color_set_satisfied(init, groups):
                stats[f"{split}_dup_or_degenerate"] += 1
                return False
            plan = solve(init, goal, region, buffer_budget=buffer_budget)
            if plan is None or plan.monotone != want_monotone:
                stats[f"{split}_wrong_class"] += 1
                return False
            if validate and not _validate(env, ws, cfg, ex, init, plan,
                                          groups, fast=validate_fast):
                stats[f"{split}_validate_failed"] += 1
                return False
            seen.add(key)
            counters[split] += 1
            n_cyl_hist[n_cyl] += 1
            plan_lens[split].append(len(plan.moves))
            _emit(split, counters[split], init, goal, groups, cfg, region,
                  plan, output_dir, n_cyl, len(set(groups)))
            return True

        # ---- Monotone (train): spread over the requested bottle counts -----
        alloc = {n: num_train // len(mono_n_cyls) for n in mono_n_cyls}
        for i in range(num_train % len(mono_n_cyls)):
            alloc[mono_n_cyls[i]] += 1
        for n_cyl in mono_n_cyls:
            cnt = alloc[n_cyl]
            got = attempts = 0
            maxa = max(cnt, 1) * max_attempts_factor
            while got < cnt and attempts < maxa:
                attempts += 1
                if attempt("train", n_cyl, True,
                           lambda g, r, n: construct_monotone(
                               rng, g, r, int(rng.integers(2, n + 1)))):
                    got += 1
            print(f"  [train] n_cyl={n_cyl}: {got}/{cnt}  "
                  f"(total {counters['train']}/{num_train}, "
                  f"{time.time()-t0:.0f}s)", flush=True)

        # ---- Non-monotone (val) at the requested bottle count --------------
        got = attempts = 0
        maxa = max(num_val, 1) * max_attempts_factor
        while got < num_val and attempts < maxa:
            attempts += 1
            if attempt("val", nonmono_n_cyl, False,
                       lambda g, r, n: construct_nonmonotone(
                           rng, g, r, int(rng.integers(0, max(1, n - 1))))):
                got += 1
                if got % 10 == 0 or got == 1:
                    print(f"  [val] {got}/{num_val}  ({time.time()-t0:.0f}s)",
                          flush=True)

    print(f"\n=== done in {time.time() - t0:.0f}s ===")
    for split, kind in (("train", "monotone"), ("val", "non-monotone")):
        pl = plan_lens[split]
        if pl:
            print(f"  {split} ({kind}): {len(pl)} problems, "
                  f"moves {min(pl)}-{max(pl)} mean={np.mean(pl):.1f}")
        else:
            print(f"  {split} ({kind}): 0 problems")
    print(f"  bottle-count histogram: {dict(sorted(n_cyl_hist.items()))}")
    print(f"  rejections: {dict(stats)}")
    print(f"  output: {output_dir}")


def _split_ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate confined_shelf monotone(train)+non-monotone(val) "
                    "color-by-column sorting dataset")
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-train", type=int, default=350)
    p.add_argument("--num-val", type=int, default=50)
    p.add_argument("--mono-n-cyls", type=_split_ints, default=[3, 4, 5, 6, 7, 8, 9],
                   help="comma-separated bottle counts to spread monotone "
                        "training instances over")
    p.add_argument("--nonmono-n-cyl", type=int, default=10,
                   help="bottle count for the non-monotone validation set")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--buffer-budget", type=int, default=20000)
    p.add_argument("--no-validate", action="store_true",
                   help="skip the sim replay (symbolic solve only)")
    p.add_argument("--full-validate", action="store_true",
                   help="validate each plan with the FULL RRT* executor + "
                        "physics (slower, true ground truth) instead of FAST")
    args = p.parse_args()
    generate(args.num_train, args.num_val, args.output_dir,
             mono_n_cyls=args.mono_n_cyls, nonmono_n_cyl=args.nonmono_n_cyl,
             seed=args.seed, buffer_budget=args.buffer_budget,
             validate=not args.no_validate,
             validate_fast=not args.full_validate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
