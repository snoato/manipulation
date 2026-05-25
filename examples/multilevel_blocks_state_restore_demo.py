"""Visual verification that ``restore_state(on_held='attach')`` works.

Walks a generated PDDL plan from ``data/multilevel_blocks/`` step by
step.  For each action it:

  1. Renders the env BEFORE the action.
  2. Executes the action via the real ``MultilevelBlocksExecutor`` so
     the world state changes physically.
  3. Renders the env AFTER execution.
  4. Reads the resulting symbolic state via ``bridge.ground_state``.
  5. **Resets the env and re-restores from that state via
     ``restore_state(on_held='attach')``.**
  6. Renders the env AFTER restore.
  7. Asserts that block positions and held-block status match between
     (3) and (6).

The point is to catch the previous silent bug where a state containing
a ``(held-cube cube_5)`` fluent restored to a parked cube with an empty
gripper — meaning every subsequent ``put`` from that state ran with no
attached block and trivially "succeeded" while being meaningless.

Outputs (per step):
  ``state_restore_out/step_<N>_<action>_{before,after_exec,after_restore}__{cam}.png``
plus a per-step side-by-side composite for easy visual comparison.

Usage::

  python examples/multilevel_blocks_state_restore_demo.py \\
      --plan data/multilevel_blocks/test/L0/config_2.pddl.plan

Optional ``--max-steps N`` to short-circuit long plans.
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import mujoco
import numpy as np
from PIL import Image

from tampanda.scenes.builder import _look_at_xyaxes
from tampanda.symbolic.domains.multilevel_blocks import (
    MultilevelBlocksConfig,
    MultilevelBlocksExecutor,
    cube_block_name,
    long_block_name,
    make_multilevel_blocks_builder,
    make_multilevel_blocks_bridge,
    oblong_block_name,
    restore_state,
)
from tampanda.planners.rrt_star import RRTStar


_W, _H = 640, 480

_VIZ_DEFAULT = Path("state_restore_out")


# ----------------------------------------------------------------------
# PDDL parsing — extract initial ``(in block cell)`` predicates.
# ----------------------------------------------------------------------


_RX_INIT = re.compile(r"\(:init\s+(.*?)\)\s*\(:goal", re.DOTALL)
_RX_INPRED = re.compile(r"\(in\s+(\w+)\s+(\w+)\)")
_RX_HELD = re.compile(r"\((held-(?:cube|flat-x|flat-y|upright))\s+(\w+)\)")


def parse_initial_state(pddl_path: Path) -> Dict[Tuple, bool]:
    """Return ``{("in", block, cell): True, ...}`` plus any
    ``("held-*", block): True`` entries the PDDL declares in :init.

    We only need these two predicate families — restore_state derives
    everything else from them.
    """
    text = pddl_path.read_text()
    m = _RX_INIT.search(text)
    if m is None:
        raise RuntimeError(f"could not find (:init ...) in {pddl_path}")
    init_body = m.group(1)

    state: Dict[Tuple, bool] = {}
    for block, cell in _RX_INPRED.findall(init_body):
        # Filter out e.g. ``(in-parts ...)`` which is in-parts not in.
        # The regex starts with ``\(in\s`` so already strict, but double-check.
        if block.startswith("cube_") or block.startswith("oblong_") or block.startswith("long_"):
            state[("in", block, cell)] = True
    for fluent, block in _RX_HELD.findall(init_body):
        state[(fluent, block)] = True
    return state


# ----------------------------------------------------------------------
# Plan parsing — extract ``(action arg1 arg2 ...)`` per line.
# ----------------------------------------------------------------------


_RX_PLAN_LINE = re.compile(r"\(([\w-]+)((?:\s+\w+)+)\)")


def parse_plan(plan_path: Path) -> List[Tuple]:
    actions: List[Tuple] = []
    for line in plan_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        m = _RX_PLAN_LINE.match(line)
        if m is None:
            continue
        name = m.group(1)
        args = m.group(2).split()
        actions.append((name, *args))
    return actions


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def _add_cameras(builder, cfg: MultilevelBlocksConfig) -> None:
    """Two views — one looking over the stack table from the parts side,
    one wide-overview from the side."""
    sx, sy, sz = cfg.stack_table_pos
    px, py, pz = cfg.parts_table_pos
    centre = [(sx + px) / 2.0,
              (sy + py) / 2.0,
              sz + 0.27 + 0.15]
    # Over-the-shoulder view from the parts side looking towards stack.
    front_pos = [centre[0] + 0.25, py - 0.40, centre[2] + 0.45]
    side_pos  = [centre[0] + 0.85, centre[1] - 0.05, centre[2] + 0.40]
    builder.add_camera("front", pos=front_pos,
                            xyaxes=_look_at_xyaxes(front_pos, centre),
                            fovy=55.0)
    builder.add_camera("side", pos=side_pos,
                            xyaxes=_look_at_xyaxes(side_pos, centre),
                            fovy=55.0)


def _render_views(env, label: str, out_dir: Path) -> Dict[str, np.ndarray]:
    """Render each camera; save PNGs; return the raw images dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs: Dict[str, np.ndarray] = {}
    for cam_name in ("front", "side"):
        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            continue
        with mujoco.Renderer(env.model, height=_H, width=_W) as renderer:
            renderer.update_scene(env.data, camera=cam_id)
            img = renderer.render()
        out = out_dir / f"{label}__{cam_name}.png"
        Image.fromarray(img).save(out)
        imgs[cam_name] = img
    return imgs


def _compose_step(step_idx: int, action_label: str,
                    triples: Dict[str, Dict[str, np.ndarray]],
                    out_dir: Path) -> None:
    """Build a 2x3 grid for the step: rows=cameras, cols=phases."""
    phases = ("before", "after_exec", "after_restore")
    cams = list(triples["before"])
    if not cams:
        return
    rows: List[np.ndarray] = []
    for cam in cams:
        row = np.concatenate(
            [triples[p][cam] for p in phases], axis=1
        )
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    # Label band
    pad = 24
    canvas = np.full((grid.shape[0] + pad, grid.shape[1], 3),
                          255, dtype=np.uint8)
    canvas[pad:, :, :] = grid
    Image.fromarray(canvas).save(
        out_dir / f"step_{step_idx:02d}_{action_label}_compose.png"
    )


# ----------------------------------------------------------------------
# State extraction for restore — keep only what restore_state reads.
# ----------------------------------------------------------------------


def _minimal_state_from_bridge(bridge, objects: Dict[str, List[str]]) -> Dict[Tuple, bool]:
    """Strip the bridge ground_state down to (in, ...) + held-*."""
    full = bridge.ground_state(objects)
    out: Dict[Tuple, bool] = {}
    for key, val in full.items():
        if not val:
            continue
        if not isinstance(key, tuple):
            continue
        if len(key) >= 1 and key[0] == "in":
            out[key] = True
        elif len(key) >= 1 and isinstance(key[0], str) and key[0].startswith("held-"):
            out[key] = True
    return out


# ----------------------------------------------------------------------
# Verification — compare two world snapshots
# ----------------------------------------------------------------------


def _block_positions(env, block_names: List[str]) -> Dict[str, np.ndarray]:
    out = {}
    for name in block_names:
        pos, _ = env.get_object_pose(name)
        out[name] = np.asarray(pos, dtype=float).copy()
    return out


def _diff_positions(a: Dict[str, np.ndarray],
                     b: Dict[str, np.ndarray],
                     tol: float = 0.02) -> List[Tuple[str, float]]:
    """Report (name, l2) for every block that moved more than tol between
    the two snapshots.  Skips blocks at hide_far_x (parked) for both."""
    out: List[Tuple[str, float]] = []
    for name in sorted(a):
        if name not in b:
            continue
        # Skip if both parked (>20 m).
        if abs(a[name][0]) > 20.0 and abs(b[name][0]) > 20.0:
            continue
        d = float(np.linalg.norm(a[name] - b[name]))
        if d > tol:
            out.append((name, d))
    return out


# ----------------------------------------------------------------------
# Main demo
# ----------------------------------------------------------------------


def _infer_config_from_pddl(pddl_path: Path) -> MultilevelBlocksConfig:
    """Read the (:objects ...) section and count block types.

    Cleaner than guessing — handles any dataset variant.
    """
    text = pddl_path.read_text()
    n_cubes = len(re.findall(r"\bcube_\d+\b", text))
    n_oblong = len(re.findall(r"\boblong_\d+\b", text))
    n_long = len(re.findall(r"\blong_\d+\b", text))
    # The PDDL lists each object multiple times (objects section + every
    # init / goal reference).  Take the max index instead.
    def _max_idx(prefix: str) -> int:
        names = re.findall(rf"\b{prefix}_(\d+)\b", text)
        if not names:
            return 0
        return max(int(n) for n in names) + 1
    return MultilevelBlocksConfig(
        n_cubes=_max_idx("cube"),
        n_oblong=_max_idx("oblong"),
        n_long=_max_idx("long"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path,
                            default=Path("data/multilevel_blocks/test/L0/config_2.pddl.plan"))
    parser.add_argument("--out-dir", type=Path, default=_VIZ_DEFAULT)
    parser.add_argument("--max-steps", type=int, default=4,
                            help="cap action count for fast smoke runs")
    parser.add_argument("--pos-tol", type=float, default=0.02,
                            help="L2 tolerance (m) for block-position match")
    args = parser.parse_args()

    plan_path: Path = args.plan
    problem_path = plan_path.with_suffix("")  # strip .plan -> .pddl
    if not problem_path.exists():
        problem_path = Path(str(plan_path).removesuffix(".plan"))
    if not problem_path.exists():
        print(f"ERROR: cannot find problem file for plan {plan_path}",
                  file=sys.stderr)
        return 2

    print(f"Plan:    {plan_path}")
    print(f"Problem: {problem_path}")
    actions = parse_plan(plan_path)
    print(f"Actions: {len(actions)}")
    if args.max_steps and args.max_steps < len(actions):
        actions = actions[: args.max_steps]
        print(f"  (truncated to first {args.max_steps})")
    for i, a in enumerate(actions):
        print(f"  [{i+1:2d}] {a}")
    print()

    # ---- Build env --------------------------------------------------
    cfg = _infer_config_from_pddl(problem_path)
    print(f"Config: n_cubes={cfg.n_cubes} n_oblong={cfg.n_oblong} "
              f"n_long={cfg.n_long}  (n_blocks={cfg.n_blocks})")

    with tempfile.TemporaryDirectory(prefix="mlb_demo_") as scratch:
        builder, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=Path(scratch), config=cfg,
        )
        _add_cameras(builder, cfg)
        env = builder.build_env(rate=10000.0)

        # Use the REAL executor (not fast) so the renders show actual
        # arm motion — that's what makes the visual check meaningful.
        rrt = RRTStar(env, max_iterations=3000)
        executor = MultilevelBlocksExecutor(env, ws, cfg, motion_planner=rrt)
        bridge, objects = make_multilevel_blocks_bridge(env, ws, cfg, executor=executor)

        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- Initial state ------------------------------------------
        init_state = parse_initial_state(problem_path)
        print(f"\nInitial (in, ...) atoms: "
                  f"{sum(1 for k in init_state if k[0]=='in')}")
        restore_state(env, ws, cfg, init_state,
                            on_held="attach", executor=executor)
        _render_views(env, "step_00_initial", out_dir)

        all_blocks = ([cube_block_name(i) for i in range(cfg.n_cubes)]
                          + [oblong_block_name(i) for i in range(cfg.n_oblong)]
                          + [long_block_name(i) for i in range(cfg.n_long)])

        # ---- Walk the plan ------------------------------------------
        n_pass = 0; n_fail = 0
        for step, action in enumerate(actions, start=1):
            action_label = "_".join(str(x) for x in action).replace("-", "")[:40]
            print(f"\n=== step {step}: {action} ===")

            # 1. before
            before = _render_views(env, f"step_{step:02d}_{action_label}_before", out_dir)
            triples = {"before": before}

            # 2. execute
            #
            # bridge.execute_action returns ``(success: bool, fluent_delta)``.
            # We only run the restore-correctness check on SUCCESS — if
            # the physical executor itself failed, the post-exec world
            # state reflects half-finished motion / physics drift, which
            # the symbolic state can't represent (and isn't expected to).
            success, fluent_delta = bridge.execute_action(
                action[0], *action[1:],
            )
            print(f"  execute -> success={success}  delta={dict(fluent_delta)}")
            if not success:
                print(f"  action {action} failed in physical execution. "
                          f"This is an executor issue, not a restore issue. "
                          f"Stopping demo.")
                break

            # 3. after_exec
            after_exec = _render_views(
                env, f"step_{step:02d}_{action_label}_after_exec", out_dir,
            )
            triples["after_exec"] = after_exec
            pos_after_exec = _block_positions(env, all_blocks)
            ee_after_exec = env.data.site_xpos[executor.ee_site_id].copy()
            held_after_exec = executor._held_block

            # 4. extract minimal state, reset env, restore
            new_state = _minimal_state_from_bridge(bridge, objects)
            n_in = sum(1 for k in new_state if k[0] == "in")
            n_held = sum(1 for k in new_state if isinstance(k[0], str)
                              and k[0].startswith("held-"))
            print(f"  state after exec: in×{n_in}, held×{n_held}")

            restore_state(env, ws, cfg, new_state,
                                on_held="attach", executor=executor)

            # 5. after_restore
            after_restore = _render_views(
                env, f"step_{step:02d}_{action_label}_after_restore", out_dir,
            )
            triples["after_restore"] = after_restore
            pos_after_restore = _block_positions(env, all_blocks)
            ee_after_restore = env.data.site_xpos[executor.ee_site_id].copy()
            held_after_restore = executor._held_block

            # 6. compose
            _compose_step(step, action_label, triples, out_dir)

            # 7. compare
            diffs = _diff_positions(pos_after_exec, pos_after_restore, args.pos_tol)
            ok_held = (held_after_exec == held_after_restore)
            # When a block is held, its post-restore position WILL differ
            # from its post-exec position (it follows the EE; restore
            # parks arm at a HANDOFF config, exec leaves arm at LIFT
            # config).  So drop the held block from the position diff.
            if held_after_exec is not None:
                diffs = [(n, d) for n, d in diffs if n != held_after_exec]

            if diffs:
                print(f"  POSITION MISMATCH: {diffs}")
                n_fail += 1
            elif not ok_held:
                print(f"  HELD MISMATCH: after_exec={held_after_exec} "
                          f"vs after_restore={held_after_restore}")
                n_fail += 1
            else:
                print(f"  OK — positions match within {args.pos_tol*1000:.0f} mm; "
                          f"held={held_after_exec}")
                n_pass += 1

            # Diagnostic: where is the held block relative to the EE in
            # each snapshot?  (Should be roughly equal in magnitude.)
            if held_after_exec is not None:
                d_exec = pos_after_exec[held_after_exec] - ee_after_exec
                d_rest = pos_after_restore[held_after_exec] - ee_after_restore
                print(f"    held offset (world): exec={d_exec.round(4)} "
                          f"restore={d_rest.round(4)}")

        # ---- Summary ------------------------------------------------
        total = n_pass + n_fail
        print(f"\n==============================")
        print(f"RESULT: {n_pass}/{total} steps verified")
        print(f"Outputs: {out_dir.resolve()}")
        return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
