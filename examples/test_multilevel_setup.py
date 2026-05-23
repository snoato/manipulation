"""Interactive testing tool for the multilevel_blocks env.

Spawns a MuJoCo passive viewer + a CLI prompt.  Build an env with
chosen table positions / sizes, place objects at specified cells,
then issue pick / put / status commands at the prompt to verify
whether the motion planner + executor can complete each action.
The viewer reflects every state change live.

Run with ``mjpython`` on macOS — the passive viewer requires a Cocoa
main thread.

Usage::

    # default geometry, drop into interactive shell (viewer + prompt)
    mjpython examples/test_multilevel_setup.py

    # custom table positions
    mjpython examples/test_multilevel_setup.py \\
        --stack-pos 0 0.55 0.13 \\
        --parts-pos 0 -0.45 0

    # start with some initial placements + run a scripted sequence
    mjpython examples/test_multilevel_setup.py \\
        --place cube_0 parts__7_7 \\
        --script "pick cube_0; put-cube cube_0 stack_L0__5_5; status"

    # run a script without launching the viewer (CI / headless)
    python examples/test_multilevel_setup.py --no-viewer \\
        --place cube_0 parts__7_7 \\
        --script "pick cube_0; put-cube cube_0 stack_L0__5_5; status"

Interactive commands::

    help                            list commands
    status                          print all block + EE poses
    place  <block> <cell>           teleport block to cell centre
    place-flat-x <block> <c1> <c2>  teleport oblong spanning c1, c2 along x
    place-flat-y <block> <c1> <c2>  teleport oblong spanning c1, c2 along y
    place-long-x <block> <c1> <c2> <c3>   teleport 3×1 along x
    place-long-y <block> <c1> <c2> <c3>   teleport 3×1 along y
    pick   <block>                  pick the block (auto-detects shape)
    put-cube <block> <cell>
    put-flat-x <block> <c1> <c2>
    put-flat-y <block> <c1> <c2>
    put-upright <block> <c-low> <c-high>
    put-long-x <block> <c1> <c2> <c3>
    put-long-y <block> <c1> <c2> <c3>
    put-long-upright <block> <c-low> <c-mid> <c-high>
    make-upright-from-x <block>
    make-upright-from-y <block>
    make-flat-x-from-upright <block>
    make-flat-y-from-upright <block>
    turn-x-to-y <block>
    turn-y-to-x <block>
    make-long-upright-from-x <block>
    make-long-upright-from-y <block>
    make-long-flat-x-from-upright <block>
    make-long-flat-y-from-upright <block>
    turn-long-x-to-y <block>
    turn-long-y-to-x <block>
    render [path]                   save a snapshot to /tmp/setup_snap.png by default
    blocks                          list available block names
    cells                           list cell-name conventions
    quit / exit / Ctrl-D            leave the prompt
"""
from __future__ import annotations

import argparse
import atexit
import os
import shlex
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Enable up-arrow history + Ctrl-R reverse-search + line editing at the
# ``input()`` prompt.  On macOS the standard ``readline`` module is
# linked against ``libedit`` (BSD's readline-compatible replacement),
# which works fine with the same import.  Persists across sessions via
# a history file in ~/.cache.
try:
    import readline
    _HISTORY_FILE = Path.home() / ".cache" / "tampanda_test_multilevel.history"
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(_HISTORY_FILE)
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(2000)
    atexit.register(
        lambda: readline.write_history_file(str(_HISTORY_FILE)))
except ImportError:
    pass

import mujoco
import mujoco.viewer
import numpy as np

# Allow running from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tampanda.symbolic.domains.multilevel_blocks import (
    MultilevelBlocksConfig,
    MultilevelBlocksExecutor,
    cube_block_name,
    long_block_name,
    make_multilevel_blocks_bridge,
    make_multilevel_blocks_builder,
    oblong_block_name,
    stack_region_name,
)
from tampanda.symbolic.workspace import Cell


_QUAT_FLAT_X = np.array([1.0, 0.0, 0.0, 0.0])
_QUAT_FLAT_Y = np.array([0.7071068, 0.0, 0.0, 0.7071068])


class Session:
    """Live env + executor + bridge, plus convenience helpers."""

    def __init__(self, args):
        # Build config with overrides
        kwargs = dict(
            n_cubes=args.n_cubes,
            n_oblong=args.n_oblong,
            n_long=args.n_long,
            stack_table_pos=tuple(args.stack_pos),
            parts_table_pos=tuple(args.parts_pos),
        )
        # Allow overriding grid dims for quick experiments
        if args.stack_cells is not None:
            kwargs["stack_grid_cells"] = tuple(args.stack_cells)
        if args.parts_cells is not None:
            kwargs["parts_grid_cells"] = tuple(args.parts_cells)
        if args.cube_size is not None:
            kwargs["cube_half_extent"] = args.cube_size / 2

        self.cfg = MultilevelBlocksConfig(**kwargs)
        self.scratch = tempfile.TemporaryDirectory(
            prefix="test_multilevel_setup_")
        builder, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=Path(self.scratch.name), config=self.cfg)
        self.cfg = cfg
        self.ws = ws
        self.env = builder.build_env(rate=10000.0)

        # Park all blocks far away initially.  Spread along +x so the
        # parked blocks don't overlap each other (parallel-jaw fingers
        # collide with the overlap geom during transit otherwise).
        for i, name in enumerate(self._all_block_names()):
            parked = np.array([cfg.hide_far_x + 0.15 * i, 0.0, 0.05])
            self.env.set_object_pose(name, parked)
        self.env.reset_velocities()
        self.env.forward()

        from tampanda.planners.rrt_star import RRTStar
        rrt = RRTStar(self.env, max_iterations=3000)
        self.executor = MultilevelBlocksExecutor(
            self.env, self.ws, self.cfg, motion_planner=rrt,
            verbose=args.verbose,
        )
        self.bridge, self.objects = make_multilevel_blocks_bridge(
            self.env, self.ws, self.cfg, executor=self.executor)

        # Initialise the arm at the executor's neutral HOME pose so the
        # first pick / put doesn't fail trying to lerp out of an
        # all-zeros joint config that may pass through colliding
        # midpoints.
        from tampanda.symbolic.domains.multilevel_blocks.executor import (
            _HOME_NEUTRAL_Q,
        )
        self.env.data.qpos[:7] = _HOME_NEUTRAL_Q
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.reset_velocities()
        self.env.forward()

        # Pre-render a renderer for snapshots.
        self.env.model.vis.global_.offwidth = max(
            1280, self.env.model.vis.global_.offwidth)
        self.env.model.vis.global_.offheight = max(
            720, self.env.model.vis.global_.offheight)
        self._renderer = mujoco.Renderer(self.env.model, height=720, width=1280)
        self._cam = mujoco.MjvCamera()
        self._cam.azimuth = 35
        self._cam.elevation = -28
        self._cam.distance = 2.4
        sp = cfg.stack_table_pos
        self._cam.lookat[:] = [0.0, (sp[1] + cfg.parts_table_pos[1]) / 2, 0.35]

    def __del__(self):
        try:
            self._renderer.close()
        except Exception:
            pass
        try:
            self.scratch.cleanup()
        except Exception:
            pass

    # ---- helpers ----------------------------------------------------

    def _all_block_names(self) -> List[str]:
        names = [cube_block_name(i) for i in range(self.cfg.n_cubes)]
        names += [oblong_block_name(i) for i in range(self.cfg.n_oblong)]
        names += [long_block_name(i) for i in range(self.cfg.n_long)]
        return names

    def _shape_of(self, name: str) -> str:
        if name.startswith("cube_"):
            return "cube"
        if name.startswith("long_"):
            return "long"
        return "oblong"

    def _block_pos(self, name: str) -> np.ndarray:
        return np.asarray(self.env.get_object_pose(name)[0])

    def _ee_pos(self) -> np.ndarray:
        return self.env.data.site_xpos[self.executor.ee_site_id].copy()

    def _cell(self, cid: str) -> Cell:
        # Typo-tolerant: accept ``parts_7_7`` as shorthand for the canonical
        # ``parts__7_7`` (single underscore as separator), and similarly for
        # ``stack_L0_7_7`` → ``stack_L0__7_7``.
        if "__" not in cid:
            # Try to interpret as ``<region>_<ix>_<iy>``.
            parts = cid.rsplit("_", 2)
            if len(parts) == 3:
                cid = f"{parts[0]}__{parts[1]}_{parts[2]}"
        return self.executor._parse_cell(cid)

    def _cell_world(self, cid: str) -> np.ndarray:
        return np.asarray(self.ws.pose_for(self._cell(cid)))

    def _detect_block_cells(self, name: str) -> List[Cell]:
        """Use the bridge's grounding to figure out which cells the block
        currently occupies."""
        from tampanda.symbolic.domains.multilevel_blocks.bridge import (
            _block_cells,
        )
        return _block_cells(self.env, self.ws, name,
                                self._shape_of(name),
                                self.cfg.cube_half_extent)

    # ---- commands ---------------------------------------------------

    def cmd_status(self, args: Sequence[str]) -> None:
        ee = self._ee_pos()
        print(f"  EE site: {ee}")
        print(f"  Gripper q: {self.env.data.qpos[7:9]}")
        print("  Blocks:")
        for n in self._all_block_names():
            p = self._block_pos(n)
            cells = self._detect_block_cells(n)
            tag = ", ".join(c.id for c in cells) if cells else "(parked/free)"
            print(f"    {n:<12}  pos={p}   cells={tag}")
        held = []
        for n in self._all_block_names():
            for pred in ("held-cube", "held-flat-x", "held-flat-y",
                            "held-upright"):
                if self.executor.env._attached == n:
                    held.append(f"{n} (attached)")
                    break
        if held:
            print(f"  Held: {', '.join(held)}")
        else:
            ge = self.executor._fluent_state.get(("gripper-empty",), True) \
                if hasattr(self.executor, "_fluent_state") else None
            print("  Held: nothing")

    def _sync_viewer(self) -> None:
        v = getattr(self.env, "viewer", None)
        if v is not None:
            v.sync()

    def cmd_place(self, args: Sequence[str]) -> None:
        if len(args) != 2:
            print("usage: place <block> <cell>")
            return
        name, cell_id = args
        cell = self._cell(cell_id)
        xyz = self.ws.pose_for(cell)
        self.env.set_object_pose(name, np.asarray(xyz), _QUAT_FLAT_X)
        self.env.reset_velocities()
        self.env.forward()
        self._sync_viewer()
        print(f"  placed {name} at {cell_id} → {xyz}")

    def cmd_place_flat_x(self, args: Sequence[str]) -> None:
        if len(args) != 3:
            print("usage: place-flat-x <block> <c1> <c2>  (c2 must be east of c1)")
            return
        name, c1, c2 = args
        p1 = self._cell_world(c1)
        p2 = self._cell_world(c2)
        centre = (p1 + p2) / 2
        self.env.set_object_pose(name, centre, _QUAT_FLAT_X)
        self.env.reset_velocities()
        self.env.forward()
        self._sync_viewer()
        print(f"  placed {name} flat-x at midpoint of {c1}/{c2} → {centre}")

    def cmd_place_flat_y(self, args: Sequence[str]) -> None:
        if len(args) != 3:
            print("usage: place-flat-y <block> <c1> <c2>  (c2 must be north of c1)")
            return
        name, c1, c2 = args
        p1 = self._cell_world(c1)
        p2 = self._cell_world(c2)
        centre = (p1 + p2) / 2
        self.env.set_object_pose(name, centre, _QUAT_FLAT_Y)
        self.env.reset_velocities()
        self.env.forward()
        self._sync_viewer()
        print(f"  placed {name} flat-y at midpoint of {c1}/{c2} → {centre}")

    def cmd_place_long_x(self, args: Sequence[str]) -> None:
        if len(args) != 4:
            print("usage: place-long-x <block> <c1> <c2> <c3>  "
                      "(c1→c2→c3 chain east along x)")
            return
        name, c1, _c2, c3 = args
        p1 = self._cell_world(c1)
        p3 = self._cell_world(c3)
        centre = (p1 + p3) / 2
        self.env.set_object_pose(name, centre, _QUAT_FLAT_X)
        self.env.reset_velocities()
        self.env.forward()
        self._sync_viewer()
        print(f"  placed {name} long-x at midpoint of {c1}..{c3} → {centre}")

    def cmd_place_long_y(self, args: Sequence[str]) -> None:
        if len(args) != 4:
            print("usage: place-long-y <block> <c1> <c2> <c3>  "
                      "(c1→c2→c3 chain north along y)")
            return
        name, c1, _c2, c3 = args
        p1 = self._cell_world(c1)
        p3 = self._cell_world(c3)
        centre = (p1 + p3) / 2
        self.env.set_object_pose(name, centre, _QUAT_FLAT_Y)
        self.env.reset_velocities()
        self.env.forward()
        self._sync_viewer()
        print(f"  placed {name} long-y at midpoint of {c1}..{c3} → {centre}")

    def _exec(self, action: str, *params: str) -> None:
        try:
            ok, delta = self.bridge.execute_action(
                action, *params, objects=self.objects)
            tag = "OK" if ok else "FAIL"
            print(f"  -> {tag}; fluent_delta={delta}")
        except Exception as e:
            print(f"  CRASHED: {type(e).__name__}: {e}")
            traceback.print_exc()
        # Always re-sync viewer at the end of an action so the final
        # state is visible even if motion completed in between syncs.
        self._sync_viewer()

    def cmd_pick(self, args: Sequence[str]) -> None:
        """Auto-detect pick action from the block's current cells."""
        if len(args) != 1:
            print("usage: pick <block>")
            return
        name = args[0]
        cells = self._detect_block_cells(name)
        if not cells:
            print(f"  {name}: not in any cell (parked?).  Place it first.")
            return
        shape = self._shape_of(name)
        if shape == "cube":
            print(f"  pick-cube({name}, {cells[0].id})")
            self._exec("pick-cube", name, cells[0].id)
            return
        if shape == "long":
            if len(cells) != 3:
                print(f"  {name}: long should occupy exactly 3 cells, "
                          f"got {len(cells)}.")
                return
            # Sort cells along whichever axis varies.
            same_region = all(c.region == cells[0].region for c in cells)
            if same_region and all(c.iy == cells[0].iy for c in cells):
                ordered = sorted(cells, key=lambda c: c.ix)
                if (ordered[1].ix - ordered[0].ix == 1
                        and ordered[2].ix - ordered[1].ix == 1):
                    print(f"  pick-long-x({name}, {ordered[0].id}, "
                              f"{ordered[1].id}, {ordered[2].id})")
                    self._exec("pick-long-x", name,
                                  ordered[0].id, ordered[1].id, ordered[2].id)
                    return
            if same_region and all(c.ix == cells[0].ix for c in cells):
                ordered = sorted(cells, key=lambda c: c.iy)
                if (ordered[1].iy - ordered[0].iy == 1
                        and ordered[2].iy - ordered[1].iy == 1):
                    print(f"  pick-long-y({name}, {ordered[0].id}, "
                              f"{ordered[1].id}, {ordered[2].id})")
                    self._exec("pick-long-y", name,
                                  ordered[0].id, ordered[1].id, ordered[2].id)
                    return
            # Vertical chain across 3 stack levels → upright.
            if (all(c.region.startswith("stack_L") for c in cells)
                    and all(c.ix == cells[0].ix and c.iy == cells[0].iy
                                for c in cells)):
                ordered = sorted(cells,
                                       key=lambda c: int(c.region.split("_L")[1]))
                lvls = [int(c.region.split("_L")[1]) for c in ordered]
                if lvls[1] - lvls[0] == 1 and lvls[2] - lvls[1] == 1:
                    print(f"  pick-long-upright({name}, {ordered[0].id}, "
                              f"{ordered[1].id}, {ordered[2].id})")
                    self._exec("pick-long-upright", name,
                                  ordered[0].id, ordered[1].id, ordered[2].id)
                    return
            print(f"  {name}: long cells {[c.id for c in cells]} don't form "
                      f"a contiguous chain")
            return

        # oblong: figure out orientation from cells
        if len(cells) != 2:
            print(f"  {name}: oblong should occupy exactly 2 cells, got {len(cells)}.")
            return
        c1, c2 = cells
        if c1.region == c2.region:
            # same level — flat
            if c1.iy == c2.iy and abs(c1.ix - c2.ix) == 1:
                west, east = (c1, c2) if c1.ix < c2.ix else (c2, c1)
                print(f"  pick-flat-x({name}, {west.id}, {east.id})")
                self._exec("pick-flat-x", name, west.id, east.id)
            elif c1.ix == c2.ix and abs(c1.iy - c2.iy) == 1:
                south, north = (c1, c2) if c1.iy < c2.iy else (c2, c1)
                print(f"  pick-flat-y({name}, {south.id}, {north.id})")
                self._exec("pick-flat-y", name, south.id, north.id)
            else:
                print(f"  {name}: cells {c1.id} and {c2.id} aren't adjacent")
        elif c1.region.startswith("stack_L") and c2.region.startswith("stack_L"):
            # different levels — upright
            lvl1 = int(c1.region.split("_L")[1])
            lvl2 = int(c2.region.split("_L")[1])
            if c1.ix == c2.ix and c1.iy == c2.iy and abs(lvl1 - lvl2) == 1:
                low, high = (c1, c2) if lvl1 < lvl2 else (c2, c1)
                print(f"  pick-upright({name}, {low.id}, {high.id})")
                self._exec("pick-upright", name, low.id, high.id)
            else:
                print(f"  {name}: cells {c1.id} and {c2.id} aren't a vertical pair")
        else:
            print(f"  {name}: cells {c1.id} and {c2.id} don't form a valid oblong pose")

    def cmd_put_cube(self, args: Sequence[str]) -> None:
        if len(args) != 2:
            print("usage: put-cube <block> <cell>")
            return
        self._exec("put-cube", *args)

    def cmd_put_flat_x(self, args: Sequence[str]) -> None:
        if len(args) != 3:
            print("usage: put-flat-x <block> <c1> <c2>")
            return
        self._exec("put-flat-x", *args)

    def cmd_put_flat_y(self, args: Sequence[str]) -> None:
        if len(args) != 3:
            print("usage: put-flat-y <block> <c1> <c2>")
            return
        self._exec("put-flat-y", *args)

    def cmd_put_upright(self, args: Sequence[str]) -> None:
        if len(args) != 3:
            print("usage: put-upright <block> <c-low> <c-high>")
            return
        self._exec("put-upright", *args)

    def cmd_put_long_x(self, args: Sequence[str]) -> None:
        if len(args) != 4:
            print("usage: put-long-x <block> <c1> <c2> <c3>")
            return
        self._exec("put-long-x", *args)

    def cmd_put_long_y(self, args: Sequence[str]) -> None:
        if len(args) != 4:
            print("usage: put-long-y <block> <c1> <c2> <c3>")
            return
        self._exec("put-long-y", *args)

    def cmd_put_long_upright(self, args: Sequence[str]) -> None:
        if len(args) != 4:
            print("usage: put-long-upright <block> <c-low> <c-mid> <c-high>")
            return
        self._exec("put-long-upright", *args)

    def cmd_transform(self, action: str, args: Sequence[str]) -> None:
        if len(args) != 1:
            print(f"usage: {action} <block>")
            return
        self._exec(action, args[0])

    def cmd_move_table(self, args: Sequence[str]) -> None:
        """Move one of the two tables to a new world position at runtime.

        Updates BOTH the physical body pose AND the workspace region
        origins / level_z so cell IDs like ``stack_L0__9_9`` resolve to
        the cell of the new table position.  Subsequent pick/put commands
        therefore target the moved table correctly.

        usage: move-table <stack|parts> <x> <y> <z>
        """
        if len(args) != 4:
            print("usage: move-table <stack|parts> <x> <y> <z>")
            return
        which = args[0]
        if which not in ("stack", "parts"):
            print(f"  unknown table {which!r}; expected 'stack' or 'parts'")
            return
        body_name = f"{which}_table"
        try:
            new_pos = np.array([float(args[1]), float(args[2]),
                                  float(args[3])])
        except ValueError:
            print("  could not parse xyz")
            return

        # Detect a change in table z.  Leg length is baked into the
        # compiled XML, so any z change requires regenerating the table
        # XML + reloading the scene — a simple set_object_pose would
        # leave the legs floating (or sunk) at the new height.
        old_z = (self.cfg.stack_table_pos[2] if which == "stack"
                     else self.cfg.parts_table_pos[2])
        z_changed = abs(float(new_pos[2]) - float(old_z)) > 1e-6

        if z_changed:
            self._rebuild_for_table_move(which, new_pos)
            return

        # 1. Teleport the physical body (xy-only move).
        self.env.set_object_pose(body_name, new_pos)
        self.env.reset_velocities()
        self.env.forward()
        self._sync_viewer()

        # 2. Re-compute the workspace's region origins + level_z so cell
        #    IDs map to the NEW table position.  Mirror of the geometry
        #    logic in ``make_multilevel_blocks_builder``.
        cube_size = self.cfg.cube_size
        table_top_local_z = 0.27   # body-local z of the surface geom
        sx, sy, sz = new_pos
        if which == "stack":
            cells_x, cells_y, cells_z = self.cfg.stack_grid_cells
            extent = (cells_x * cube_size, cells_y * cube_size)
            origin = (sx - extent[0] / 2, sy - extent[1] / 2)
            table_top_world_z = sz + table_top_local_z
            for level in range(cells_z):
                region_name = stack_region_name(level)
                region = self.ws[region_name]
                region.origin = origin
                region.extent = extent
                region.level_z = (table_top_world_z
                                       + (level + 0.5) * cube_size)
        else:  # parts
            cells_x, cells_y = self.cfg.parts_grid_cells
            extent = (cells_x * cube_size, cells_y * cube_size)
            origin = (sx - extent[0] / 2, sy - extent[1] / 2)
            table_top_world_z = sz + table_top_local_z
            region = self.ws["parts"]
            region.origin = origin
            region.extent = extent
            region.level_z = table_top_world_z + 0.5 * cube_size

        # 3. Update the config copy too (kept consistent for any code
        #    that reads cfg.stack_table_pos / parts_table_pos at runtime).
        #    MultilevelBlocksConfig is a frozen dataclass, so we use
        #    object.__setattr__ to bypass the freeze for this in-place
        #    update.
        if which == "stack":
            object.__setattr__(self.cfg, "stack_table_pos",
                                  (float(sx), float(sy), float(sz)))
        else:
            object.__setattr__(self.cfg, "parts_table_pos",
                                  (float(sx), float(sy), float(sz)))

        # 4. If the executor caches anything position-dependent
        #    (currently just the hand-off q-configs), recompute them so
        #    transit moves still find the right poses.
        self.executor._precompute_handoffs()

        print(f"  moved {body_name} to {new_pos}")
        print(f"  workspace cell origins recomputed; cell IDs now map "
               "to new table position")

    def _rebuild_for_table_move(self, which: str,
                                       new_pos: np.ndarray) -> None:
        """Regenerate the scene when a table z changes (legs are baked
        into the XML).  Preserves robot + block state across the reload.
        """
        from tampanda.scenes.reloader import SceneReloader
        from tampanda.planners.rrt_star import RRTStar
        from tampanda.symbolic.domains.multilevel_blocks.executor import (
            _HOME_NEUTRAL_Q,
        )

        # 1. Update cfg with the new table position so the next builder
        #    call regenerates the table XML with the right leg length.
        new_tuple = (float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
        if which == "stack":
            object.__setattr__(self.cfg, "stack_table_pos", new_tuple)
        else:
            object.__setattr__(self.cfg, "parts_table_pos", new_tuple)

        # 2. Snapshot dynamic state.  Drop both table bodies — their
        #    positions are determined by the new XML / add_object pos.
        reloader = SceneReloader()
        state = reloader.snapshot(self.env)
        for tname in ("stack_table", "parts_table"):
            state["objects"].pop(tname, None)

        # 3. Close the current viewer so the new env can launch its own.
        v = getattr(self.env, "viewer", None)
        if v is not None:
            try:
                v.close()
            except Exception:
                pass
            self.env.viewer = None

        # 4. Rebuild scene with the new config.  Replace the scratch dir
        #    so old table XMLs don't linger.
        try:
            self.scratch.cleanup()
        except Exception:
            pass
        self.scratch = tempfile.TemporaryDirectory(
            prefix="test_multilevel_setup_")
        builder, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=Path(self.scratch.name), config=self.cfg)
        self.cfg = cfg
        self.ws = ws
        self.env = builder.build_env(rate=10000.0)

        # 5. Restore robot + block state into the new env.
        reloader.restore(self.env, state)

        # 6. Rebuild executor + bridge (they hold env references).
        rrt = RRTStar(self.env, max_iterations=3000)
        self.executor = MultilevelBlocksExecutor(
            self.env, self.ws, self.cfg, motion_planner=rrt,
            verbose=self.executor.verbose,
        )
        self.bridge, self.objects = make_multilevel_blocks_bridge(
            self.env, self.ws, self.cfg, executor=self.executor)

        # 7. Rebuild the offscreen renderer (bound to the old model).
        try:
            self._renderer.close()
        except Exception:
            pass
        self.env.model.vis.global_.offwidth = max(
            1280, self.env.model.vis.global_.offwidth)
        self.env.model.vis.global_.offheight = max(
            720, self.env.model.vis.global_.offheight)
        self._renderer = mujoco.Renderer(
            self.env.model, height=720, width=1280)

        # 8. Reopen the passive viewer on the new model/data.
        try:
            new_viewer = mujoco.viewer.launch_passive(
                self.env.model, self.env.data)
            self.env.viewer = new_viewer
            new_viewer.sync()
        except Exception as e:
            print(f"  warning: could not relaunch viewer ({e}); "
                   "continuing headless.")

        print(f"  rebuilt scene with {which}_table at {new_pos} "
               f"(legs regenerated; robot + block state preserved)")

    def cmd_render(self, args: Sequence[str]) -> None:
        path = Path(args[0]) if args else Path("/tmp/setup_snap.png")
        self._renderer.update_scene(self.env.data, camera=self._cam)
        img = self._renderer.render()
        import imageio.v3 as iio
        iio.imwrite(path, img)
        print(f"  wrote {path}")

    def cmd_blocks(self, args: Sequence[str]) -> None:
        for n in self._all_block_names():
            shape = self._shape_of(n)
            print(f"  {n}  ({shape})")

    def cmd_cells(self, args: Sequence[str]) -> None:
        sx, sy, sz = self.cfg.stack_grid_cells
        px, py = self.cfg.parts_grid_cells
        print(f"  parts        — parts__<ix>_<iy>  ix ∈ [0,{px-1}], iy ∈ [0,{py-1}]")
        for lvl in range(sz):
            print(f"  {stack_region_name(lvl):<12} — stack_L{lvl}__<ix>_<iy>  "
                   f"ix ∈ [0,{sx-1}], iy ∈ [0,{sy-1}]")

    def cmd_help(self, args: Sequence[str]) -> None:
        print(__doc__.split("Interactive commands::")[1])

    # ---- dispatcher -------------------------------------------------

    _COMMANDS = {
        "help": "cmd_help", "?": "cmd_help",
        "status": "cmd_status", "s": "cmd_status",
        "blocks": "cmd_blocks",
        "cells": "cmd_cells",
        "place": "cmd_place",
        "place-flat-x": "cmd_place_flat_x",
        "place-flat-y": "cmd_place_flat_y",
        "place-long-x": "cmd_place_long_x",
        "place-long-y": "cmd_place_long_y",
        "pick": "cmd_pick",
        "put-cube": "cmd_put_cube",
        "put-flat-x": "cmd_put_flat_x",
        "put-flat-y": "cmd_put_flat_y",
        "put-upright": "cmd_put_upright",
        "put-long-x": "cmd_put_long_x",
        "put-long-y": "cmd_put_long_y",
        "put-long-upright": "cmd_put_long_upright",
        "render": "cmd_render",
        "snap": "cmd_render", "screenshot": "cmd_render",
        "move-table": "cmd_move_table",
    }

    def dispatch(self, line: str) -> bool:
        """Execute one command line; return False to exit the loop.

        Any exception raised inside a command is caught + printed so a
        typo or bad arg doesn't terminate the whole session.
        """
        line = line.strip()
        if not line or line.startswith("#"):
            return True
        if line in ("quit", "exit", "q"):
            return False
        try:
            parts = shlex.split(line)
        except ValueError as e:
            print(f"  parse error: {e}")
            return True
        if not parts:
            return True
        cmd, args = parts[0], parts[1:]
        try:
            if cmd in self._COMMANDS:
                getattr(self, self._COMMANDS[cmd])(args)
            elif cmd in ("make-upright-from-x", "make-upright-from-y",
                           "make-flat-x-from-upright",
                           "make-flat-y-from-upright",
                           "turn-x-to-y", "turn-y-to-x",
                           "make-long-upright-from-x",
                           "make-long-upright-from-y",
                           "make-long-flat-x-from-upright",
                           "make-long-flat-y-from-upright",
                           "turn-long-x-to-y", "turn-long-y-to-x"):
                self.cmd_transform(cmd, args)
            else:
                print(f"  unknown command: {cmd!r}.  "
                       "Type 'help' for the list.")
        except Exception as e:
            print(f"  error: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stack-pos", type=float, nargs=3,
                    default=[0.0, 0.50, 0.0],
                    help="Stack table position (x y z).  Default 0 0.5 0.")
    p.add_argument("--parts-pos", type=float, nargs=3,
                    default=[0.0, -0.45, 0.0],
                    help="Parts table position.  Default 0 -0.45 0.")
    p.add_argument("--stack-cells", type=int, nargs=3, default=None,
                    help="Override stack grid dims (cells_x cells_y cells_z).")
    p.add_argument("--parts-cells", type=int, nargs=2, default=None,
                    help="Override parts grid dims (cells_x cells_y).")
    p.add_argument("--cube-size", type=float, default=None,
                    help="Override cube edge length in metres "
                         "(default 0.030).")
    p.add_argument("--n-cubes", type=int, default=2)
    p.add_argument("--n-oblong", type=int, default=2)
    p.add_argument("--n-long", type=int, default=1)
    p.add_argument("--place", action="append", nargs=2, default=[],
                    metavar=("BLOCK", "CELL"),
                    help="Initial placement(s) before entering the prompt.  "
                         "Repeat for multiple placements.")
    p.add_argument("--script", default=None,
                    help="Semicolon-separated command sequence to run.  "
                         "If given, no interactive prompt; the session "
                         "exits after the script finishes.")
    p.add_argument("--verbose", action="store_true",
                    help="Verbose executor logging.")
    p.add_argument("--no-viewer", action="store_true",
                    help="Skip the MuJoCo passive viewer.  Required when "
                         "running headlessly (CI, ssh without X11) or via "
                         "plain python instead of mjpython on macOS.")
    args = p.parse_args()

    sess = Session(args)

    # Launch the passive viewer in the same process.  Attaching the
    # handle to env.viewer makes env.step() auto-sync after every
    # mj_step so motion-mode actions animate live.
    viewer_handle = None
    if not args.no_viewer:
        try:
            viewer_handle = mujoco.viewer.launch_passive(
                sess.env.model, sess.env.data)
            sess.env.viewer = viewer_handle
            viewer_handle.sync()
        except Exception as e:
            print(f"  warning: could not launch viewer ({e}); "
                   "continuing headless.  On macOS, run with `mjpython`.")
            viewer_handle = None

    print("multilevel_blocks test session ready.  "
           "Type 'help' or 'status'.\n")

    try:
        # Apply initial placements
        for block, cell in args.place:
            sess.dispatch(f"place {block} {cell}")

        if args.script:
            for line in args.script.split(";"):
                line = line.strip()
                if line:
                    print(f">>> {line}")
                    if not sess.dispatch(line):
                        break
        else:
            # Interactive loop
            while True:
                try:
                    line = input(">>> ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not sess.dispatch(line):
                    break
    finally:
        if viewer_handle is not None:
            sess.env.viewer = None
            try:
                viewer_handle.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
