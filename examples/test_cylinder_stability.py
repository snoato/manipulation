"""Cylinder placement stability tests.

Two experiments:

  1. Z-clearance sweep  — for each cylinder type, place it at a range of
     Z offsets above the computed resting position, step 500 timesteps, and
     record: final resting Z, peak velocity, and whether it settled cleanly.
     Reveals the correct per-type clearance and whether cylinders bounce.

  2. Settling convergence — place all cylinder types simultaneously at their
     *current* clearance (0.013 m as used in _sample_state) and record
     per-cylinder XY/Z displacement and peak |qvel| every 10 timesteps up
     to 500 steps.  Shows how many settling steps are actually needed.

Usage:
    python examples/test_cylinder_stability.py
    python examples/test_cylinder_stability.py --steps 800 --reps 3
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from tampanda import FrankaEnvironment, SCENE_SYMBOLIC
from tampanda.symbolic.domains.tabletop.grid_domain import GridDomain
from tampanda.symbolic.domains.tabletop.state_manager import StateManager

_XML = SCENE_SYMBOLIC

TABLE_BODY  = "simple_table"
TABLE_GEOM  = "table_surface"

# One representative cylinder index per type
CYLINDER_TYPES = {
    "thin":   {"idx": 0,  "color": "steelblue"},
    "medium": {"idx": 15, "color": "darkorange"},
    "thick":  {"idx": 25, "color": "seagreen"},
}

# Z clearances to sweep (metres above computed resting Z)
Z_SWEEP = np.linspace(-0.005, 0.025, 13)   # -5 mm … +25 mm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_env():
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)

    # Bypass rate limiter
    _dt = env.model.opt.timestep
    def _fast_step():
        mujoco.mj_step(env.model, env.data)
        env.sim_time += _dt
        return _dt
    env.step = _fast_step

    grid = GridDomain(
        model=env.model,
        cell_size=0.04,
        working_area=(0.4, 0.3),
        table_body_name=TABLE_BODY,
        table_geom_name=TABLE_GEOM,
        grid_offset_x=0.05,
        grid_offset_y=0.25,
    )
    state_manager = StateManager(grid, env)
    return env, grid, state_manager


def hide_all(state_manager):
    for i in range(30):
        state_manager._hide_cylinder(i)
    mujoco.mj_forward(state_manager.env.model, state_manager.env.data)


def place_cylinder(state_manager, cyl_idx, x, y, z):
    state_manager._set_cylinder_position(cyl_idx, x, y, z)
    state_manager.env.data.qvel[:] = 0.0
    mujoco.mj_forward(state_manager.env.model, state_manager.env.data)


def get_cyl_qvel_indices(env, cyl_idx):
    """Return the slice of qvel that corresponds to cylinder_<cyl_idx>."""
    body_name = f"cylinder_{cyl_idx}"
    body_id   = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jnt_start = env.model.body_jntadr[body_id]
    jnt_num   = env.model.body_jntnum[body_id]
    if jnt_num == 0:
        return slice(0, 0)
    dof_start = env.model.jnt_dofadr[jnt_start]
    dof_num   = sum(env.model.jnt_dofadr[jnt_start + k + 1]
                    - env.model.jnt_dofadr[jnt_start + k]
                    for k in range(jnt_num - 1))
    dof_num  += 1  # last joint
    # Simpler: free joint = 6 DOF
    return slice(dof_start, dof_start + 6)


def cylinder_speed(env, cyl_idx):
    sl = get_cyl_qvel_indices(env, cyl_idx)
    if sl.start == sl.stop:
        return 0.0
    return float(np.linalg.norm(env.data.qvel[sl]))


def cylinder_pos(env, cyl_idx):
    body_name = f"cylinder_{cyl_idx}"
    return env.get_object_position(body_name).copy()


# ---------------------------------------------------------------------------
# Experiment 1 — Z-clearance sweep
# ---------------------------------------------------------------------------

def exp1_z_sweep(env, state_manager, n_steps=500):
    print("\n" + "="*65)
    print("  Experiment 1: Z-clearance sweep")
    print("="*65)
    print(f"  Steps per trial: {n_steps}   dt={env.model.opt.timestep:.4f}s"
          f"   total={n_steps*env.model.opt.timestep:.2f}s")
    print()

    # Fixed XY — centre of the grid
    grid  = state_manager.grid
    table_z = grid.table_height
    cx = (grid.table_bounds['min_x'] + grid.table_bounds['max_x']) / 2
    cy = (grid.table_bounds['min_y'] + grid.table_bounds['max_y']) / 2

    results = {}  # type_name -> list of dicts

    for type_name, info in CYLINDER_TYPES.items():
        cyl_idx = info["idx"]
        radius, height = StateManager.CYLINDER_SPECS[cyl_idx]
        resting_z = table_z + height   # h is the MuJoCo geom half-height → body centre

        print(f"  [{type_name}]  idx={cyl_idx}  r={radius*1000:.1f}mm"
              f"  h={height*1000:.0f}mm  resting_z={resting_z:.4f}m")

        rows = []
        for dz in Z_SWEEP:
            placed_z = resting_z + dz
            hide_all(state_manager)
            place_cylinder(state_manager, cyl_idx, cx, cy, placed_z)

            peak_speed = 0.0
            for _ in range(n_steps):
                env.step()
                speed = cylinder_speed(env, cyl_idx)
                if speed > peak_speed:
                    peak_speed = speed

            final_pos   = cylinder_pos(env, cyl_idx)
            final_speed = cylinder_speed(env, cyl_idx)
            drop        = placed_z - final_pos[2]

            rows.append({
                "dz_mm":        dz * 1000,
                "placed_z":     placed_z,
                "final_z":      final_pos[2],
                "drop_mm":      drop * 1000,
                "peak_speed":   peak_speed,
                "final_speed":  final_speed,
                "settled":      final_speed < 1e-3,
            })
            status = "OK " if final_speed < 1e-3 else "BAD"
            print(f"    dz={dz*1000:+6.1f}mm  final_z={final_pos[2]:.4f}"
                  f"  drop={drop*1000:+6.2f}mm  peak_v={peak_speed:.4f}"
                  f"  final_v={final_speed:.5f}  {status}")

        results[type_name] = rows
        print()

    return results


# ---------------------------------------------------------------------------
# Experiment 2 — Settling convergence
# ---------------------------------------------------------------------------

def exp2_settling(env, state_manager, n_steps=500, current_clearance=0.013, reps=3):
    print("\n" + "="*65)
    print("  Experiment 2: Settling convergence")
    print(f"  clearance={current_clearance*1000:.1f}mm   steps={n_steps}")
    print("="*65)

    grid    = state_manager.grid
    table_z = grid.table_height

    # Fixed placement positions — spread across grid
    positions = [
        (grid.table_bounds['min_x'] + grid.table_bounds['max_x']) / 2,
        (grid.table_bounds['min_y'] + grid.table_bounds['max_y']) / 2,
    ]

    # Record series: {type_name -> array shape (n_steps,)}
    speed_series  = {n: np.zeros(n_steps) for n in CYLINDER_TYPES}
    xy_disp_series = {n: np.zeros(n_steps) for n in CYLINDER_TYPES}
    z_disp_series  = {n: np.zeros(n_steps) for n in CYLINDER_TYPES}

    for rep in range(reps):
        hide_all(state_manager)

        # Place all three types simultaneously
        placed_z0 = {}
        placed_xy0 = {}
        for type_name, info in CYLINDER_TYPES.items():
            cyl_idx = info["idx"]
            _, height = StateManager.CYLINDER_SPECS[cyl_idx]
            resting_z = table_z + height
            pz = resting_z + current_clearance

            # Spread them so they don't overlap
            offsets = {"thin": (-0.08, 0.0), "medium": (0.0, 0.0), "thick": (0.08, 0.0)}
            dx, dy  = offsets[type_name]
            px, py  = positions[0] + dx, positions[1] + dy

            place_cylinder(state_manager, cyl_idx, px, py, pz)
            placed_z0[type_name]  = pz
            placed_xy0[type_name] = np.array([px, py])

        for step in range(n_steps):
            env.step()
            for type_name, info in CYLINDER_TYPES.items():
                cyl_idx = info["idx"]
                pos     = cylinder_pos(env, cyl_idx)
                speed   = cylinder_speed(env, cyl_idx)

                xy_disp = np.linalg.norm(pos[:2] - placed_xy0[type_name])
                z_disp  = abs(pos[2] - placed_z0[type_name])

                speed_series[type_name][step]   += speed   / reps
                xy_disp_series[type_name][step] += xy_disp / reps
                z_disp_series[type_name][step]  += z_disp  / reps

        print(f"  rep {rep+1}/{reps} done")

    # Print summary: first step where speed < threshold
    threshold = 1e-3
    print(f"\n  Steps to settle (speed < {threshold}):")
    for type_name in CYLINDER_TYPES:
        arr   = speed_series[type_name]
        below = np.where(arr < threshold)[0]
        if len(below):
            print(f"    {type_name:<8} settled at step {below[0]:4d}"
                  f"  ({below[0]*env.model.opt.timestep*1000:.0f}ms)")
        else:
            print(f"    {type_name:<8} NEVER settled within {n_steps} steps")

    return speed_series, xy_disp_series, z_disp_series


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_exp1(results, n_steps, save_path="cylinder_stability_exp1.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Exp 1: Z-clearance sweep ({n_steps} steps per trial)", fontsize=13)

    metrics = [
        ("final_speed",  "Final speed (m/s)",       "tab:red"),
        ("peak_speed",   "Peak speed (m/s)",         "tab:orange"),
        ("drop_mm",      "Z drop from placed (mm)",  "tab:blue"),
    ]

    for ax, (key, label, color) in zip(axes, metrics):
        for type_name, info in CYLINDER_TYPES.items():
            rows  = results[type_name]
            xs    = [r["dz_mm"] for r in rows]
            ys    = [r[key] for r in rows]
            ax.plot(xs, ys, marker="o", label=type_name, color=info["color"])
        ax.axvline(13, color="gray", linestyle="--", linewidth=0.8, label="current (13mm)")
        ax.axvline(3,  color="black", linestyle=":",  linewidth=0.8, label="set_from_state (3mm)")
        ax.set_xlabel("Z clearance above resting (mm)")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n  Saved: {save_path}")


def plot_exp2(speed_series, xy_disp_series, z_disp_series, n_steps, dt,
              save_path="cylinder_stability_exp2.png"):
    steps = np.arange(n_steps)
    time_ms = steps * dt * 1000

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Exp 2: Settling convergence over time", fontsize=13)

    datasets = [
        (speed_series,   "Speed (m/s)"),
        (xy_disp_series, "XY displacement from placed (m)"),
        (z_disp_series,  "Z displacement from placed (m)"),
    ]

    for ax, (series, label) in zip(axes, datasets):
        for type_name, info in CYLINDER_TYPES.items():
            ax.plot(time_ms, series[type_name], label=type_name, color=info["color"])
        ax.axhline(1e-3, color="gray", linestyle="--", linewidth=0.8, label="1e-3 threshold")
        ax.axvline(1 * dt * 1000, color="red", linestyle=":", linewidth=1.0,
                   label="current (1 step)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500,
                    help="Timesteps per trial (both experiments)")
    ap.add_argument("--reps",  type=int, default=3,
                    help="Repetitions for experiment 2 (averaged)")
    ap.add_argument("--out-dir", type=str, default=".",
                    help="Directory to save plot images")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Building environment...")
    env, grid, state_manager = build_env()

    dt = env.model.opt.timestep
    print(f"  dt={dt}s  table_z={grid.table_height:.4f}m")

    # --- Exp 1 ---
    r1 = exp1_z_sweep(env, state_manager, n_steps=args.steps)
    plot_exp1(r1, args.steps, save_path=str(out / "cylinder_stability_exp1.png"))

    # --- Exp 2 ---
    speed_s, xy_s, z_s = exp2_settling(
        env, state_manager,
        n_steps=args.steps,
        current_clearance=0.002,
        reps=args.reps,
    )
    plot_exp2(speed_s, xy_s, z_s, args.steps, dt,
              save_path=str(out / "cylinder_stability_exp2.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
