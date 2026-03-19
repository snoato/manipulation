"""IK-based grasping using GraspPlanner for geometry-aware grasp candidates.

Compare with grasping_ik.py which uses hardcoded EE offset poses.
GraspPlanner differences:
  - Reads object geometry (half-extents) to compute valid grasp widths
  - Generates and ranks multiple candidates (top-down X/Y, front approach)
  - Accounts for table clearance and gripper opening limits
  - Aligns finger axis with block orientation to minimise required opening
"""

from pathlib import Path
import time

from manipulation import FrankaEnvironment, ControllerStatus, GraspPlanner
from manipulation.planners.grasp_planner import GraspType

_HERE = Path(__file__).parent
_XML = _HERE / ".." / "manipulation" / "environments" / "assets" / "franka_emika_panda" / "scene_test.xml"

_TABLE_Z = 0.27


def _pick_candidate(candidates):
    """Select best candidate for tall thin cylinders.

    GraspPlanner scores top-down highest (+20), but for tall cylinders the
    top-down grasp point lands inside the cylinder body (contact at body
    centre ≠ top surface) and the round surface makes it mechanically
    unstable.  A front (side) approach contacts at the correct height and
    grips around the cylinder properly.
    """
    front = next((c for c in candidates if c.grasp_type == GraspType.FRONT), None)
    return front if front is not None else (candidates[0] if candidates else None)


def main():
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    ik = env.get_ik()
    grasp_planner = GraspPlanner(table_z=_TABLE_Z)

    with env.launch_viewer() as viewer:
        targets   = []
        target    = None
        candidate = None

        while viewer.is_running():
            if target is None and len(targets) == 0:
                targets = ["cylinder1", "cylinder2", "cylinder3"]
                env.reset()
                env.rest(2.0)

            dt = env.step()

            if env.sim_time > 0.0 and target is None and len(targets) > 0:
                target     = targets[0]
                target_pos = env.get_object_position(target)
                half_size  = env.get_object_half_size(target)
                block_quat = env.get_object_orientation(target)
                candidates = grasp_planner.generate_candidates(target_pos, half_size, block_quat)
                candidate  = _pick_candidate(candidates)
                step = 0
                if candidate:
                    print(f"Grasp candidate: {candidate.grasp_type.value}  score={candidate.score:.0f}")
                else:
                    print(f"No valid grasp candidate for {target}, skipping.")

            if target is not None and candidate is not None:
                if env.controller.get_status() == ControllerStatus.IDLE:
                    step += 1
                    print(f"Time: {env.sim_time:.2f}, Step: {step}")

                    if step == 1:
                        print(f"Approaching {target}...")
                        ik.set_target_position(candidate.approach_pos, candidate.grasp_quat)
                        converged = ik.converge_ik(dt)
                        print("IK result:", "Converged" if converged else "Not converged")
                        env.move_to_compensated(ik.configuration.q[:7])

                    if step == 2:
                        print("Moving to grasp pose...")
                        ik.set_target_position(candidate.grasp_pos, candidate.grasp_quat)
                        converged = ik.converge_ik(dt)
                        print("IK result:", "Converged" if converged else "Not converged")
                        env.move_to_compensated(ik.configuration.q[:7])

                    if step == 3:
                        print("Closing gripper...")
                        env.controller.close_gripper()

                    if step == 4:
                        print("Lifting object...")
                        env.attach_object_to_ee(target)
                        ik.set_target_position(candidate.lift_pos, candidate.grasp_quat)
                        converged = ik.converge_ik(dt)
                        print("IK result:", "Converged" if converged else "Not converged")
                        env.move_to_compensated(ik.configuration.q[:7])

                    if step == 5:
                        print("Moving to dropoff...")
                        dropoff_pos, dropoff_quat = env.get_dropoff_pose()
                        ik.set_target_position(dropoff_pos, dropoff_quat)
                        converged = ik.converge_ik(dt)
                        print("IK result:", "Converged" if converged else "Not converged")
                        env.move_to_compensated(ik.configuration.q[:7])

                    if step == 6:
                        print("Opening gripper...")
                        env.detach_object()
                        env.controller.open_gripper()

                    if step == 7:
                        print(f"Completed {target}!")
                        target    = None
                        candidate = None
                        step = 0
                        if len(targets) > 1:
                            targets = targets[1:]
                        else:
                            targets = []
                            time.sleep(2.0)

                if step == 3 or step == 6:
                    env.rest(2.0)

            elif target is not None and candidate is None:
                # No valid grasp found — skip to next target
                target = None
                if len(targets) > 1:
                    targets = targets[1:]
                else:
                    targets = []

            env.controller.step()


if __name__ == "__main__":
    main()
