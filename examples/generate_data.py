"""Example combining symbolic planning with grasping RRT."""

from pathlib import Path
import time
import argparse
import numpy as np
from collections import deque
import multiprocessing as mp
try:
    import wandb
except ImportError:
    wandb = None

from manipulation import FrankaEnvironment, RRTStar, ControllerStatus
from manipulation.symbolic import GridDomain, StateManager, visualize_grid_state

_HERE = Path(__file__).parent
_XML = _HERE / ".." / "manipulation" / "environments" / "assets" / "franka_emika_panda" / "scene_symbolic.xml"

# Default values
_DEFAULT_OUTPUT_DIR = _HERE / ".." / "data"
_DEFAULT_TRAIN_EXAMPLES = 1000
_DEFAULT_TEST_EXAMPLES = 100
_DEFAULT_VAL_EXAMPLES = 100
_DEFAULT_TARGETS_MAX = 7
_DEFAULT_TARGETS_MIN = 3
_DEFAULT_GRID_SIZE = (0.4, 0.3)  # 40cm x 40cm
_DEFAULT_CELL_SIZE = 0.04  # 2cm cells
_DEFAULT_GRID_OFFSET_Y = 0.02
_DEFAULT_NUM_WORKERS = 1


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate symbolic planning problem configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_DEFAULT_OUTPUT_DIR.as_posix(),
        help="Base output directory for generated data"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num-train",
        type=int,
        default=_DEFAULT_TRAIN_EXAMPLES,
        help="Number of training examples to generate"
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=_DEFAULT_TEST_EXAMPLES,
        help="Number of testing examples to generate"
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=_DEFAULT_VAL_EXAMPLES,
        help="Number of validation examples to generate"
    )
    
    parser.add_argument(
        "--min-objects",
        type=int,
        default=_DEFAULT_TARGETS_MIN,
        help="Minimum number of objects per configuration"
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=_DEFAULT_TARGETS_MAX,
        help="Maximum number of objects per configuration"
    )
    # Grid configuration
    parser.add_argument(
        "--grid-width",
        type=float,
        default=_DEFAULT_GRID_SIZE[0],
        help="Grid width in meters"
    )
    parser.add_argument(
        "--grid-height",
        type=float,
        default=_DEFAULT_GRID_SIZE[1],
        help="Grid height in meters"
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=_DEFAULT_CELL_SIZE,
        help="Grid cell size in meters"
    )
    parser.add_argument(
        "--grid-offset-y",
        type=float,
        default=_DEFAULT_GRID_OFFSET_Y,
        help="Grid Y-axis offset in meters"
    )
    
    # Visualization and monitoring
    parser.add_argument(
        "--viewer",
        action="store_true",
        default=False,
        help="Enable MuJoCo viewer during generation"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        default=False,
        help="Disable visualization image generation"
    )
    
    # Wandb integration
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="manipulation-data-generation",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (auto-generated if not specified)"
    )
    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    # Parallelization
    parser.add_argument(
        "--num-workers",
        type=int,
        default=_DEFAULT_NUM_WORKERS,
        help="Number of parallel worker processes (0 = use all CPUs)"
    )
    
    return parser.parse_args()

def validate_grasp(env, planner, target_obj, ignored_objects=None):
    """
    Check if a target object can be grasped given a set of ignored objects (collision exceptions).
    
    Args:
        env: FrankaEnvironment
        planner: RRTStar planner
        target_obj: Name of target object
        ignored_objects: List of object names to ignore during collision checking (objects already removed)
    """
    if ignored_objects is None:
        ignored_objects = []
        
    # Save current collision state
    saved_exceptions = env.collision_exceptions.copy()
    
    try:
        # Apply collision exceptions: ignore removed objects + target itself for grasp
        # Use set_collision_exceptions to properly sync cached IDs
        env.set_collision_exceptions(list(set(ignored_objects + [target_obj])))
        
        # Get approach pose for target
        obj_pos = env.get_object_position(target_obj)
        approach_pose, approach_ori = env.get_approach_pose(obj_pos)
        
        # 3. Plan to approach pose
        # Lower max_iterations for validation speed (e.g. 200 or 500)
        path = planner.plan_to_pose(approach_pose, approach_ori, max_iterations=200)
        
        return path is not None
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False
    finally:
        # Restore collision state using setter to sync cached IDs
        env.set_collision_exceptions(saved_exceptions)

def find_shortest_plan(env, planner, target_obj, all_objects):
    """
    Find shortest sequence of object removals to reach target using BFS.
    
    Args:
        env: FrankaEnvironment
        planner: RRTStar planner
        target_obj: Name of target object
        all_objects: List of all object names in the scene
        
    Returns:
        List of object names to remove (pick) in order, ending with target_obj.
        Returns None if no plan found.
    """
    all_objs_set = set(all_objects)
    if target_obj not in all_objs_set:
        return None
        
    # State: (objects_currently_on_table, plan_sequence)
    # Using frozenset for visited check
    start_state = frozenset(all_objs_set)
    queue = deque([(start_state, [])])
    visited = {start_state}
    
    while queue:
        current_objects, plan = queue.popleft()
        
        # Check if target is reachable in current state
        # Ignored objects are those NOT in current_objects (already removed)
        ignored = list(all_objs_set - current_objects)
        
        if validate_grasp(env, planner, target_obj, ignored):
            # Found a solution! Plan is sequence of removals + target
            return plan + [target_obj]
            
        # Try to remove other objects
        # We can only remove an object if it is reachable/graspable in current state
        # Sorted for determinism
        potential_removals = sorted([o for o in current_objects if o != target_obj])
        
        for obj in potential_removals:
            # Check if we can pick 'obj'
            if validate_grasp(env, planner, obj, ignored):
                new_objects = current_objects - {obj}
                if new_objects not in visited:
                    visited.add(new_objects)
                    new_plan = plan + [obj]
                    queue.append((new_objects, new_plan))
                    
    return None


def worker_generate(worker_id, split_name, start_idx, count, args_dict, result_queue):
    """
    Worker function that generates a subset of examples.
    
    Each worker creates its own environment and planner instances.
    
    Args:
        worker_id: Unique worker identifier
        split_name: Dataset split name (train/test/validation)
        start_idx: Starting config index for this worker
        count: Number of examples this worker should generate
        args_dict: Parsed arguments as dict (for pickling)
        result_queue: Queue to report results back to main process
    """
    # Reconstruct args namespace from dict
    class Args:
        pass
    args = Args()
    for k, v in args_dict.items():
        setattr(args, k, v)
    
    # Set unique seed for this worker
    if args.seed is not None:
        worker_seed = args.seed + worker_id * 10000 + hash(split_name) % 10000
    else:
        worker_seed = int(time.time() * 1000) % (2**31) + worker_id * 10000
    np.random.seed(worker_seed)
    
    # Initialize environment (each worker gets its own)
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    
    # Create grid domain
    grid = GridDomain(
        model=env.model,
        cell_size=args.cell_size,
        working_area=(args.grid_width, args.grid_height),
        table_body_name="simple_table",
        table_geom_name="table_surface",
        grid_offset_y=args.grid_offset_y
    )
    
    # Create state manager and planner
    state_manager = StateManager(grid, env)
    planner = RRTStar(env)
    
    # Output directories
    output_base = Path(args.output_dir) / "tabletop-domain"
    problem_dir = output_base / split_name
    viz_dir = output_base / split_name / "viz"
    
    start_time = time.time()
    generated_count = 0
    
    while generated_count < count:
        config_num = start_idx + generated_count + 1
        
        # Sample random state
        n_cylinders = np.random.randint(args.min_objects, args.max_objects + 1)
        state_manager.sample_random_state(n_cylinders=n_cylinders)
        
        grounded_state = state_manager.ground_state()
        cylinders = sorted(grounded_state['cylinders'].keys())
        target_cylinder = np.random.choice(cylinders)

        # Fast check: target reachable in isolation
        all_others = [c for c in cylinders if c != target_cylinder]
        if not validate_grasp(env, planner, target_cylinder, all_others):
            continue
            
        # Find shortest plan
        plan_sequence = find_shortest_plan(env, planner, target_cylinder, cylinders)
        
        if plan_sequence is None:
            continue
            
        actual_cylinders = len(cylinders)
        
        # Generate PDDL problem
        problem_path = problem_dir / f"config_{config_num}.pddl"
        state_manager.generate_pddl_problem(
            f"config-{config_num}",
            problem_path,
            f"(holding {target_cylinder})"
        )

        # Generate Plan file
        plan_path = problem_dir / f"config_{config_num}.pddl.plan"
        with open(plan_path, "w") as f:
            # For obstacles: pick then drop (to clear the way)
            for action_obj in plan_sequence[:-1]:
                f.write(f"(pick {action_obj})\n")
                f.write(f"(drop {action_obj})\n")
            # For target: only pick (goal is holding)
            f.write(f"(pick {plan_sequence[-1]})\n")
            f.write(f"; Total actions: {len(plan_sequence)*2 - 1}\n")
            f.write(f"; Target: {target_cylinder}\n")
            f.write(f"; Objects: {', '.join(cylinders)}\n")
        
        # Visualize if enabled
        if not args.no_viz:
            viz_path = viz_dir / f"config_{config_num}.png"
            visualize_grid_state(
                state_manager,
                save_path=viz_path,
                title=f"{split_name.capitalize()} Conf {config_num} (len={len(plan_sequence)})",
                target_cylinder=target_cylinder
            )
        
        generated_count += 1
        
        # Progress update every 10 configs
        if generated_count % 10 == 0 or generated_count == 1:
            elapsed = time.time() - start_time
            print(f"   [Worker {worker_id}/{split_name}] {generated_count}/{count} | "
                  f"Config {config_num} | Len: {len(plan_sequence)} | "
                  f"Elapsed: {elapsed:.1f}s")

    total_time = time.time() - start_time
    result_queue.put({
        'worker_id': worker_id,
        'split_name': split_name,
        'generated': generated_count,
        'time': total_time
    })
    
    # Cleanup
    env.close()


def generate_dataset_parallel(split_name, num_examples, args, num_workers):
    """
    Generate a dataset split using multiple worker processes.
    """
    if num_examples <= 0:
        print(f"Skipping {split_name} split (0 examples requested)")
        return

    print(f"\n{'='*70}")
    print(f"Generating {split_name} dataset ({num_examples} examples) with {num_workers} workers")
    print(f"{'='*70}")
    
    # Create output directories (main process)
    output_base = Path(args.output_dir) / "tabletop-domain"
    problem_dir = output_base / split_name
    viz_dir = output_base / split_name / "viz"
    
    problem_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"   Problem directory: {problem_dir}")
    if not args.no_viz:
        print(f"   Visualization directory: {viz_dir}")

    # Convert args to dict for pickling
    args_dict = vars(args)
    
    # Distribute work among workers
    base_count = num_examples // num_workers
    remainder = num_examples % num_workers
    
    result_queue = mp.Queue()
    processes = []
    
    start_idx = 0
    start_time = time.time()
    
    for worker_id in range(num_workers):
        # Give first 'remainder' workers one extra example
        worker_count = base_count + (1 if worker_id < remainder else 0)
        
        if worker_count > 0:
            p = mp.Process(
                target=worker_generate,
                args=(worker_id, split_name, start_idx, worker_count, args_dict, result_queue)
            )
            processes.append(p)
            p.start()
            start_idx += worker_count
    
    # Wait for all workers to complete
    for p in processes:
        p.join()
    
    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    total_time = time.time() - start_time
    total_generated = sum(r['generated'] for r in results)
    
    print(f"   {split_name} complete: {total_generated} configs in {total_time:.1f}s")
    
    return results


def generate_dataset(split_name, num_examples, env, grid, args, wandb_run=None):
    """
    Generate a dataset split.
    
    Args:
        split_name (str): Name of the split (train, test, val)
        num_examples (int): Number of examples to generate
        env (FrankaEnvironment): The initialized environment
        grid (GridDomain): The initialized grid domain
        args (Namespace): Parsed command line arguments
        wandb_run: Active wandb run, or None
    """
    if num_examples <= 0:
        print(f"Skipping {split_name} split (0 examples requested)")
        return

    print(f"\n{'='*70}")
    print(f"Generating {split_name} dataset ({num_examples} examples)")
    print(f"{'='*70}")

    # Create state manager
    state_manager = StateManager(grid, env)
    
    # Create output directories
    output_base = Path(args.output_dir) / "tabletop-domain"
    # Structure: {output_dir}/tabletop-domain/{split}/
    problem_dir = output_base / split_name
    viz_dir = output_base / split_name / "viz"
    
    problem_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"   Problem directory: {problem_dir}")
    if not args.no_viz:
        print(f"   Visualization directory: {viz_dir}")

    # Initialize RRTStar planner for check
    planner = RRTStar(env)

    start_time = time.time()
    
    i = 0
    generated_count = 0
    
    while generated_count < num_examples:
        config_num = generated_count + 1
        
        # Sample random state
        n_cylinders = np.random.randint(args.min_objects, args.max_objects + 1)
        state_manager.sample_random_state(n_cylinders=n_cylinders)
        
        # Ground state to get object details
        grounded_state = state_manager.ground_state()
        cylinders = sorted(grounded_state['cylinders'].keys())
        target_cylinder = np.random.choice(cylinders)

        # 1. First fast check: Is target reachable if it was alone? (Sanity check)
        # If the target itself is in a bad pose (too far, etc), skip immediately
        all_others = [c for c in cylinders if c != target_cylinder]
        if not validate_grasp(env, planner, target_cylinder, all_others):
            continue
            
        # 2. Find shortest plan
        plan_sequence = find_shortest_plan(env, planner, target_cylinder, cylinders)
        
        if plan_sequence is None:
            continue
            
        # If valid plan found, we accept this configuration
        actual_cylinders = len(cylinders)
        
        # Generate PDDL problem
        problem_path = problem_dir / f"config_{config_num}.pddl"
        state_manager.generate_pddl_problem(
            f"config-{config_num}",
            problem_path,
            f"(holding {target_cylinder})"
        )

        # Generate Plan file
        plan_path = problem_dir / f"config_{config_num}.pddl.plan"
        with open(plan_path, "w") as f:
            for action_obj in plan_sequence[:-1]:
                f.write(f"(pick {action_obj})\n")
                f.write(f"(drop {action_obj})\n")
            f.write(f"(pick {plan_sequence[-1]})\n")
            f.write(f"; Total actions: {len(plan_sequence)*2 - 1}\n")
            f.write(f"; Target: {target_cylinder}\n")
            f.write(f"; Objects: {', '.join(cylinders)}\n")
            f.write(f"; Total time to generate: {time.time() - start_time:.2f}s\n")

        # Visualize if enabled
        viz_path = None
        if not args.no_viz:
            viz_path = viz_dir / f"config_{config_num}.png"
            visualize_grid_state(
                state_manager,
                save_path=viz_path,
                title=f"{split_name.capitalize()} Conf {config_num} (len={len(plan_sequence)})",
                target_cylinder=target_cylinder
            )
        
        # Log to wandb
        if wandb_run is not None:
            log_data = {
                f"{split_name}/progress": (generated_count + 1) / num_examples,
                f"{split_name}/config_num": config_num,
                f"{split_name}/n_objects": actual_cylinders,
                f"{split_name}/plan_length": len(plan_sequence)
            }
            
            # Log visualization image if available
            if viz_path is not None and viz_path.exists():
                log_data[f"{split_name}/visualization"] = wandb.Image(str(viz_path))
            
            wandb.log(log_data)
        
        # Rest if viewer is active
        if args.viewer:
            env.rest(2.0)
        
        generated_count += 1
        i += 1
        
        # Print progress estimate
        if generated_count % 10 == 0 or generated_count == 1:
            elapsed = time.time() - start_time
            avg_time = elapsed / generated_count
            remaining = avg_time * (num_examples - generated_count)
            print(f"   [{split_name}] {generated_count}/{num_examples} | Saved: {problem_path.name} | "
                  f"Len: {len(plan_sequence)} | "
                  f"Elapsed: {elapsed:.1f}s | "
                  f"Est. remaining: {remaining:.1f}s")

    total_time = time.time() - start_time
    print(f"   {split_name} complete in {total_time:.1f}s")

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Determine number of workers
    num_workers = args.num_workers
    if num_workers <= 0:
        num_workers = mp.cpu_count()
    
    # Viewer mode forces single worker
    if args.viewer and num_workers > 1:
        print("WARNING: Viewer mode enabled, forcing single worker")
        num_workers = 1
    
    # Initialize wandb if requested (main process only)
    wandb_run = None
    if args.wandb:
        if wandb is None:
            print("WARNING: wandb not installed. Install with: pip install wandb")
            print("Continuing without W&B logging...")
        else:
            # Initialize wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "num_train": args.num_train,
                    "num_test": args.num_test,
                    "num_val": args.num_val,
                    "min_objects": args.min_objects,
                    "max_objects": args.max_objects,
                    "grid_width": args.grid_width,
                    "grid_height": args.grid_height,
                    "cell_size": args.cell_size,
                    "grid_offset_y": args.grid_offset_y,
                    "seed": args.seed,
                    "num_workers": num_workers,
                }
            )
            print("W&B logging enabled")
    
    print("=" * 70)
    print("Symbolic Planning Data Generation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Train examples: {args.num_train}")
    print(f"  Test examples: {args.num_test}")
    print(f"  Val examples: {args.num_val}")
    print(f"  Objects per config: {args.min_objects}-{args.max_objects}")
    print(f"  Grid size: {args.grid_width}m x {args.grid_height}m")
    print(f"  Cell size: {args.cell_size}m")
    print(f"  Workers: {num_workers}")
    print(f"  Viewer: {args.viewer}")
    print(f"  Visualizations: {not args.no_viz}")
    
    # Create tabletop-domain directory and copy domain.pddl (main process)
    import shutil
    output_base = Path(args.output_dir) / "tabletop-domain"
    output_base.mkdir(parents=True, exist_ok=True)
    
    domain_src = _HERE / ".." / "manipulation" / "symbolic" / "domains" / "tabletop" / "pddl" / "domain.pddl"
    domain_dst = output_base / "domain.pddl"
    if domain_src.exists():
        shutil.copy(domain_src, domain_dst)
        print(f"   Copied domain.pddl to {domain_dst}")
        
    # Generate splits
    start_total = time.time()
    
    if num_workers > 1:
        # Parallel mode
        generate_dataset_parallel("train", args.num_train, args, num_workers)
        generate_dataset_parallel("test", args.num_test, args, num_workers)
        generate_dataset_parallel("validation", args.num_val, args, num_workers)
    else:
        # Single-threaded mode (with viewer support)
        print("\n1. Loading environment...")
        env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
        
        print("2. Creating grid domain...")
        grid = GridDomain(
            model=env.model,
            cell_size=args.cell_size,
            working_area=(args.grid_width, args.grid_height),
            table_body_name="simple_table",
            table_geom_name="table_surface",
            grid_offset_y=args.grid_offset_y
        )
        
        info = grid.get_grid_info()
        print(f"   Grid: {info['grid_dimensions'][0]}x{info['grid_dimensions'][1]} cells ({info['cell_size']*100:.1f}cm)")
        
        if args.viewer:
            viewer = env.launch_viewer()
            if not viewer.is_running():
                raise Exception("no viewer available")
            print("   Viewer launched")
        
        generate_dataset("train", args.num_train, env, grid, args, wandb_run)
        generate_dataset("test", args.num_test, env, grid, args, wandb_run)
        generate_dataset("validation", args.num_val, env, grid, args, wandb_run)
        
        env.close()
    
    # Summary
    total_time = time.time() - start_total
    total_examples = args.num_train + args.num_test + args.num_val
    print(f"\n{'='*70}")
    print(f"All Generation Tasks Complete!")
    print(f"{'='*70}")
    print(f"Total configurations: {total_examples}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    if total_examples > 0:
        print(f"Average time per config: {total_time/total_examples:.2f}s")
    print(f"Output directory: {args.output_dir}")
    
    if wandb_run is not None:
        wandb.log({
            "total_time_seconds": total_time,
            "avg_time_per_config": total_time / total_examples if total_examples > 0 else 0,
        })
        wandb.finish()
        print("W&B run finished")


if __name__ == "__main__":
    main()
