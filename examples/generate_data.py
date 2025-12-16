"""Example combining symbolic planning with grasping RRT."""

from pathlib import Path
import time
import argparse
import numpy as np

from manipulation import FrankaEnvironment, RRTStar, ControllerStatus
from manipulation.symbolic import GridDomain, StateManager, visualize_grid_state

_HERE = Path(__file__).parent
_XML = _HERE / ".." / "manipulation" / "environments" / "assets" / "franka_emika_panda" / "scene_symbolic.xml"

# Default values
_DEFAULT_OUTPUT_DIR = _HERE / ".." / "data"
_DEFAULT_EXAMPLES = 1000
_DEFAULT_TARGETS_MAX = 7
_DEFAULT_TARGETS_MIN = 3
_DEFAULT_GRID_SIZE = (0.4, 0.3)  # 40cm x 40cm
_DEFAULT_CELL_SIZE = 0.04  # 2cm cells
_DEFAULT_GRID_OFFSET_Y = 0.02


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
        "--num-examples",
        type=int,
        default=_DEFAULT_EXAMPLES,
        help="Number of problem configurations to generate"
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
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Initialize wandb if requested
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            
            # Initialize wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "num_examples": args.num_examples,
                    "min_objects": args.min_objects,
                    "max_objects": args.max_objects,
                    "grid_width": args.grid_width,
                    "grid_height": args.grid_height,
                    "cell_size": args.cell_size,
                    "grid_offset_y": args.grid_offset_y,
                    "seed": args.seed,
                }
            )
            print("W&B logging enabled")
        except ImportError:
            print("WARNING: wandb not installed. Install with: pip install wandb")
            print("Continuing without W&B logging...")
    
    print("=" * 70)
    print("Symbolic Planning Data Generation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Number of examples: {args.num_examples}")
    print(f"  Objects per config: {args.min_objects}-{args.max_objects}")
    print(f"  Grid size: {args.grid_width}m x {args.grid_height}m")
    print(f"  Cell size: {args.cell_size}m")
    print(f"  Viewer: {args.viewer}")
    print(f"  Visualizations: {not args.no_viz}")
    
    # Initialize environment
    print("\n1. Loading environment...")
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    
    # Create grid domain
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
    
    # Create state manager
    state_manager = StateManager(grid, env)
    
    # Create output directories
    output_base = Path(args.output_dir)
    problem_dir = output_base / "tabletop" / "problems"
    viz_dir = output_base / "viz" / "tabletop"
    problem_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"   Problem directory: {problem_dir}")
    if not args.no_viz:
        print(f"   Visualization directory: {viz_dir}")
    
    # Launch viewer if requested
    if args.viewer:
        viewer = env.launch_viewer()
        if not viewer.is_running():
            raise Exception("no viewer available")
        print("   Viewer launched")
    
    # Generation loop
    print("\n3. Generating configurations...")
    start_time = time.time()
    
    for i in range(args.num_examples):
        config_num = i+1
        print(f"\n{'='*70}")
        print(f"Configuration {config_num} ({i+1}/{args.num_examples})")
        print(f"{'='*70}")
        
        # Sample random state
        n_cylinders = np.random.randint(args.min_objects, args.max_objects + 1)
        state_manager.sample_random_state(n_cylinders=n_cylinders)
        
        # Ground and visualize state
        grounded_state = state_manager.ground_state()
        actual_cylinders = len(grounded_state['cylinders'])
        print(f"   Active cylinders: {actual_cylinders}")
        
        # Generate PDDL problem
        cylinders = sorted(state_manager.ground_state()['cylinders'].keys())
        target_cylinder = np.random.choice(cylinders)
        problem_path = problem_dir / f"config_{config_num}.pddl"
        state_manager.generate_pddl_problem(
            f"config-{config_num}",
            problem_path,
            f"(holding gripper1 {target_cylinder})"
        )
        print(f"   Saved PDDL: {problem_path.name}")
        
        # Visualize if enabled
        viz_path = None
        if not args.no_viz:
            viz_path = viz_dir / f"config_{config_num}.png"
            visualize_grid_state(
                state_manager,
                save_path=viz_path,
                title=f"Configuration {config_num}"
            )
            print(f"   Saved visualization: {viz_path.name}")
        
        # Log to wandb
        if wandb_run is not None:
            log_data = {
                "config_num": config_num,
                "n_objects": actual_cylinders,
                "progress": (i + 1) / args.num_examples,
            }
            
            # Log visualization image if available
            if viz_path is not None and viz_path.exists():
                log_data["visualization"] = wandb.Image(str(viz_path))
            
            wandb.log(log_data)
        
        # Rest if viewer is active
        if args.viewer:
            env.rest(2.0)
        
        # Print progress estimate
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (args.num_examples - i - 1)
            print(f"   Progress: {i+1}/{args.num_examples} | "
                  f"Elapsed: {elapsed:.1f}s | "
                  f"Est. remaining: {remaining:.1f}s")
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Generation Complete!")
    print(f"{'='*70}")
    print(f"Total configurations: {args.num_examples}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"Average time per config: {total_time/args.num_examples:.2f}s")
    print(f"Output directory: {output_base}")
    
    if wandb_run is not None:
        wandb.log({
            "total_time_seconds": total_time,
            "avg_time_per_config": total_time / args.num_examples,
        })
        wandb.finish()
        print("W&B run finished")

if __name__ == "__main__":
    main()
