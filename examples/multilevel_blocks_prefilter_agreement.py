"""Agreement test: prefilter must have ZERO false negatives.

For every successfully-executed plan action in the dataset, the
prefilter MUST return ``UNKNOWN`` (let it through to the full executor).
If the prefilter returns ``INFEASIBLE`` on any plan action, that's a
false negative — the filter would silently drop a feasible action,
which is a correctness bug.

Walks every problem in --in-dir/test/L*/, plays its plan step-by-step
symbolically (tracking which cells are occupied), and at each step
invokes ``prefilter.filter_action`` on the action about to execute.
Tally + report all false-negative violations.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Set, Tuple

from tampanda.symbolic.domains.multilevel_blocks.env_builder import (
    MultilevelBlocksConfig,
)
from tampanda.symbolic.domains.multilevel_blocks.prefilter import (
    INFEASIBLE,
    UNKNOWN,
    filter_action,
)


_RX_INIT = re.compile(r"\(:init\s+(.*?)\)\s*\(:goal", re.DOTALL)
_RX_INPRED = re.compile(r"\(in\s+(\w+)\s+(\w+)\)")
_RX_PLAN = re.compile(r"\(([\w-]+)((?:\s+\w+)+)\)")


def parse_initial_occupied(pddl_path: Path) -> Set[str]:
    """Return the set of cells with at least one block in the init state."""
    text = pddl_path.read_text()
    m = _RX_INIT.search(text)
    if m is None:
        return set()
    occ: Set[str] = set()
    for block, cell in _RX_INPRED.findall(m.group(1)):
        if block.startswith(("cube_", "oblong_", "long_")):
            occ.add(cell)
    return occ


def parse_plan(plan_path: Path) -> List[Tuple]:
    out: List[Tuple] = []
    for line in plan_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        m = _RX_PLAN.match(line)
        if m is None:
            continue
        out.append((m.group(1), *m.group(2).split()))
    return out


def evolve_occupied(occupied: Set[str], action: Tuple) -> Set[str]:
    """Apply pick / put effects to the occupied set.  Returns a NEW set."""
    name = action[0]
    new_occ = set(occupied)
    if name.startswith("pick"):
        for cell in action[2:]:
            new_occ.discard(cell)
    elif name.startswith("put"):
        for cell in action[2:]:
            new_occ.add(cell)
    # transforms (make-*, turn-*) don't change occupancy.
    return new_occ


def occupied_to_state(occupied: Set[str]) -> dict:
    """Synthesize the minimal ``(in dummy cell)`` state that filter expects."""
    return {("in", "_dummy", c): True for c in occupied}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path,
                            default=Path("data/multilevel_blocks"))
    parser.add_argument("--levels", type=int, nargs="+",
                            default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--limit-per-level", type=int, default=None,
                            help="cap problems per level (sanity-check mode)")
    args = parser.parse_args()

    cfg = MultilevelBlocksConfig()

    rule_counts: Counter = Counter()
    false_negatives: List[Tuple[str, int, Tuple, str]] = []
    n_actions_total = 0
    n_problems = 0

    for level in args.levels:
        level_dir = args.in_dir / "test" / f"L{level}"
        if not level_dir.exists():
            print(f"WARNING: {level_dir} missing", file=sys.stderr)
            continue
        problems = sorted(level_dir.glob("config_*.pddl"))
        if args.limit_per_level:
            problems = problems[: args.limit_per_level]
        for pddl_path in problems:
            plan_path = Path(str(pddl_path) + ".plan")
            if not plan_path.exists():
                continue
            occupied = parse_initial_occupied(pddl_path)
            plan = parse_plan(plan_path)
            n_problems += 1
            for step_idx, action in enumerate(plan, start=1):
                state = occupied_to_state(occupied)
                decision, reason = filter_action(state, action, cfg)
                n_actions_total += 1
                if decision == INFEASIBLE:
                    rule_counts[reason or "(no rule)"] += 1
                    false_negatives.append(
                        (str(pddl_path.relative_to(args.in_dir)),
                         step_idx, action, reason or ""))
                # advance occupancy.
                occupied = evolve_occupied(occupied, action)

    print(f"Problems scanned:        {n_problems}")
    print(f"Actions evaluated:       {n_actions_total}")
    print(f"False negatives (filter says INFEASIBLE on plan action): "
              f"{len(false_negatives)}")
    if false_negatives:
        print()
        print("Per-rule false-negative counts:")
        for rule, n in rule_counts.most_common():
            print(f"  {n:>4d}  {rule}")
        print()
        print("First 10 false negatives:")
        for problem, step, action, reason in false_negatives[:10]:
            print(f"  {problem}  step {step}  {action}  → {reason}")
        return 1
    print()
    print("AGREEMENT PASS — no false negatives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
