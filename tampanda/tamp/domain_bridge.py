"""
DomainBridge — connects a PDDL domain to robot manipulation by wiring together
predicate evaluators, action executors, and state samplers registered at runtime.
"""

import itertools
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class DomainBridge:
    """
    Bridges a PDDL domain with robot manipulation by wiring together:

    * **Code-evaluated predicates** — any Python callable ``(env, fluent_state,
      *objects) -> bool`` evaluated each time ``ground_state`` is called.
    * **Fluent predicates** — truth maintained by tracking action effects rather
      than re-evaluating geometry (e.g. ``holding``, ``gripper-empty``).
    * **Action executors** — arbitrary callables mapped to PDDL action names;
      return ``(success: bool, fluent_delta: dict)`` so the bridge can update
      its tracked fluent state automatically.
    * **Per-type samplers** — callables that propose object poses for random
      scene initialisation, with a framework-level retry loop for collision gating.

    Planning is delegated to `unified-planning <https://github.com/aiplan4eu/unified-planning>`_,
    which bridges multiple planners (fast-downward, pyperplan, enhsp, …).

    Example usage::

        bridge = DomainBridge("blocks_domain.pddl", env, strict_preconditions=True)

        @bridge.predicate("on")
        def eval_on(env, fluents, block_top, block_bot):
            # check z-height and xy overlap in sim
            return ...

        bridge.fluent("holding")                          # nothing held initially
        bridge.fluent("gripper-empty", initial=[("g0",)])  # g0 starts empty

        @bridge.action("pick-from-table")
        def exec_pick(env, fluents, gripper, block):
            success = run_pick_motion(env, block)
            return success, {
                ("holding", gripper, block): True,
                ("gripper-empty", gripper): False,
            }

        @bridge.sampler("block")
        def sample_block(env, placed_so_far, rng):
            x = rng.uniform(-0.2, 0.2)
            y = rng.uniform(-0.2, 0.2)
            # return None to reject and retry
            return (x, y, 0.05, np.array([1, 0, 0, 0]))

        objects = {"block": ["b0", "b1", "b2"], "gripper": ["g0"]}
        plan = bridge.plan(objects, goals=[("on", "b0", "b1")])
    """

    def __init__(
        self,
        pddl_domain_path: Union[str, Path],
        environment,
        strict_preconditions: bool = False,
        sampler_max_retries: int = 100,
    ) -> None:
        """
        Args:
            pddl_domain_path: Path to a PDDL domain file.
            environment: Simulation environment (e.g. FrankaEnvironment).
            strict_preconditions: If True, ``execute_action`` verifies PDDL
                preconditions against the current grounded state before calling
                the executor and raises ``RuntimeError`` on violation.
            sampler_max_retries: How many times to retry a sampler that returns
                ``None`` before raising ``RuntimeError``.
        """
        try:
            from unified_planning.io import PDDLReader  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "unified-planning is required: pip install unified-planning[engines]"
            ) from exc

        self.env = environment
        self._strict = strict_preconditions
        self._max_retries = sampler_max_retries

        domain_str = Path(pddl_domain_path).read_text(encoding="utf-8")
        self._domain_str = re.sub(r"\(:requirements[^)]*\)", "", domain_str)
        m = re.search(r"\(define\s+\(domain\s+([^\s)]+)\)", self._domain_str)
        if m is None:
            raise ValueError(f"Could not find domain name in {pddl_domain_path}")
        self._pddl_domain_name = m.group(1)

        # Parse with a minimal dummy problem to extract structural info
        from unified_planning.io import PDDLReader
        dummy = (
            f"(define (problem domain-init) "
            f"(:domain {self._pddl_domain_name}) (:init) (:goal (and)))"
        )
        reader = PDDLReader()
        self._up_domain = reader.parse_problem_string(self._domain_str, dummy)

        self._up_fluents: Dict[str, Any] = {f.name: f for f in self._up_domain.fluents}
        self._up_actions: Dict[str, Any] = {a.name: a for a in self._up_domain.actions}
        self._up_types: Dict[str, Any] = {t.name: t for t in self._up_domain.user_types}

        self._code_predicates: Dict[str, Callable] = {}
        # Flat dict of {(pred_name, *args): bool} for action-tracked predicates.
        # Keys without args are zero-arity: {("gripper-empty",): True}.
        self._fluent_state: Dict[Tuple, bool] = {}
        self._fluent_names: set = set()
        self._action_executors: Dict[str, Callable] = {}
        self._samplers: Dict[str, Callable] = {}

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def predicate(self, name: str) -> Callable:
        """Decorator — register a code-evaluated predicate.

        The decorated function receives ``(env, fluent_state, *typed_objects)``
        and must return a bool.  It is called for every type-consistent
        combination of objects when ``ground_state`` is invoked.

        Example::

            @bridge.predicate("clear")
            def eval_clear(env, fluents, block):
                return not any_block_above(env, block)
        """
        if name not in self._up_fluents:
            raise ValueError(
                f"Predicate '{name}' not found in PDDL domain. "
                f"Available: {list(self._up_fluents)}"
            )

        def decorator(fn: Callable) -> Callable:
            self._code_predicates[name] = fn
            return fn

        return decorator

    def fluent(self, name: str, initial=None) -> None:
        """Register a predicate as action-tracked (not re-evaluated from sim).

        ``_fluent_state`` stores ``{(pred_name, *args): bool}`` — the same flat
        format as ``ground_state`` output.  Action executors update it via the
        ``fluent_delta`` dict they return.

        Args:
            name: PDDL predicate name.
            initial: Starting truth value(s).

                * ``True`` / ``False`` — only for zero-arity predicates.
                * ``None`` — nothing initially True (valid for any arity).
                * ``list`` / ``set`` of arg tuples — each tuple is a
                  combination of object names that is initially True.
                  E.g. ``[("g0",)]`` for a single-parameter predicate.
                * ``dict`` mapping arg-tuple → bool — full explicit control.
        """
        if name not in self._up_fluents:
            raise ValueError(
                f"Predicate '{name}' not found in PDDL domain. "
                f"Available: {list(self._up_fluents)}"
            )
        arity = len(self._up_fluents[name].signature)
        self._fluent_names.add(name)

        if initial is None:
            pass
        elif isinstance(initial, bool):
            if arity != 0:
                raise ValueError(
                    f"Use a list/set/dict of arg tuples for parametric "
                    f"predicate '{name}' (arity {arity})."
                )
            self._fluent_state[(name,)] = initial
        elif isinstance(initial, dict):
            for args, val in initial.items():
                self._fluent_state[(name, *args)] = val
        elif isinstance(initial, (list, set)):
            for args in initial:
                if isinstance(args, str):
                    args = (args,)  # convenience: bare string → 1-tuple
                self._fluent_state[(name, *args)] = True
        else:
            raise TypeError(
                f"Unsupported initial type for fluent '{name}': {type(initial)}"
            )

    def action(self, name: str) -> Callable:
        """Decorator — register an executor for a PDDL action.

        The decorated function receives ``(env, fluent_state, *param_names)``
        where ``*param_names`` are the PDDL object arguments in declaration
        order.  It must return ``(success: bool, fluent_delta: dict)``.
        ``fluent_delta`` is merged into the internal fluent state only on
        success.

        Example::

            @bridge.action("stack")
            def exec_stack(env, fluents, gripper, top, bottom):
                success = plan_and_execute_stack(env, top, bottom)
                return success, {
                    ("holding", gripper, top): False,
                    ("gripper-empty", gripper): True,
                    ("on", top, bottom): True,
                }
        """
        if name not in self._up_actions:
            raise ValueError(
                f"Action '{name}' not found in PDDL domain. "
                f"Available: {list(self._up_actions)}"
            )

        def decorator(fn: Callable) -> Callable:
            self._action_executors[name] = fn
            return fn

        return decorator

    def sampler(self, type_name: str) -> Callable:
        """Decorator — register a pose sampler for a PDDL type.

        The decorated function receives ``(env, placed_so_far: list, rng)``
        and should return a pose (any structure the caller will later use,
        e.g. ``(x, y, z, quat)``) or ``None`` to reject and trigger a retry.

        The framework calls the sampler up to ``sampler_max_retries`` times
        per object before raising ``RuntimeError``.
        """
        def decorator(fn: Callable) -> Callable:
            self._samplers[type_name] = fn
            return fn

        return decorator

    # -------------------------------------------------------------------------
    # State grounding
    # -------------------------------------------------------------------------

    def ground_state(self, objects: Dict[str, List[str]]) -> Dict[Tuple, bool]:
        """Evaluate all registered predicates against the current simulation.

        Code-evaluated predicates are called for every type-consistent
        combination of objects.  Fluent predicates are read from the internal
        tracked state and projected onto object combinations.

        Args:
            objects: ``{type_name: [object_name, ...]}`` for all objects in
                scope for this grounding pass.

        Returns:
            Flat dict ``{(predicate_name, *obj_names): bool}``.
            Zero-arity predicates are keyed as ``(predicate_name,)``.
        """
        state: Dict[Tuple, bool] = {}

        for pred_name, fn in self._code_predicates.items():
            param_types = [
                p.type.name for p in self._up_fluents[pred_name].signature
            ]
            if not param_types:
                state[(pred_name,)] = bool(fn(self.env, self._fluent_state))
            else:
                for combo in itertools.product(*[objects.get(t, []) for t in param_types]):
                    state[(pred_name, *combo)] = bool(
                        fn(self.env, self._fluent_state, *combo)
                    )

        for pred_name in self._fluent_names:
            param_types = [
                p.type.name for p in self._up_fluents[pred_name].signature
            ]
            if not param_types:
                state[(pred_name,)] = self._fluent_state.get((pred_name,), False)
            else:
                for combo in itertools.product(*[objects.get(t, []) for t in param_types]):
                    state[(pred_name, *combo)] = self._fluent_state.get(
                        (pred_name, *combo), False
                    )

        return state

    # -------------------------------------------------------------------------
    # UP problem construction and planning
    # -------------------------------------------------------------------------

    def build_up_problem(
        self,
        objects: Dict[str, List[str]],
        grounded_state: Dict[Tuple, bool],
        goals: List,
        problem_name: str = "problem",
    ):
        """Construct a ``unified_planning.model.Problem`` from a grounded state.

        Args:
            objects: ``{type_name: [object_name, ...]}``
            grounded_state: output of :meth:`ground_state`.
            goals: Each entry is either a ``(pred_name, *obj_names)`` tuple
                (asserting that predicate True), ``("not", pred_name, *obj_names)``
                (negative literal), or a UP ``FNode``.
            problem_name: Name for the UP Problem instance.

        Returns:
            ``unified_planning.model.Problem`` ready for planning.
        """
        obj_lines: List[str] = []
        for type_name, obj_names in objects.items():
            if type_name not in self._up_types:
                raise ValueError(f"Type '{type_name}' not in PDDL domain.")
            if obj_names:
                obj_lines.append(f"  {' '.join(obj_names)} - {type_name}")
        objects_block = "(:objects\n" + "\n".join(obj_lines) + "\n)" if obj_lines else "(:objects)"

        init_lines: List[str] = []
        for key, value in grounded_state.items():
            if not value:
                continue
            pred_name, *obj_names = key
            if obj_names:
                init_lines.append(f"  ({pred_name} {' '.join(obj_names)})")
            else:
                init_lines.append(f"  ({pred_name})")
        init_block = "(:init\n" + "\n".join(init_lines) + "\n)"

        goal_strs: List[str] = []
        for goal in goals:
            if isinstance(goal, tuple):
                if goal[0] == "not":
                    _, pred_name, *obj_names = goal
                    inner = f"({pred_name} {' '.join(obj_names)})" if obj_names else f"({pred_name})"
                    goal_strs.append(f"(not {inner})")
                else:
                    pred_name, *obj_names = goal
                    goal_strs.append(
                        f"({pred_name} {' '.join(obj_names)})" if obj_names else f"({pred_name})"
                    )
            else:
                raise NotImplementedError(
                    "UP FNode goals are not yet supported; express goals as "
                    "(pred_name, *obj_names) or ('not', pred_name, *obj_names) tuples."
                )
        goal_block = "(:goal (and\n  " + "\n  ".join(goal_strs) + "\n))"

        problem_str = (
            f"(define (problem {problem_name})\n"
            f"(:domain {self._pddl_domain_name})\n"
            f"{objects_block}\n"
            f"{init_block}\n"
            f"{goal_block}\n)"
        )

        from unified_planning.io import PDDLReader
        return PDDLReader().parse_problem_string(self._domain_str, problem_str)

    def plan(
        self,
        objects: Dict[str, List[str]],
        goals: List,
        planner_name: Optional[str] = None,
        problem_name: str = "problem",
    ) -> Optional[List[Tuple[str, Tuple[str, ...]]]]:
        """Ground state, build a UP problem, and call a planner.

        Requires at least one UP-compatible planner to be installed
        (e.g. ``pip install up-pyperplan``).

        Args:
            objects: ``{type_name: [object_name, ...]}``
            goals: list of ``(pred_name, *obj_names)`` tuples.
            planner_name: specific UP planner name, or ``None`` to auto-select.
            problem_name: name for the UP problem.

        Returns:
            List of ``(action_name, (param0, param1, ...))`` tuples in plan
            order, or ``None`` if the problem is unsolvable.
        """
        from unified_planning.shortcuts import OneshotPlanner

        state = self.ground_state(objects)
        problem = self.build_up_problem(objects, state, goals, problem_name)

        kwargs = (
            {"name": planner_name}
            if planner_name
            else {"problem_kind": problem.kind}
        )
        with OneshotPlanner(**kwargs) as planner:
            result = planner.solve(problem)

        if result.plan is None:
            return None

        # UP 1.x uses .actions; older versions used .action_instances
        action_list = getattr(result.plan, "actions", None) or result.plan.action_instances
        return [
            (ai.action.name, tuple(arg.object().name for arg in ai.actual_parameters))
            for ai in action_list
        ]

    # -------------------------------------------------------------------------
    # Action execution
    # -------------------------------------------------------------------------

    def execute_action(
        self,
        action_name: str,
        *params: str,
        objects: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Execute a registered action executor.

        If ``strict_preconditions=True`` (set at construction), the PDDL
        preconditions are evaluated against the current grounded state before
        calling the executor.  ``objects`` is required in that case.

        Args:
            action_name: PDDL action name.
            *params: Object name arguments matching the PDDL parameter order.
            objects: Required when ``strict_preconditions=True``.

        Returns:
            ``(success, fluent_delta)`` — on success the internal fluent state
            is updated with ``fluent_delta``.

        Raises:
            ValueError: Action not registered or ``objects`` missing in strict mode.
            RuntimeError: Precondition violated in strict mode.
        """
        if action_name not in self._action_executors:
            raise ValueError(
                f"No executor registered for action '{action_name}'. "
                f"Registered: {list(self._action_executors)}"
            )

        if self._strict:
            if objects is None:
                raise ValueError(
                    "objects dict is required for strict precondition checking."
                )
            state = self.ground_state(objects)
            self._check_preconditions(action_name, params, state)

        success, fluent_delta = self._action_executors[action_name](
            self.env, self._fluent_state, *params
        )
        if success:
            self._fluent_state.update(fluent_delta)

        return success, fluent_delta

    # -------------------------------------------------------------------------
    # State sampling
    # -------------------------------------------------------------------------

    def sample_random_state(
        self,
        type_counts: Dict[str, int],
        seed: Optional[int] = None,
    ) -> Dict[str, List]:
        """Sample random initial poses using registered per-type samplers.

        Samplers are called with ``(env, placed_so_far, rng)``.  Returning
        ``None`` rejects the candidate and triggers a retry (up to
        ``sampler_max_retries`` times per object).

        Args:
            type_counts: ``{type_name: n_objects}``
            seed: Optional integer seed for the RNG passed to samplers.

        Returns:
            ``{type_name: [pose, ...]}`` where *pose* is whatever each sampler
            returned.

        Raises:
            ValueError: No sampler registered for a requested type.
            RuntimeError: A sampler could not place an object within the retry limit.
        """
        rng = np.random.default_rng(seed)
        placed: List = []
        result: Dict[str, List] = {}

        for type_name, count in type_counts.items():
            if type_name not in self._samplers:
                raise ValueError(
                    f"No sampler registered for type '{type_name}'. "
                    f"Registered: {list(self._samplers)}"
                )
            sampler = self._samplers[type_name]
            poses: List = []
            for i in range(count):
                for _ in range(self._max_retries):
                    pose = sampler(self.env, placed, rng)
                    if pose is not None:
                        placed.append(pose)
                        poses.append(pose)
                        break
                else:
                    raise RuntimeError(
                        f"Could not place {type_name}[{i}] after "
                        f"{self._max_retries} retries — sampler kept returning None."
                    )
            result[type_name] = poses

        return result

    # -------------------------------------------------------------------------
    # Precondition checking
    # -------------------------------------------------------------------------

    def _check_preconditions(
        self,
        action_name: str,
        params: Tuple[str, ...],
        state: Dict[Tuple, bool],
    ) -> None:
        action = self._up_actions[action_name]
        param_map = {p.name: v for p, v in zip(action.parameters, params)}

        preconditions = getattr(action, "preconditions", None)
        if preconditions is None:
            precond = getattr(action, "precondition", None)
            preconditions = [precond] if precond is not None else []

        for cond in preconditions:
            if not self._eval_fnode(cond, param_map, state):
                raise RuntimeError(
                    f"Precondition violated for "
                    f"'{action_name}({', '.join(str(p) for p in params)})'."
                )

    def _eval_fnode(
        self,
        node,
        param_map: Dict[str, str],
        state: Dict[Tuple, bool],
    ) -> bool:
        if node.is_and():
            return all(self._eval_fnode(a, param_map, state) for a in node.args)
        if node.is_or():
            return any(self._eval_fnode(a, param_map, state) for a in node.args)
        if node.is_not():
            return not self._eval_fnode(node.args[0], param_map, state)
        if node.is_true():
            return True
        if node.is_false():
            return False
        if node.is_fluent_exp():
            fluent_name = node.fluent().name
            resolved: List[str] = []
            for arg in node.args:
                if arg.is_parameter_exp():
                    resolved.append(param_map.get(arg.parameter().name, ""))
                elif arg.is_object_exp():
                    resolved.append(arg.object().name)
                else:
                    return True
            return state.get((fluent_name, *resolved), False)
        return True

    # -------------------------------------------------------------------------
    # Inspection
    # -------------------------------------------------------------------------

    @property
    def domain_name(self) -> str:
        return self._pddl_domain_name

    @property
    def predicate_names(self) -> List[str]:
        return list(self._up_fluents)

    @property
    def action_names(self) -> List[str]:
        return list(self._up_actions)

    @property
    def type_names(self) -> List[str]:
        return list(self._up_types)

    def describe(self) -> str:
        """Return a human-readable summary of domain structure and registered hooks."""
        lines = [
            f"Domain: {self.domain_name}",
            f"  Types:             {self.type_names}",
            f"  Predicates:        {self.predicate_names}",
            f"  Actions:           {self.action_names}",
            f"  Code predicates:   {list(self._code_predicates)}",
            f"  Fluent predicates: {sorted(self._fluent_names)}",
            f"  Action executors:  {list(self._action_executors)}",
            f"  Samplers:          {list(self._samplers)}",
        ]
        return "\n".join(lines)
