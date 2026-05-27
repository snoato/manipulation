"""PDDL domain + problem generation for the dense-YCB tabletop-access fork.

The domain has **per-footprint-size** ``pick_<W>x<H>`` / ``put_<W>x<H>``
action schemas.  An object whose footprint is ``W × H`` cells uses the
``WxH`` schema; the schema's ``W·H`` cell parameters are pinned into a
rigid contiguous block by static ``(adjacent east …)`` / ``(adjacent
north …)`` preconditions, so a put/pick marks *every* covered cell
occupied/empty — multi-cell occupancy stays correct.

Pure STRIPS + typing (no negative/derived/existential preconditions →
pyperplan- and pymimir-friendly; single domain file).  Goals are concrete
``(occupied …)`` literals, as in the parent ``access`` domain.

The static spatial signal for the GNN is the same ``(adjacent ?dir ?c1
?c2)`` relation the sister domains use (north = +iy depth, east = +ix
column); pick/put consume it only as rigid-shape glue, but the encoder
ingests every edge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

DOMAIN_NAME = "tabletop-access-ycb"

Size = Tuple[int, int]   # (W, H) = (dx along east/x, dy along north/y)


def fp_predicate(size: Size) -> str:
    return f"fp_{size[0]}x{size[1]}"


def _cell_var(i: int, j: int) -> str:
    return f"?c{i}_{j}"


def _rigid_adjacency(W: int, H: int) -> List[str]:
    """Static (adjacent …) literals pinning the W×H cell vars into a block."""
    lits: List[str] = []
    for i in range(W):
        for j in range(H):
            if i + 1 < W:
                lits.append(f"(adjacent east {_cell_var(i, j)} {_cell_var(i + 1, j)})")
            if j + 1 < H:
                lits.append(f"(adjacent north {_cell_var(i, j)} {_cell_var(i, j + 1)})")
    return lits


def _action(kind: str, size: Size) -> str:
    W, H = size
    cells = [(i, j) for i in range(W) for j in range(H)]
    cell_vars = " ".join(_cell_var(i, j) for i, j in cells)
    params = f"?o - movable {cell_vars} - cell"
    rigid = _rigid_adjacency(W, H)
    fp = f"({fp_predicate(size)} ?o)"

    if kind == "pick":
        pre = ["(gripper-empty)", fp, *rigid]
        pre += [f"(occupied {_cell_var(i, j)} ?o)" for i, j in cells]
        eff = ["(not (gripper-empty))", "(holding ?o)"]
        for i, j in cells:
            eff.append(f"(not (occupied {_cell_var(i, j)} ?o))")
            eff.append(f"(empty {_cell_var(i, j)})")
    elif kind == "put":
        pre = ["(holding ?o)", fp, *rigid]
        pre += [f"(empty {_cell_var(i, j)})" for i, j in cells]
        eff = ["(gripper-empty)", "(not (holding ?o))"]
        for i, j in cells:
            eff.append(f"(occupied {_cell_var(i, j)} ?o)")
            eff.append(f"(not (empty {_cell_var(i, j)}))")
    else:
        raise ValueError(kind)

    pre_s = "\n      ".join(pre)
    eff_s = "\n      ".join(eff)
    return (
        f"  (:action {kind}_{W}x{H}\n"
        f"    :parameters ({params})\n"
        f"    :precondition (and\n      {pre_s}\n    )\n"
        f"    :effect (and\n      {eff_s}\n    )\n"
        f"  )\n"
    )


def domain_pddl(sizes: Iterable[Size], domain_name: str = DOMAIN_NAME) -> str:
    sizes = sorted(set(sizes))
    fp_preds = "\n    ".join(f"({fp_predicate(s)} ?o - movable)" for s in sizes)
    actions = "\n".join(_action(k, s) for s in sizes for k in ("pick", "put"))
    return (
        f";; {domain_name} — dense YCB tabletop-access (fork of access).\n"
        f";; Per-footprint-size pick_<W>x<H> / put_<W>x<H> schemas; W·H cell\n"
        f";; params pinned into a rigid block by (adjacent …) statics so every\n"
        f";; covered cell's occupancy updates.  Pure STRIPS+typing.\n"
        f";; Footprint sizes present: {', '.join(f'{w}x{h}' for w, h in sizes)}\n\n"
        f"(define (domain {domain_name})\n"
        f"  (:requirements :strips :typing)\n\n"
        f"  (:types cell movable direction)\n\n"
        f"  (:constants north east - direction)\n\n"
        f"  (:predicates\n"
        f"    (adjacent ?dir - direction ?cel1 - cell ?cel2 - cell)\n"
        f"    (occupied ?cel - cell ?obj - movable)\n"
        f"    (empty ?cel - cell)\n"
        f"    (holding ?obj - movable)\n"
        f"    (gripper-empty)\n"
        f"    {fp_preds}\n"
        f"  )\n\n"
        f"{actions}"
        f")\n"
    )


def write_domain_pddl(sizes: Iterable[Size], path: Path,
                      domain_name: str = DOMAIN_NAME) -> None:
    Path(path).write_text(domain_pddl(sizes, domain_name))


# ----------------------------------------------------------------------
# Problem generation
# ----------------------------------------------------------------------

def adjacency_facts(cells: Sequence[str]) -> List[str]:
    """``(adjacent …)`` edges among the given cells (within-region only)."""
    from tampanda.symbolic.workspace import Cell
    present = set(cells)
    facts: List[str] = []
    for cid in cells:
        c = Cell.parse(cid)
        north = f"{c.region}__{c.ix}_{c.iy + 1}"
        east = f"{c.region}__{c.ix + 1}_{c.iy}"
        if north in present:
            facts.append(f"(adjacent north {cid} {north})")
        if east in present:
            facts.append(f"(adjacent east {cid} {east})")
    return facts


def grounded_plan_action(kind: str, obj: str, anchor, fp) -> str:
    """Render one plan step as the grounded PDDL action matching the domain.

    The schema is ``{kind}_{W}x{H}``; cell params follow the domain order
    (i outer, j inner — see :func:`_action`).  ``anchor`` is a Cell;
    ``fp`` an ObjectFootprint (rectangular, so dx·dy cells).
    """
    W, H = fp.dx, fp.dy
    cells = [f"{anchor.region}__{anchor.ix + i}_{anchor.iy + j}"
             for i in range(W) for j in range(H)]
    return f"({kind}_{W}x{H} {obj} {' '.join(cells)})"


def problem_pddl(
    name: str,
    cells: Sequence[str],
    movables: Sequence[str],
    fp_markers: Dict[str, Size],          # obj -> (W,H)
    source_occupied: Dict[str, List[str]],  # obj -> [cell_id, ...] (footprint)
    goal_occupied: Dict[str, List[str]],    # obj -> [cell_id, ...]
    domain_name: str = DOMAIN_NAME,
) -> str:
    """Render a problem.  ``occupied`` is emitted for EVERY footprint cell of
    each placed object; ``empty`` for every cell no object covers."""
    occupied_cells = {c for cl in source_occupied.values() for c in cl}
    init: List[str] = list(adjacency_facts(cells))
    for obj, size in fp_markers.items():
        init.append(f"({fp_predicate(size)} {obj})")
    for obj, cl in source_occupied.items():
        for c in cl:
            init.append(f"(occupied {c} {obj})")
    init += [f"(empty {c})" for c in cells if c not in occupied_cells]
    init.append("(gripper-empty)")

    goal = [f"(occupied {c} {obj})"
            for obj, cl in goal_occupied.items() for c in cl]

    lines = [
        f"(define (problem {name})",
        f"  (:domain {domain_name})",
        "  (:objects",
        f"    {' '.join(movables)} - movable",
        f"    {' '.join(cells)} - cell",
        "  )",
        "  (:init",
        *(f"    {p}" for p in init),
        "  )",
        "  (:goal (and",
        *(f"    {g}" for g in goal),
        "  ))",
        ")",
    ]
    return "\n".join(lines) + "\n"
