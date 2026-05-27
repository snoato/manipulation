;; Tabletop-access domain — filter-mode (Bouhsain HAL 2025).
;;
;; Mirrors the tabletop spatial-put PDDL: adds a ``direction`` type
;; with ``north``/``east`` constants plus the ``(adjacent ?dir ?c1 ?c2)``
;; static relation.  Pick/put preconditions don't use adjacency — it's
;; the structural grid signal for the GNN, whose message passing
;; propagates embeddings along these edges to learn the spatial
;; semantics of the workspace.
;;
;; Access-19 has TWO disjoint grids — ``shelf_interior`` (7×7 inside
;; the closed-top cubicle) and ``shelf_top`` (7×7 open deck above).
;; Adjacency is emitted within each grid in the problem file; there
;; is no cross-grid adjacency.
;;
;; In filter mode, grasp face is a refinement choice handled by the
;; feasibility checker; PDDL only sees ``(pick ?obj ?cel)`` and
;; ``(put ?obj ?cel)``.  See ``domain_face.pddl`` for the
;; parameterised variant.

(define (domain tabletop-access)
  (:requirements :strips :typing :negative-preconditions
                 :disjunctive-preconditions :existential-preconditions)

  (:types
    cell movable direction
  )

  (:constants
    north east - direction
  )

  (:predicates
    (adjacent ?dir - direction ?cel1 - cell ?cel2 - cell)
    (occupied ?cel - cell ?obj - movable)
    (empty ?cel - cell)
    (holding ?obj - movable)
    (gripper-empty)
    ;; Static structural relation: ``?cel-front`` is in the same
    ;; cube column as ``?cel-back`` and strictly closer to the open
    ;; -y face (``iy_front < iy_back``).  Emitted per problem in
    ;; ``:init``.  Planner doesn't use it (declared as a no-op);
    ;; powers the ``(blocking …)`` derived predicate in
    ;; ``domain_derived.pddl`` for GNN structural reasoning.
    (same-column-front ?cel-front - cell ?cel-back - cell)
  )

  (:action pick
    :parameters (?obj - movable ?cel - cell)
    :precondition (and
      (gripper-empty)
      (occupied ?cel ?obj)
    )
    :effect (and
      (not (gripper-empty))
      (not (occupied ?cel ?obj))
      (empty ?cel)
      (holding ?obj)
    )
  )

  (:action put
    :parameters (?obj - movable ?cel - cell)
    :precondition (and
      (holding ?obj)
      (empty ?cel)
    )
    :effect (and
      (gripper-empty)
      (not (holding ?obj))
      (not (empty ?cel))
      (occupied ?cel ?obj)
    )
  )
)
