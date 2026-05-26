;; Tabletop-access domain — filter-mode (Bouhsain HAL 2025).
;;
;; Reuses the confined-shelf / access-19 predicate vocabulary
;; (occupied / empty / holding / gripper-empty) and adds the structural
;; ``(adjacent ?dir ?c1 ?c2)`` relation with a ``direction`` type and
;; ``north``/``east`` constants.  Pick/put preconditions DON'T use
;; adjacency — it's the spatial grid signal for the GNN, whose message
;; passing propagates embeddings along these edges to learn the
;; workspace's spatial semantics (front-to-back row order, column
;; adjacency) rather than parsing it from cell-id strings.
;;
;; The ``access`` scene has FOUR disjoint grids — floor_left, floor_right,
;; middle_deck, top_deck.  Adjacency is emitted WITHIN each grid in the
;; problem file (north = +iy depth, east = +ix column); there is no
;; cross-grid adjacency (levels are traversed by the chain executor, not
;; the symbolic layer).
;;
;; In filter mode, grasp face is a refinement choice handled by the
;; feasibility checker; PDDL only sees ``(pick ?obj ?cel)`` and
;; ``(put ?obj ?cel)``.  See ``domain_face.pddl`` for the parameterised
;; variant.

(define (domain tabletop-access)
  (:requirements :strips :typing :negative-preconditions)

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
