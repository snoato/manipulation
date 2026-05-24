;; Tabletop-access domain — filter-mode (Bouhsain HAL 2025).
;;
;; Reuses the confined-shelf predicate vocabulary verbatim
;; (occupied / empty / holding / gripper-empty) so the rgnet predicate
;; pipeline carries over.  Renamed type: ``cylinder`` → ``object`` because
;; HAL's items are arbitrary YCB / generic-box shapes, not only cylinders.
;;
;; In filter mode, grasp face is a refinement choice handled by the
;; feasibility checker; PDDL only sees ``(pick ?obj ?cel)`` and
;; ``(put ?obj ?cel)``.  Pick succeeds iff *some* face has a feasible
;; grasp.  See ``domain_face.pddl`` for the parameterised variant.

(define (domain tabletop-access)
  (:requirements :strips :typing :negative-preconditions)

  (:types
    cell movable
  )

  (:predicates
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
