;; Tabletop-access domain — face-mode (Bouhsain HAL 2025).
;;
;; Same predicate vocabulary as the filter-mode domain, plus:
;;
;;   * ``face`` type with constants {top, bottom, front, back, left, right}.
;;   * ``(face-grasp-clear ?obj ?face)`` predicate — code-evaluated by the
;;     bridge from the live MuJoCo state.  The simple heuristic in v1:
;;     true iff the cell adjacent to ?obj's current cell in direction
;;     ?face is empty AND the region containing ?obj allows a grasp from
;;     that direction (region.access_modes).
;;
;; Pick and put are parameterised on the chosen face; the planner can
;; therefore reason about exposing or covering specific faces.

(define (domain tabletop-access-face)
  (:requirements :strips :typing :negative-preconditions)

  (:types
    cell movable face
  )

  ;; Faces are declared as ``face``-typed objects in each problem
  ;; instance — the bridge supplies the standard six values (top, bottom,
  ;; front, back, left, right).  Declaring them as constants here would
  ;; collide with that, so we don't.

  (:predicates
    (occupied ?cel - cell ?obj - movable)
    (empty ?cel - cell)
    (holding ?obj - movable)
    (gripper-empty)
    (face-grasp-clear ?obj - movable ?f - face)
  )

  (:action pick
    :parameters (?obj - movable ?f - face ?cel - cell)
    :precondition (and
      (gripper-empty)
      (occupied ?cel ?obj)
      (face-grasp-clear ?obj ?f)
    )
    :effect (and
      (not (gripper-empty))
      (not (occupied ?cel ?obj))
      (empty ?cel)
      (holding ?obj)
    )
  )

  (:action put
    :parameters (?obj - movable ?f - face ?cel - cell)
    :precondition (and
      (holding ?obj)
      (empty ?cel)
      (face-grasp-clear ?obj ?f)
    )
    :effect (and
      (gripper-empty)
      (not (holding ?obj))
      (not (empty ?cel))
      (occupied ?cel ?obj)
    )
  )
)
