;; Tabletop-access domain — rgnet/GNN dataset variant with derived
;; predicates.  Identical to ``domain.pddl`` except for:
;;
;;   * ``:derived-predicates`` requirement
;;   * new ``(blocking ?obj ?goal-cell)`` derived predicate
;;
;; The derived rule says: an object ``?o`` is blocking goal cell
;; ``?goal`` iff it currently occupies a cell in the SAME cube column
;; AND closer to the open -y face than ``?goal``.  Reads
;; ``(same-column-front …)`` static facts emitted per problem (same
;; facts both domain variants see).
;;
;; Why a separate file: unified-planning rejects
;; ``:derived-predicates``; the planner runtime uses ``domain.pddl``
;; (without the requirement).  The rgnet dataset bundle includes
;; BOTH files — rgnet loads this one so its GNN sees the blocking
;; structural signal.  Problem files are unchanged regardless of
;; which domain is loaded (the static ``(same-column-front …)``
;; facts are declared in both domains; ``(blocking …)`` is purely
;; derived here, never appears in problem ``:init``).

(define (domain tabletop-access)
  (:requirements :strips :typing :negative-preconditions
                 :disjunctive-preconditions :existential-preconditions
                 :derived-predicates)

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
    (same-column-front ?cel-front - cell ?cel-back - cell)
    ;; Derived: ``?o`` is occupying a cell that is in the same cube
    ;; column as ``?goal`` and strictly in front of it (smaller iy).
    ;; True iff there exists such a cell.
    (blocking ?obj - movable ?goal - cell)
  )

  (:derived (blocking ?obj - movable ?goal - cell)
    (exists (?c - cell)
      (and (occupied ?c ?obj)
           (same-column-front ?c ?goal))))

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
