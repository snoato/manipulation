;; Tabletop-access COMBINED action — single ``pick-place`` per move.
;;
;; Replaces the two-step ``pick``/``put`` action pair from the base
;; domain (``domain.pddl``) with a single ``pick-place ?obj ?cf ?ct``
;; that re-arranges an object from one cell to another in one symbolic
;; step.  Halves plan depth — at canonical_18 + return-all the
;; symbolic plan goes from 50 → 25 actions.
;;
;; Gripper / held state is hidden inside the executor.  PDDL never
;; sees ``(holding …)`` or ``(gripper-empty)`` for the combined
;; action: between the two sub-chains the executor relies on FAST
;; mode's ``_finish_in_fast`` post-grasp short-circuit to teleport
;; the arm back to the canonical staging-home before dispatching the
;; put sub-chain.
;;
;; Includes the ``(blocking ?obj ?goal)`` derived predicate from the
;; v4 derived domain — single-file design (no separate planner +
;; rgnet variants) because no PDDL planner will plan from this
;; domain: the plans come from collapsing v4 pick/put pairs via the
;; ``regenerate_combined.py`` tool.
;;
;; Static facts emitted per problem (unchanged from v4):
;;   * ``(adjacent ?dir ?c1 ?c2)`` — grid adjacency
;;   * ``(same-column-front ?cf ?cb)`` — static cube-column ordering
;;     for the ``(blocking …)`` derivation.

(define (domain tabletop-access-combined)
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
    (gripper-empty)
    (same-column-front ?cel-front - cell ?cel-back - cell)
    ;; Derived: ``?obj`` is occupying a cell that is in the same
    ;; cube column as ``?goal`` and strictly in front of it (smaller
    ;; iy).  Drives GNN structural reasoning about which objects need
    ;; clearing before a target cell becomes reachable.
    (blocking ?obj - movable ?goal - cell)
  )

  (:derived (blocking ?obj - movable ?goal - cell)
    (exists (?c - cell)
      (and (occupied ?c ?obj)
           (same-column-front ?c ?goal))))

  (:action pick-place
    :parameters (?obj - movable ?cf - cell ?ct - cell)
    :precondition (and
      (occupied ?cf ?obj)
      (empty ?ct)
    )
    :effect (and
      (not (occupied ?cf ?obj))
      (empty ?cf)
      (not (empty ?ct))
      (occupied ?ct ?obj)
    )
  )
)
