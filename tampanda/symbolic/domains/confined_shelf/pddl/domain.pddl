;; Confined-shelf rearrangement domain (Wang ICAPS-2022).
;;
;; Reuses the tabletop predicate vocabulary (occupied / empty / holding /
;; gripper-empty) so rgnet's existing predicate handling carries over.
;; Adds:
;;   - 'direction' type with north/east constants + (adjacent ?dir ?c1 ?c2)
;;     static grid relation — the structural signal the GNN propagates
;;     embeddings along; pick/put preconditions don't use it,
;;   - 'color' type + (color-of ?cyl ?c) static predicate for the
;;     colour-by-column goal.
;;
;; Action semantics match the tabletop 'spatial_put' variant: pick removes
;; an object from a cell into the gripper; put places the held object at a
;; specific empty cell.  No drop/discard.
;;
;; The confined-shelf nature of the problem — front-occlusion and lateral
;; gripper-clearance BLOCKING — is NOT encoded symbolically; PDDL only
;; knows occupied/empty (can't pick an empty cell, can't put to an occupied
;; one).  Feasibility (which cylinder blocks which) lives entirely in the
;; external feasibility checker the rearrangement search consults.

(define (domain confined-shelf)
  (:requirements :strips :typing :negative-preconditions)

  (:types
    cell cylinder color direction
  )

  (:constants
    north east - direction
  )

  (:predicates
    (adjacent ?dir - direction ?cel1 - cell ?cel2 - cell) ; static grid edge
    (occupied ?cel - cell ?cyl - cylinder)   ; cell holds the cylinder
    (empty ?cel - cell)                      ; cell is unoccupied
    (holding ?cyl - cylinder)                ; gripper currently holds cyl
    (gripper-empty)                          ; gripper not holding anything
    (color-of ?cyl - cylinder ?c - color)    ; static — cylinder colour group
  )

  (:action pick
    :parameters (?cyl - cylinder ?cel - cell)
    :precondition (and
      (gripper-empty)
      (occupied ?cel ?cyl)
    )
    :effect (and
      (not (gripper-empty))
      (not (occupied ?cel ?cyl))
      (empty ?cel)
      (holding ?cyl)
    )
  )

  (:action put
    :parameters (?cyl - cylinder ?cel - cell)
    :precondition (and
      (holding ?cyl)
      (empty ?cel)
    )
    :effect (and
      (gripper-empty)
      (not (holding ?cyl))
      (not (empty ?cel))
      (occupied ?cel ?cyl)
    )
  )
)
