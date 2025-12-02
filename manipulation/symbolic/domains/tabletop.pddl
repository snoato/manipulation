(define (domain tabletop-manipulation)
  (:requirements :strips :typing)
  
  (:types
    cell cylinder gripper
  )
  
  (:predicates
    (adjacent ?c1 - cell ?c2 - cell)  ; cells c1 and c2 are adjacent
    (occupied ?c - cell)               ; cell c contains a cylinder
    (holding ?g - gripper ?cyl - cylinder)  ; gripper g is holding cylinder cyl
    (gripper-empty ?g - gripper)       ; gripper g is not holding anything
  )
  
  (:action pick
    :parameters (?g - gripper ?cyl - cylinder ?c - cell)
    :precondition (and
      (gripper-empty ?g)
      (occupied ?c)
    )
    :effect (and
      (not (gripper-empty ?g))
      (not (occupied ?c))
      (holding ?g ?cyl)
    )
  )
  
  (:action place
    :parameters (?g - gripper ?cyl - cylinder ?c - cell)
    :precondition (and
      (holding ?g ?cyl)
      (not (occupied ?c))
    )
    :effect (and
      (gripper-empty ?g)
      (not (holding ?g ?cyl))
      (occupied ?c)
    )
  )
)
