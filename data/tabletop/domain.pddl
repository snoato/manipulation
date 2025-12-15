(define (domain tabletop-manipulation)
  (:requirements :strips :typing)
  
  (:types
    cell cylinder gripper direction - object
  )
  
  (:constants
    north south east west - direction
  )
  
  (:predicates
    (adjacent ?dir - direction ?c1 - cell ?c2 - cell)  ; cell c2 is adjacent to c1 in direction dir
    (occupied ?c - cell ?cyl - cylinder)  ; cell c contains cylinder cyl
    (empty ?c - cell)                     ; cell c is empty
    (holding ?g - gripper ?cyl - cylinder)  ; gripper g is holding cylinder cyl
    (gripper-empty ?g - gripper)       ; gripper g is not holding anything
  )
  
  (:action pick
    :parameters (?g - gripper ?cyl - cylinder ?c - cell)
    :precondition (and
      (gripper-empty ?g)
      (occupied ?c ?cyl)
    )
    :effect (and
      (not (gripper-empty ?g))
      (not (occupied ?c ?cyl))
      (empty ?c)
      (holding ?g ?cyl)
    )
  )

  (:action drop
    :parameters (?g - gripper ?cyl - cylinder)
    :precondition (and
      (holding ?g ?cyl)
    )
    :effect (and
      (gripper-empty ?g)
      (not (holding ?g ?cyl))
    )
  )
)
