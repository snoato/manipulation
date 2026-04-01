;; Blocks world domain for stacking cubes and cuboids
;; Based on classic blocks world with pick-from-table, place-on-table, stack, unstack actions

(define (domain blocks-world)
  (:requirements :strips :typing)
  
  (:types
    block
    gripper
  )
  
  (:predicates
    (on ?x - block ?y - block)          ; Block x is on block y
    (on-table ?x - block)               ; Block x is on the table
    (clear ?x - block)                  ; Block x has nothing on top
    (holding ?g - gripper ?x - block)   ; Gripper g is holding block x
    (gripper-empty ?g - gripper)        ; Gripper g is not holding anything
  )
  
  ;; Pick up a block from the table
  (:action pick-from-table
    :parameters (?g - gripper ?x - block)
    :precondition (and
      (gripper-empty ?g)
      (on-table ?x)
      (clear ?x)
    )
    :effect (and
      (holding ?g ?x)
      (not (gripper-empty ?g))
      (not (on-table ?x))
      (not (clear ?x))
    )
  )
  
  ;; Place a held block on the table
  (:action place-on-table
    :parameters (?g - gripper ?x - block)
    :precondition (and
      (holding ?g ?x)
    )
    :effect (and
      (gripper-empty ?g)
      (on-table ?x)
      (clear ?x)
      (not (holding ?g ?x))
    )
  )
  
  ;; Stack block x on top of block y
  (:action stack
    :parameters (?g - gripper ?x - block ?y - block)
    :precondition (and
      (holding ?g ?x)
      (clear ?y)
    )
    :effect (and
      (gripper-empty ?g)
      (on ?x ?y)
      (clear ?x)
      (not (holding ?g ?x))
      (not (clear ?y))
    )
  )
  
  ;; Unstack block x from block y
  (:action unstack
    :parameters (?g - gripper ?x - block ?y - block)
    :precondition (and
      (gripper-empty ?g)
      (on ?x ?y)
      (clear ?x)
    )
    :effect (and
      (holding ?g ?x)
      (clear ?y)
      (not (gripper-empty ?g))
      (not (on ?x ?y))
      (not (clear ?x))
    )
  )
)
