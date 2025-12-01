"""RRT* motion planner implementation."""

import numpy as np
from typing import List, Optional

from manipulation.core.base_mp import BaseMotionPlanner


class Node:
    """RRT* tree node."""
    
    def __init__(self, config: np.ndarray):
        self.config = config.copy()
        self.parent: Optional[Node] = None
        self.cost: float = 0.0
        self.children: List[Node] = []


class RRTStar(BaseMotionPlanner):
    """RRT* motion planner for robot manipulation."""

    def __init__(
        self,
        environment,
        max_iterations: int = 5000,
        step_size: float = 0.1,
        search_radius: float = 0.5,
        goal_threshold: float = 0.05
    ):
        self.env = environment
        self.model = environment.get_model()
        self.data = environment.get_data()
        self.ik = environment.get_ik()
        
        # RRT* parameters
        self.max_iterations = max_iterations
        self.goal_sample_rate = 0.1
        self.step_size = step_size
        self.search_radius = search_radius
        self.goal_threshold = goal_threshold
        
        # Joint limits for sampling
        self.joint_limits_low = self.model.jnt_range[:7, 0].copy()
        self.joint_limits_high = self.model.jnt_range[:7, 1].copy()
    
    def sample_random_config(self) -> np.ndarray:
        return np.random.uniform(self.joint_limits_low, self.joint_limits_high)
    
    def distance(self, config1: np.ndarray, config2: np.ndarray) -> float:
        return np.linalg.norm(config1 - config2)
    
    def nearest_node(self, tree: List[Node], config: np.ndarray) -> Node:
        return min(tree, key=lambda node: self.distance(node.config, config))
    
    def steer(self, from_config: np.ndarray, to_config: np.ndarray) -> np.ndarray:
        direction = to_config - from_config
        dist = np.linalg.norm(direction)
        if dist <= self.step_size:
            return to_config.copy()
        return from_config + (direction / dist) * self.step_size
    
    def is_path_collision_free(
        self,
        config1: np.ndarray,
        config2: np.ndarray,
        steps: int = 10
    ) -> bool:
        for i in range(steps + 1):
            alpha = i / steps
            config = (1 - alpha) * config1 + alpha * config2
            if not self.env.is_collision_free(config):
                return False
        return True
    
    def near_nodes(self, tree: List[Node], config: np.ndarray, radius: float) -> List[Node]:
        return [node for node in tree if self.distance(node.config, config) < radius]
    
    def choose_parent(self, new_node: Node, near_nodes: List[Node]) -> Node:
        if not near_nodes:
            return new_node
        
        best_parent = new_node.parent
        min_cost = new_node.cost
        
        for near_node in near_nodes:
            potential_cost = near_node.cost + self.distance(near_node.config, new_node.config)
            if potential_cost < min_cost and self.is_path_collision_free(
                near_node.config, new_node.config
            ):
                best_parent = near_node
                min_cost = potential_cost
        
        if best_parent != new_node.parent:
            if new_node.parent:
                new_node.parent.children.remove(new_node)
            new_node.parent = best_parent
            new_node.cost = min_cost
            best_parent.children.append(new_node)
        
        return new_node
    
    def rewire(self, tree: List[Node], new_node: Node, near_nodes: List[Node]):
        for near_node in near_nodes:
            if near_node == new_node or near_node == new_node.parent:
                continue
            
            potential_cost = new_node.cost + self.distance(new_node.config, near_node.config)
            if potential_cost < near_node.cost and self.is_path_collision_free(
                new_node.config, near_node.config
            ):
                if near_node.parent:
                    near_node.parent.children.remove(near_node)
                near_node.parent = new_node
                new_node.children.append(near_node)
                self._update_costs_recursive(near_node, potential_cost)
    
    def _update_costs_recursive(self, node: Node, new_cost: float):
        cost_diff = new_cost - node.cost
        node.cost = new_cost
        for child in node.children:
            self._update_costs_recursive(child, child.cost + cost_diff)

    def plan(
        self,
        start_config: np.ndarray,
        goal_config: np.ndarray,
        max_iterations: Optional[int] = None
    ) -> Optional[List[np.ndarray]]:
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        if not self.env.is_collision_free(start_config):
            print("Start configuration is in collision!")
            return None
        
        if not self.env.is_collision_free(goal_config):
            print("Goal configuration is in collision!")
            return None
        
        start_node = Node(start_config)
        tree = [start_node]
        goal_node = None
        
        print(f"Planning path with RRT* (max iterations: {max_iterations})...")
        
        for iteration in range(max_iterations):
            # Sample configuration
            if np.random.random() < self.goal_sample_rate:
                random_config = goal_config.copy()
            else:
                random_config = self.sample_random_config()
            
            nearest = self.nearest_node(tree, random_config)
            new_config = self.steer(nearest.config, random_config)
            
            if not self.is_path_collision_free(nearest.config, new_config):
                continue
            
            new_node = Node(new_config)
            new_node.parent = nearest
            new_node.cost = nearest.cost + self.distance(nearest.config, new_config)
            nearest.children.append(new_node)
            
            # Adaptive radius
            radius = min(
                self.search_radius,
                self.search_radius * np.power(np.log(len(tree)) / len(tree), 1.0 / 7.0)
            )
            near = self.near_nodes(tree, new_config, radius)
            
            new_node = self.choose_parent(new_node, near)
            tree.append(new_node)
            self.rewire(tree, new_node, near)
            
            # Check goal
            if self.distance(new_config, goal_config) < self.goal_threshold:
                if self.is_path_collision_free(new_config, goal_config):
                    goal_node = Node(goal_config)
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + self.distance(new_config, goal_config)
                    new_node.children.append(goal_node)
                    tree.append(goal_node)
                    break
        
        if goal_node is None:
            print("Failed to find path to goal.")
            return None
        
        path = self._extract_path(goal_node)
        print(f"Path found with {len(path)} waypoints, iteration: {iteration + 1}")
        return path
    
    def _extract_path(self, goal_node: Node) -> List[np.ndarray]:
        path = []
        current = goal_node
        while current is not None:
            path.append(current.config.copy())
            current = current.parent
        path.reverse()
        return path
    
    def smooth_path(
        self,
        path: List[np.ndarray],
        max_iterations: int = 100
    ) -> List[np.ndarray]:
        if len(path) <= 2:
            return path
        
        smoothed = [config.copy() for config in path]
        for _ in range(max_iterations):
            if len(smoothed) <= 2:
                break
            i = np.random.randint(0, len(smoothed) - 2)
            j = np.random.randint(i + 2, len(smoothed))
            if self.is_path_collision_free(smoothed[i], smoothed[j]):
                smoothed = smoothed[:i+1] + smoothed[j:]
        return smoothed
    
    def plan_to_pose(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        dt: float = 0.01,
        max_iterations: Optional[int] = None
    ) -> Optional[List[np.ndarray]]:
        start_config = self.data.qpos[:7].copy()
        self.ik.update_configuration(self.data.qpos)
        self.ik.set_target_position(target_pos, target_quat)
        
        if not self.ik.converge_ik(dt):
            print("IK failed to converge for target pose")
            return None
        
        goal_config = self.ik.configuration.q[:7].copy()
        return self.plan(start_config, goal_config, max_iterations)
    
    def get_path_cost(self, path: List[np.ndarray]) -> float:
        if len(path) < 2:
            return 0.0
        return sum(self.distance(path[i], path[i + 1]) for i in range(len(path) - 1))
