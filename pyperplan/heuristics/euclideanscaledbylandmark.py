"""
Euclidean Heuristic scaled by Landmarks
"""

from collections import defaultdict
import copy

import numpy as np

from .heuristic_base import Heuristic


def _get_relaxed_task(task):
    """
    Removes the delete effects of every operator in task
    """
    relaxed_task = copy.deepcopy(task)
    for op in relaxed_task.operators:
        op.del_effects = set()
    return relaxed_task


def get_landmarks(task):
    """Returns a set of landmarks.

    In this implementation a fact is a landmark if the goal facts cannot be
    reached without it.
    """
    task = _get_relaxed_task(task)
    landmarks = set(task.goals)
    possible_landmarks = task.facts - task.goals

    for fact in possible_landmarks:
        current_state = task.initial_state
        goal_reached = current_state >= task.goals

        while not goal_reached:
            previous_state = current_state

            for op in task.operators:
                if op.applicable(current_state) and fact not in op.add_effects:
                    current_state = op.apply(current_state)
                    if current_state >= task.goals:
                        break
            if previous_state == current_state and not current_state >= task.goals:
                landmarks.add(fact)
                break

            goal_reached = current_state >= task.goals
    return landmarks


def compute_landmark_costs(task, landmarks):
    """
    Compute uniform cost partitioning for actions depending on the landmarks
    they achieve.
    """
    op_to_lm = defaultdict(set)
    for operator in task.operators:
        for landmark in landmarks:
            if landmark in operator.add_effects:
                op_to_lm[operator].add(landmark)
    min_cost = defaultdict(lambda: float("inf"))
    for operator, landmarks in op_to_lm.items():
        landmarks_achieving = len(landmarks)
        for landmark in landmarks:
            min_cost[landmark] = min(min_cost[landmark], 1 / landmarks_achieving)
    return min_cost


def get_distance(fact_list, goals_pos, landmark_cost_pos, node):
    # sort the sets in order to later transform the state and goal to points in an n-dim linear space
    state_list = sorted(node.state)

    # initialize state vector with -3 (all possible facts are False)
    state_pos = np.full(shape=len(fact_list), fill_value=(-3))

    # if a fact is in state list, that coordinate is set to 1
    for i in range(len(fact_list)):
        if fact_list[i] in state_list:
            state_pos[i] = 1

    state_pos = np.multiply(state_pos, landmark_cost_pos)

    # calculates euclidian distance between the current state and goal
    return np.linalg.norm(state_pos - goals_pos)


class EuclideanScaledByLandmarkHeuristic(Heuristic):
    def __init__(self, task):
        self.task = task
        self.goals = task.goals

        self.landmarks = get_landmarks(task)
        assert self.task.goals <= self.landmarks
        self.costs = compute_landmark_costs(task, self.landmarks)

        self.fact_list = sorted(task.facts)
        self.goals_list = sorted(task.goals)
        self.landmarks_list = sorted(self.landmarks)

        # initialize goal vector and landmark cost vector
        self.landmark_cost_pos = np.zeros(len(self.fact_list))
        self.goals_pos = np.zeros(len(self.fact_list))

        self.mult = 2*(1/min([x[1] for x in self.costs.items()]))

        # if a fact is in goals list, that coordinate is set to 1
        for i in range(len(self.fact_list)):
            if self.fact_list[i] in self.landmarks_list:
                self.landmark_cost_pos[i] = self.costs[self.fact_list[i]]*self.mult #use landmark costs
            else:
                self.landmark_cost_pos[i] = 1   #add 1 to all coordinates to allow for entry-wise multiplication,
                                                #and to make one-cost landmarks have a value in this heuristic

        for i in range(len(self.fact_list)):
            if self.fact_list[i] in self.goals_list:
                self.goals_pos[i] = 1

        self.goals_pos = np.multiply(self.goals_pos, self.landmark_cost_pos) #element-wise multiplication


    def __call__(self, node):
        """Returns the heuristic value for "node"."""
        return get_distance(self.fact_list, self.goals_pos, self.landmark_cost_pos, node)
