"""
Euclidean Heuristic scaled by Presence
"""

from collections import defaultdict
import copy

import numpy as np

from .heuristic_base import Heuristic


def compute_scalator(task, fact_list):
    scalator = np.ones(len(fact_list))
    ratio_matrix = np.zeros((len(fact_list), 2))
    for i in range(len(fact_list)):
        for operator in task.operators:
            if fact_list[i] in operator.preconditions:
                ratio_matrix[i][0] += 1
            if fact_list[i] in operator.add_effects:
                ratio_matrix[i][1] += 1
        if fact_list[i] in task.goals:
            ratio_matrix[i][0] += 1
        if fact_list[i] in task.initial_state:
            ratio_matrix[i][1] += 1

    for i in range(len(ratio_matrix)):
        if ratio_matrix[i][1] == 0:
            scalator[i] = ratio_matrix[i][0]
        else:
            scalator[i] = ratio_matrix[i][0] / ratio_matrix[i][1]

    return scalator


def get_distance(fact_list, goals_pos, scalator, node):
    # sort the sets in order to later transform the state and goal to points in an n-dim linear space
    state_list = sorted(node.state)

    # initialize state vector with -3 (all possible facts are False)
    state_pos = np.full(shape=len(fact_list), fill_value=(-3))

    # if a fact is in state list, that coordinate is set to 1
    for i in range(len(fact_list)):
        if fact_list[i] in state_list:
            state_pos[i] = 1

    # scale axis according to presence
    state_pos = np.multiply(state_pos, scalator)

    # calculates euclidian distance between the current state and goal
    return np.linalg.norm(state_pos - goals_pos)


class EuclideanScaledByPresenceHeuristic(Heuristic):
    def __init__(self, task):
        self.task = task
        self.goals = task.goals

        self.fact_list = sorted(task.facts)
        self.goals_list = sorted(task.goals)

        self.scalator = compute_scalator(self.task, self.fact_list)
        self.goals_pos = np.zeros(len(self.fact_list))

        for i in range(len(self.fact_list)):
            if self.fact_list[i] in self.goals_list:
                self.goals_pos[i] = 1

        self.goals_pos = np.multiply(
            self.goals_pos, self.scalator
        )  # element-wise multiplication

    def __call__(self, node):
        """Returns the heuristic value for "node"."""
        return get_distance(self.fact_list, self.goals_pos, self.scalator, node)
