"""
Euclidean Heuristic with unequal Axis
"""

from .heuristic_base import Heuristic

import numpy as np


def get_distance(fact_list, goals_pos, goals_scalator, node):
    # sort the sets in order to later transform the state and goal to points in an n-dim linear space
    state_list = sorted(node.state)

    # initialize state vector with -3 (all possible facts are False)
    state_pos = np.full(shape=len(fact_list), fill_value=(-3))

    # if a fact is in state/goals list, that coordinate is set to 1
    for i in range(len(fact_list)):
        if fact_list[i] in state_list:
            state_pos[i] = 1

    state_pos = np.multiply(state_pos, goals_scalator)

    # calculates euclidian distance between the current state and goal
    return np.linalg.norm(state_pos - goals_pos)


class EuclideanUnequalAxisHeuristic(Heuristic):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goals = task.goals

        self.fact_list = sorted(task.facts)
        self.goals_list = sorted(task.goals)

        # initialize goal vector with zeros (all possible facts are undefined)
        self.goals_pos = np.zeros(len(self.fact_list))

        # if a fact is in goals list, that coordinate is set to 2
        for i in range(len(self.fact_list)):
            if self.fact_list[i] in self.goals_list:
                self.goals_pos[i] = 1

        self.goals_scalator = self.goals_pos + np.ones(len(self.goals_pos))

        self.goals_pos = np.multiply(self.goals_pos, self.goals_scalator)

    def __call__(self, node):
        """Returns the heuristic value for node"""

        return get_distance(self.fact_list, self.goals_pos, self.goals_scalator, node)
