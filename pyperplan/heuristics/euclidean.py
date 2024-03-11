"""
Euclidean Heuristic
"""
from .heuristic_base import Heuristic

import numpy as np


def get_distance(fact_list, goals_pos, d_min, node):
    # sort the sets in order to later transform the state and goal to points in an n-dim linear space
    state_list = sorted(node.state)

    # initialize state vector with -1 (all possible facts are False)
    state_pos = np.full(shape=len(fact_list), fill_value=(-1))

    # if a fact is in state list, that coordinate is set to 1
    for i in range(len(fact_list)):
        if fact_list[i] in state_list:
            state_pos[i] = 1

    # calculates euclidian distance between the current state and goal and subtracts d_min at the end
    return np.linalg.norm(state_pos - goals_pos) - d_min


def get_distance_to_goal_states(goals_pos, fact_list):
    """
    calculate d_min by calculating the distance between goals_pos and the state where all variables are true
    (which is a goal state by definition)
    """
    return np.linalg.norm(np.ones(len(fact_list)) - goals_pos)


class EuclideanHeuristic(Heuristic):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goals = task.goals

        self.fact_list = sorted(task.facts)
        self.goals_list = sorted(task.goals)

        # initialize goal vector with zeros (all possible facts are undefined)
        self.goals_pos = np.zeros(len(self.fact_list))

        # if a fact is in goals list, that coordinate is set to 1
        for i in range(len(self.fact_list)):
            if self.fact_list[i] in self.goals_list:
                self.goals_pos[i] = 1

        # get d_min
        self.d_min = get_distance_to_goal_states(self.goals_pos, self.fact_list)

    def __call__(self, node):
        """Returns the heuristic value for node"""

        return get_distance(self.fact_list, self.goals_pos, self.d_min, node)
