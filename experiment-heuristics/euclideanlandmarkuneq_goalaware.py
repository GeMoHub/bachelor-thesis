"""
Euclidean Landmark Heuristic with unequal axis, goal-aware
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


def get_distance_to_landmarks(fact_list, unreached, node):
    # sort the sets in order to later transform the state and goal to points in an n-dim linear space
    state_list = sorted(node.state)

    # initialize state vector with -1 (all possible facts are False)
    state_pos = np.full(shape=len(fact_list), fill_value=(-3))

    unreached_list = sorted(unreached)
    unreached_pos = np.zeros(len(fact_list))

    # if a fact is in state/goals list, that coordinate is set to 1
    for i in range(len(fact_list)):
        if fact_list[i] in state_list:
            state_pos[i] = 1

    for i in range(len(fact_list)):
        if fact_list[i] in unreached_list:
            unreached_pos[i] = 1

    # creates scalator to scale both unreached pos and state pos by which variables are unreached landmarks
    unreached_scalator = unreached_pos + np.ones(len(unreached_pos))
    unreached_pos = np.multiply(unreached_pos, unreached_scalator)

    state_pos = np.multiply(state_pos, unreached_scalator)

    #also kinda irrelevant here
    min_d = get_distance_to_unreached(unreached_pos, fact_list)

    if all([unreached_pos[i] == state_pos[i] or unreached_pos[i] == 0 for i in range(len(unreached_pos))]):
        return 0

    # calculates euclidian distance between the current state and goal and substracts min_d at the end
    return np.linalg.norm(state_pos - unreached_pos) - min_d


def get_distance_to_unreached(unreached_pos, fact_list):
    """
    calculate d by calculating the distance between goals_pos and the state where all variables are true
    (which is a goal state by definition)
    """
    return np.linalg.norm(np.ones(len(fact_list)) - unreached_pos)


class EuclideanLandmarkUnequalAxisHeuristic(Heuristic):
    def __init__(self, task):
        self.task = task

        self.landmarks = get_landmarks(task)
        assert self.task.goals <= self.landmarks

        self.fact_list = sorted(task.facts)

    def __call__(self, node):
        """Returns the heuristic value for "node"."""
        if node.parent is None:
            # At the beginning only the initial facts are achieved
            node.unreached = self.landmarks - self.task.initial_state
        else:
            # A new node reaches the facts in its add_effects
            node.unreached = node.parent.unreached - node.action.add_effects
        # We always want to keep the goal facts unreached if they are not true
        # in the current state, even if they have been reached before
        unreached = node.unreached | (self.task.goals - node.state)

        h = get_distance_to_landmarks(self.fact_list, unreached, node)
        return h
