"""
Euclidean Landmarks Heuristic scaled by Landmarks
"""

from collections import defaultdict
import copy

import numpy as np

from pyperplan.heuristics.heuristic_base import Heuristic


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


def get_distance_to_landmarks(fact_list, unreached, costs, node):
    # sort the sets in order to later transform the state and goal to points in an n-dim linear space
    state_list = sorted(node.state)

    # initialize state vector with -3 (all possible facts are False)
    state_pos = np.full(shape=len(fact_list), fill_value=(-3))

    unreached_list = sorted(unreached)
    unreached_pos = np.zeros(len(fact_list))

    mult = 2 * (1 / min([costs[landmark] for landmark in unreached], default=1))

    unreached_scalator = np.ones(len(fact_list))

    for i in range(len(fact_list)):
        if fact_list[i] in unreached_list:
            unreached_scalator[i] = costs[fact_list[i]] * mult

    # if a fact is in state/unreached list, that coordinate is set to 1
    for i in range(len(fact_list)):
        if fact_list[i] in state_list:
            state_pos[i] = 1

    for i in range(len(fact_list)):
        if fact_list[i] in unreached_list:
            unreached_pos[i] = 1

    # scale unreached pos
    unreached_pos = np.multiply(unreached_pos, unreached_scalator)

    # scale state pos
    state_pos = np.multiply(state_pos, unreached_scalator)

    # calculates euclidian distance between the current state and goal
    return np.linalg.norm(state_pos - unreached_pos)


class EuclideanLandmarkScaledByLandmarkHeuristic(Heuristic):
    def __init__(self, task):
        self.task = task

        self.landmarks = get_landmarks(task)
        assert self.task.goals <= self.landmarks

        self.fact_list = sorted(task.facts)

        self.costs = compute_landmark_costs(task, self.landmarks)

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
        h = get_distance_to_landmarks(self.fact_list, unreached, self.costs, node)

        return h
