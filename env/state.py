# env/state.py
# Shared state container passed between env, simulation, and tasks

import numpy as np
from collections import deque


class EpisodeState:
    """
    Tracks full episode-level metrics.
    Used by all three tasks and graders.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.steps               = 0
        self.total_reward        = 0.0
        self.total_cost          = 0.0
        self.total_carbon        = 0.0
        self.total_complaints    = 0
        self.total_violations    = 0
        self.total_misuse        = 0
        self.demand_sat_sum      = 0.0
        self.fairness_sum        = 0.0
        self.peak_violations     = 0
        self.system_trust        = 1.0
        self.events_encountered  = []
        self.reward_history      = deque(maxlen=100)
        self.complaint_history   = deque(maxlen=5)
        self.trust_history       = deque(maxlen=5)

    def update(self, reward, cost, carbon, complaints,
               violations, demand_sat, fairness,
               peak_violation=False, misuse=0):
        self.steps            += 1
        self.total_reward     += reward
        self.total_cost       += cost
        self.total_carbon     += carbon
        self.total_complaints += complaints
        self.total_violations += violations
        self.total_misuse     += misuse
        self.demand_sat_sum   += demand_sat
        self.fairness_sum     += fairness
        self.peak_violations  += int(peak_violation)
        self.reward_history.append(reward)
        self.complaint_history.append(complaints)
        self.trust_history.append(self.system_trust)

    def update_trust(self, violations, complaints):
        if violations > 0:
            self.system_trust *= 0.95
        else:
            self.system_trust = min(1.0, self.system_trust * 1.02)
        if complaints > 3:
            self.system_trust *= 0.98
        self.system_trust = max(0.0, self.system_trust)

    def avg_demand_satisfaction(self):
        return self.demand_sat_sum / max(1, self.steps)

    def avg_fairness(self):
        return self.fairness_sum / max(1, self.steps)

    def is_collapsed(self, complaint_threshold=50):
        return (self.system_trust < 0.2 or
                self.total_complaints > complaint_threshold)

    def summary(self):
        return {
            "demand_satisfaction" : self.avg_demand_satisfaction(),
            "violations"          : self.total_violations,
            "total_cost"          : self.total_cost,
            "total_carbon"        : self.total_carbon,
            "fairness_score"      : self.avg_fairness(),
            "peak_violations"     : self.peak_violations,
            "system_trust"        : self.system_trust,
            "total_complaints"    : self.total_complaints,
            "total_reward"        : self.total_reward,
            "events_encountered"  : len(self.events_encountered),
        }