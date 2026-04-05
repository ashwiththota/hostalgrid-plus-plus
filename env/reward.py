# env/reward.py

import numpy as np


def calculate_reward(power_saved, complaint_delta,
                     carbon_saved, fairness_score):
    """
    Base multi-objective reward used by hostelgrid_env.
    Called internally — tasks apply additional shaping on top.

    Args:
        power_saved     : kW saved this step (positive = good)
        complaint_delta : change in complaints (positive = more = bad)
        carbon_saved    : gCO2 saved (positive = good)
        fairness_score  : 0-1 (1 = perfectly fair)
    """
    w_energy   = 0.35
    w_comfort  = 0.30
    w_carbon   = 0.20
    w_fairness = 0.15

    reward = (
          w_energy   *  power_saved
        - w_comfort  *  complaint_delta
        + w_carbon   *  carbon_saved
        + w_fairness *  fairness_score
    )

    # Clip to prevent overflow (critical for Task 3)
    reward = np.clip(reward, -50.0, 50.0)
    return round(float(reward), 4)


def calculate_task1_reward(base_reward, violations, demand_satisfaction,
                            cost, complaints, penalty_weights):
    """
    Task 1 — Commitment-Aware reward shaping.
    Priority violations are penalized very heavily.
    """
    reward = (
          base_reward
        + demand_satisfaction * 0.25
        - violations          * penalty_weights["priority_violation"]
        - cost                * penalty_weights["cost"]
        - complaints          * penalty_weights["complaints"]
    )
    return np.clip(reward, -50.0, 50.0)


def calculate_task2_reward(base_reward, violations, fairness_score,
                            misuse_count, enforcement_bonus,
                            cost, complaints, penalty_weights):
    """
    Task 2 — Fair Enforcement reward shaping.
    Misuse must be detected and handled. Fairness is critical.
    """
    reward = (
          base_reward
        + enforcement_bonus
        + fairness_score * 0.3
        - violations     * penalty_weights["priority_violation"]
        - misuse_count   * penalty_weights["misuse"]
        - complaints     * penalty_weights["complaints"]
        - cost           * penalty_weights["cost"]
    )
    return np.clip(reward, -50.0, 50.0)


def calculate_task3_reward(demand_sat, violations, exam_satisfaction,
                            fairness_violation, misuse_ratio,
                            battery_ratio, cost, carbon,
                            peak_violation, system_trust,
                            flagged_count, priority_count,
                            allocatable_power, penalty_weights):
    """
    Task 3 — Crisis Governance reward shaping.
    Full multi-objective: cost + carbon + fairness + trust + commitments.
    """
    reward = 0.0

    # 1. Demand satisfaction
    reward += demand_sat * 4.0

    # 2. Commitment violations
    if violations == 0:
        reward += 1.5
    else:
        reward -= violations * penalty_weights["priority_violation"]

    # 3. Exam room protection
    reward += exam_satisfaction * 1.5

    # 4. Fairness
    if fairness_violation > 1.0:
        reward -= fairness_violation * penalty_weights["fairness_violation"]
    elif fairness_violation < 0.6:
        reward += 1.0

    # 5. Misuse enforcement
    if flagged_count > 0:
        reward += misuse_ratio * 1.5
        if misuse_ratio < 0.3:
            reward -= 1.0

    # 6. Battery management
    if 0.3 < battery_ratio < 0.7:
        reward += 0.5
    elif battery_ratio < 0.1 or battery_ratio > 0.95:
        reward -= 0.5

    # 7. Cost efficiency
    reward -= (cost / 100.0) * penalty_weights["cost"]

    # 8. Carbon footprint
    reward -= (carbon / 50.0) * penalty_weights["carbon_penalty"]

    # 9. Peak management
    if peak_violation:
        reward -= penalty_weights["peak_violation"]
    else:
        reward += 0.3

    # 10. System trust
    if system_trust > 0.9:
        reward += 0.5
    elif system_trust < 0.5:
        reward -= 2.0

    return float(np.clip(reward, -50.0, 50.0))