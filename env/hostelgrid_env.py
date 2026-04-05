# env/hostelgrid_env.py

import random
import numpy as np
from env.observation import Observation
from env.action import ACTIONS, get_action_count
from env.reward import calculate_reward


class HostelGridEnv:
    """
    Base environment for HostelGrid++.
    All three task environments inherit from this.
    Keeps base logic clean — tasks add their own complexity on top.
    """
    def __init__(self, num_rooms=20, episode_hours=24):
        self.num_rooms      = num_rooms
        self.episode_hours  = episode_hours
        self.current_hour   = 0
        self.done           = False
        self.obs            = None

    # ------------------------------------------------------------------
    def reset(self):
        self.current_hour = 0
        self.done         = False

        self.obs = Observation(self.num_rooms)
        self.obs.power_usage       = random.uniform(5.0, 15.0)
        self.obs.room_temperatures = [
            random.uniform(22.0, 30.0) for _ in range(self.num_rooms)
        ]
        self.obs.occupancy        = [
            random.randint(0, 1) for _ in range(self.num_rooms)
        ]
        self.obs.complaint_level  = random.randint(0, 3)
        self.obs.time_of_day      = 0
        self.obs.carbon_rate      = self._get_carbon_rate(0)
        self.obs.current_cost     = 0.0

        return self.obs.to_vector()

    # ------------------------------------------------------------------
    def step(self, action):
        prev_complaints  = self.obs.complaint_level
        power_before     = self.obs.power_usage

        # ── Apply action ──────────────────────────────────────────────
        if action == 0:     # increase_ac
            self.obs.power_usage        += 0.8
            self.obs.complaint_level     = max(
                0, self.obs.complaint_level - 1
            )
            self.obs.room_temperatures   = [
                t - 1 for t in self.obs.room_temperatures
            ]

        elif action == 1:   # decrease_ac
            self.obs.power_usage         = max(
                0, self.obs.power_usage - 0.5
            )
            self.obs.complaint_level    += random.randint(0, 2)

        elif action == 2:   # lights_off_empty
            empty = self.obs.occupancy.count(0)
            self.obs.power_usage         = max(
                0, self.obs.power_usage - 0.1 * empty
            )

        elif action == 3:   # lights_on
            self.obs.power_usage        += 0.2

        elif action == 4:   # defer_heavy_load
            self.obs.power_usage         = max(
                0, self.obs.power_usage - 1.5
            )
            self.obs.complaint_level    += random.randint(0, 1)

        elif action == 5:   # do_nothing
            pass

        # ── Advance time ──────────────────────────────────────────────
        self.current_hour         += 1
        self.obs.time_of_day       = self.current_hour
        self.obs.carbon_rate       = self._get_carbon_rate(self.current_hour)
        self.obs.current_cost     += self.obs.power_usage * self._get_tariff(
            self.current_hour
        )

        # ── Reward ────────────────────────────────────────────────────
        power_saved      = max(0, power_before - self.obs.power_usage)
        complaint_delta  = self.obs.complaint_level - prev_complaints
        carbon_saved     = power_saved * self.obs.carbon_rate
        fairness_score   = self._calculate_fairness()

        reward = calculate_reward(
            power_saved     = power_saved,
            complaint_delta = complaint_delta,
            carbon_saved    = carbon_saved,
            fairness_score  = fairness_score,
        )

        self.done = self.current_hour >= self.episode_hours

        info = {
            "hour"       : self.current_hour,
            "power"      : round(self.obs.power_usage, 3),
            "complaints" : self.obs.complaint_level,
            "cost"       : round(
                self.obs.power_usage * self._get_tariff(self.current_hour), 4
            ),
            "carbon_rate": self.obs.carbon_rate,
        }

        return self.obs.to_vector(), reward, self.done, info

    # ------------------------------------------------------------------
    def _get_carbon_rate(self, hour):
        if 9 <= hour <= 12 or 18 <= hour <= 22:
            return 0.82
        elif 0 <= hour <= 5:
            return 0.45
        else:
            return 0.63

    def _get_tariff(self, hour):
        """Rs per kWh"""
        if 9 <= hour <= 12 or 18 <= hour <= 22:
            return 8.5
        elif 0 <= hour <= 5:
            return 4.0
        else:
            return 6.0

    def _calculate_fairness(self):
        if not self.obs.occupancy:
            return 1.0
        occupied = sum(self.obs.occupancy)
        if occupied == 0:
            return 1.0
        return round(occupied / self.num_rooms, 4)