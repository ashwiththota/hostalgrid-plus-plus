# env/openenv_api.py
from pydantic import BaseModel
from typing import Any, Dict
import numpy as np
import random
from env.hostelgrid_env import HostelGridEnv

class Observation(BaseModel):
    power_usage:     float
    avg_temperature: float
    avg_occupancy:   float
    complaint_level: int
    time_of_day:     int
    carbon_rate:     float
    current_cost:    float

class Action(BaseModel):
    action_id: int

class Reward(BaseModel):
    value:    float
    breakdown: Dict[str, float]
    done:     bool
    info:     Dict[str, Any]

class HostelGridOpenEnv:
    def __init__(self, task_id: str = "task_easy", num_rooms: int = 20):
        self.task_id           = task_id
        self.num_rooms         = num_rooms
        self._env              = HostelGridEnv(num_rooms=num_rooms, episode_hours=24)
        self._obs_vec          = None
        self._step_count       = 0
        self._total_reward     = 0.0
        self._total_violations = 0
        self._total_complaints = 0
        self._total_cost       = 0.0

    def reset(self) -> Observation:
        self._obs_vec          = self._env.reset()
        self._step_count       = 0
        self._total_reward     = 0.0
        self._total_violations = 0
        self._total_complaints = 0
        self._total_cost       = 0.0
        return self._vec_to_obs(self._obs_vec)

    def step(self, action: Action):
        obs_vec, reward_val, done, info = self._env.step(action.action_id)
        self._obs_vec           = obs_vec
        self._step_count       += 1
        self._total_reward     += reward_val
        self._total_complaints += info.get("complaints", 0)
        self._total_cost       += info.get("cost", 0)

        observation = self._vec_to_obs(obs_vec)
        reward = Reward(
            value=reward_val,
            breakdown={
                "power":      info.get("power", 0),
                "complaints": info.get("complaints", 0),
                "cost":       info.get("cost", 0),
                "carbon":     info.get("carbon_rate", 0),
            },
            done=done,
            info=info,
        )
        return observation, reward, done, info

    def state(self) -> Dict[str, Any]:
        return {
            "task_id":          self.task_id,
            "step":             self._step_count,
            "total_reward":     round(self._total_reward, 4),
            "total_complaints": self._total_complaints,
            "total_cost":       round(self._total_cost, 4),
            "done":             self._step_count >= 24,
            "observation":      self._vec_to_obs(
                                    self._obs_vec
                                ).model_dump() if self._obs_vec is not None else {},
        }

    def _vec_to_obs(self, vec: np.ndarray) -> Observation:
        return Observation(
            power_usage     = float(vec[0]),
            avg_temperature = float(vec[1]),
            avg_occupancy   = float(vec[2]),
            complaint_level = int(vec[3]),
            time_of_day     = int(vec[4]),
            carbon_rate     = float(vec[5]),
            current_cost    = float(vec[6]),
        )

    def score(self) -> float:
        if self.task_id == "task_easy":
            return self._score_easy()
        elif self.task_id == "task_medium":
            return self._score_medium()
        elif self.task_id == "task_hard":
            return self._score_hard()
        return 0.0

    def _score_easy(self) -> float:
        s = 0.0
        # Reward positive (25%)
        if self._total_reward > 3.0:    s += 0.25
        elif self._total_reward > 0:    s += 0.10
        # Cost reasonable (25%)
        if self._total_cost < 1200:     s += 0.25
        elif self._total_cost < 1500:   s += 0.10
        # Complaints low (25%)
        if self._total_complaints < 40: s += 0.25
        elif self._total_complaints < 60: s += 0.10
        # Zero violations (25%)
        if self._total_violations == 0: s += 0.25
        return round(min(s, 1.0), 4)

    def _score_medium(self) -> float:
        s = 0.0
        if self._total_reward > 4.0:      s += 0.25
        elif self._total_reward > 0:      s += 0.10
        if self._total_cost < 1000:       s += 0.25
        elif self._total_cost < 1200:     s += 0.10
        if self._total_complaints < 80:   s += 0.25
        elif self._total_complaints < 110: s += 0.10
        if self._total_violations == 0:   s += 0.25
        return round(min(s, 1.0), 4)

    def _score_hard(self) -> float:
        s = 0.0
        if self._total_reward > 6.0:       s += 0.25
        elif self._total_reward > 0:       s += 0.10
        if self._total_cost < 1500:        s += 0.25
        elif self._total_cost < 2000:      s += 0.10
        if self._total_complaints < 60:    s += 0.25
        elif self._total_complaints < 80:  s += 0.10
        if self._total_violations == 0:    s += 0.25
        return round(min(s, 1.0), 4)