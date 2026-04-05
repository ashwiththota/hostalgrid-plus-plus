# env/observation.py

import numpy as np

class Observation:
    def __init__(self, num_rooms=20):
        self.num_rooms          = num_rooms
        self.power_usage        = 0.0        # total kW consumed
        self.room_temperatures  = [24.0] * num_rooms
        self.occupancy          = [0] * num_rooms
        self.complaint_level    = 0          # 0-10
        self.time_of_day        = 0          # 0-23
        self.carbon_rate        = 0.63       # gCO2/kWh
        self.current_cost       = 0.0        # cumulative cost

    def to_vector(self):
        return np.array([
            self.power_usage,
            np.mean(self.room_temperatures),
            np.mean(self.occupancy),
            self.complaint_level,
            self.time_of_day,
            self.carbon_rate,
            self.current_cost,
        ], dtype=np.float32)

    def update(self, power_usage, temperatures, occupancy,
               complaint_level, time_of_day, carbon_rate, current_cost):
        self.power_usage       = power_usage
        self.room_temperatures = temperatures
        self.occupancy         = occupancy
        self.complaint_level   = complaint_level
        self.time_of_day       = time_of_day
        self.carbon_rate       = carbon_rate
        self.current_cost      = current_cost

    def __repr__(self):
        return (
            f"Observation("
            f"power={self.power_usage:.2f}kW, "
            f"temp={np.mean(self.room_temperatures):.1f}C, "
            f"complaints={self.complaint_level}, "
            f"hour={self.time_of_day})"
        )