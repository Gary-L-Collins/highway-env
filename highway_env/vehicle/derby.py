from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from highway_env.road.road import Road
from highway_env.types import Vector
from highway_env.vehicle.kinematics import Vehicle
from highway_env import utils

class DerbyCar(Vehicle):
    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0):
        super().__init__(road, position, heading, speed)
        self.got_crashed = False
        self.did_crash = False
        self.crash_angle = 0.0
        self.crash_speed2 = 0.0
        
    
    def _is_colliding(self, other):
        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > self.LENGTH:
            return False
        # Accurate point-inside checks
        c = 0

        self.got_crashed = 0
        self.did_crash = 0
        other.got_crashed = 0
        other.did_crash = 0
        if utils.point_in_rotated_rectangle(self.position, other.position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading):
            print(self," was hit by ",other)
            self.got_crashed = 1
            other.did_crash  = 1
            self.crash_angle  = (self.heading - other.heading)
            other.crash_angle = self.crash_angle
            self.crash_speed2  = np.sum(np.multiply(self.velocity-other.velocity,self.velocity-other.velocity))
            other.crash_speed2 = self.crash_speed2

            c = 1
        if utils.point_in_rotated_rectangle(other.position, self.position, 0.9*self.LENGTH, 0.9*self.WIDTH, self.heading):
            print(self," hit ",other)
            self.did_crash = 1
            other.got_crashed = 1
            self.crash_angle = (self.heading - other.heading)
            other.crash_angle = self.crash_angle
            c = 1
        return c
