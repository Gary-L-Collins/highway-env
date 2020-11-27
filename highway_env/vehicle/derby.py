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
        if utils.rotated_rectangles_intersect((self.position, 0.9*self.LENGTH, 0.9*self.WIDTH, self.heading),(other.position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading))
            # Determine who hit who (striker is the one with the smallest angle between the line connecting the two centers and their heading)
            pos_self=np.array(self.position)
            pos_other=np.array(other.position)
            CenterVector = (pos_other-pos_self)/np.linalg.norm(pos_other-pos_self)
            if self.speed != 0.0:
                SelfHVec = np.array(self.velocity,dtype=np.float32)/np.fabs(1.0*self.speed)
            else:
                SelfHVec = np.array([0,0],dtype=np.float32)
                SelfHVec[0] += CenterVector[1]
                SelfHVec[1] += -1.0*CenterVector[0]
            if other.speed != 0.0:
                OtherHVec = np.array(other.velocity,dtype=np.float32)/np.fabs(1.0*other.speed)
            else:
                OtherHVec = np.array([0,0],dtype=np.float32)
                OtherHVec[0] += CenterVector[1]
                OtherHVec[1] += -1.0*CenterVector[0]
            SelfCosAlpha = SelfHVec[0]*CenterVector[0]+SelfHVec[1]*CenterVector[1]
            OtherCosAlpha = -OtherHVec[0]*CenterVector[0]-OtherHVec[1]*CenterVector[1] #minus because the vector connecting two cars is pointed towards the "other" car

            if SelfCosAlpha>OtherCosAlpha:
                print(self," hit ",other)
                self.did_crash = 1
                other.got_crashed = 1
                self.crash_angle = (self.heading - other.heading)
                other.crash_angle = self.crash_angle
                self.crash_speed2  = np.sum(np.multiply(self.velocity-other.velocity,self.velocity-other.velocity))
                other.crash_speed2 = self.crash_speed2
                c = 1
            elif OtherCosAlpha>SelfCosAlpha:
                print(self," was hit by ",other)
                self.got_crashed = 1
                other.did_crash  = 1
                self.crash_angle  = (self.heading - other.heading)
                other.crash_angle = self.crash_angle
                self.crash_speed2  = np.sum(np.multiply(self.velocity-other.velocity,self.velocity-other.velocity))
                other.crash_speed2 = self.crash_speed2
                c = 1
            else:
                print("Double Collision, both lose")
                self.got_crashed = 1
                other.got_crashed = 1
                self.crash_angle  = (self.heading - other.heading)
                other.crash_angle = self.crash_angle
                self.crash_speed2  = np.sum(np.multiply(self.velocity-other.velocity,self.velocity-other.velocity))
                other.crash_speed2 = self.crash_speed2
                c = 1
        return c
