from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from highway_env.road.road import Road
from highway_env.types import Vector
from highway_env.vehicle.kinematics import Vehicle
from highway_env import utils

def corner_positions(v: "Vehicle" = None)->np.array:
    """
    This method computes the position of each corner with a rotated car.

    """
    c1=v.position
    l1=v.LENGTH
    w1=v.WIDTH
    a1=v.heading
    c1 = np.array(c1)
    l1v = np.array([l1/2., 0.])
    w1v = np.array([0., w1/2.])
    r1_points = np.array([- l1v - w1v, - l1v + w1v, + l1v - w1v, + l1v + w1v])
    c, s = np.cos(a1), np.sin(a1)
    r = np.array([[c, -s], [s, c]])
    corners = r.dot(r1_points.transpose()).transpose()
    corners = corners+c1
    print(corners)
    return corners

def find_initial_impact(v1: "Vehicle" = None, v2: "Vehicle" = None)->np.array:
    """
    Uses shooter method to find point of initial contact backwards in time
    """
    
    # first find time in past where cars don't intersect
    
    c=utils.rotated_rectangles_intersect((v1.position,v1.LENGTH,v1.WIDTH,v1.heading),(v2.position,v2.LENGTH,v2.WIDTH,v2.heading))
    v1pos = v1.position.copy()
    v2pos = v2.position.copy()
    t=.25
    while c:
        t+=.25
        v1pos[0]=v1.position[0]-v1.velocity[0]*t
        v1pos[1]=v1.position[1]-v1.velocity[1]*t
        v2pos[0]=v2.position[0]-v2.velocity[0]*t
        v2pos[1]=v2.position[1]-v2.velocity[1]*t
        c=utils.rotated_rectangles_intersect((v1pos,v1.LENGTH,v1.WIDTH,v1.heading),(v2pos,v2.LENGTH,v2.WIDTH,v2.heading))
    
    t1=0
    t2=t
    if utils.rotated_rectangles_intersect((v1.position,v1.LENGTH,v1.WIDTH,v1.heading),(v2.position,v2.LENGTH,v2.WIDTH,v2.heading)):
        # Shooter method to find exactly when the cars hit
        while t2-t1>10E-8:
            t=(t2+t1)/2
            v1pos[0]=v1.position[0]-v1.velocity[0]*t
            v1pos[1]=v1.position[1]-v1.velocity[1]*t
            v2pos[0]=v2.position[0]-v2.velocity[0]*t
            v2pos[1]=v2.position[1]-v2.velocity[1]*t
            if utils.rotated_rectangles_intersect((v1pos,v1.LENGTH,v1.WIDTH,v1.heading),(v2pos,v2.LENGTH,v2.WIDTH,v2.heading)):
                t1 = t
            else:
                t2 = t

        v1pos[0]=v1.position[0]-v1.velocity[0]*t1
        v1pos[1]=v1.position[1]-v1.velocity[1]*t1
        v2pos[0]=v2.position[0]-v2.velocity[0]*t1
        v2pos[1]=v2.position[1]-v2.velocity[1]*t1
        
        ncorner=0.
        corner_avg=np.array([0.,0.])
        vpos1tmp = v1.position.copy()
        v1.position = v1pos.copy()
        vpos2tmp = v2.position.copy()
        v2.position = v2pos.copy()
        print("HERE")
        print(utils.rotated_rectangles_intersect((v1pos,v1.LENGTH,v1.WIDTH,v1.heading),(v2pos,v2.LENGTH,v2.WIDTH,v2.heading)))
        # Determine which corner
        if utils.has_corner_inside((v1pos,v1.LENGTH,v1.WIDTH,v1.heading),(v2pos,v2.LENGTH,v2.WIDTH,v2.heading)):
            print(utils.has_corner_inside((v1pos,v1.LENGTH,v1.WIDTH,v1.heading),(v2pos,v2.LENGTH,v2.WIDTH,v2.heading)))
            #V1 insind V2
            corners = corner_positions(v1)
            for i in range(4):
                if utils.point_in_rotated_rectangle(corners[i,:],v2.position,v2.LENGTH*1.01,v2.WIDTH*1.01,v2.heading):
                    print("hit")
                    corner_avg[0]+=corners[i,0]
                    corner_avg[1]+=corners[i,1]
                    ncorner+=1.
        if utils.has_corner_inside((v2pos,v2.LENGTH,v2.WIDTH,v2.heading),(v1pos,v1.LENGTH,v1.WIDTH,v1.heading)):
            print(utils.has_corner_inside((v2pos,v2.LENGTH,v2.WIDTH,v2.heading),(v1pos,v1.LENGTH,v1.WIDTH,v1.heading)))
            #V2 insind V1
            corners = corner_positions(v2)
            for i in range(4):
                if utils.point_in_rotated_rectangle(corners[i,:],v1.position,v1.LENGTH*1.01,v1.WIDTH*1.01,v1.heading):
                    print("hit")
                    corner_avg[0]+=corners[i,0]
                    corner_avg[1]+=corners[i,1]
                    ncorner+=1.
        v1.position = vpos1tmp.copy()
        v2.position = vpos2tmp.copy()
        if ncorner < 1.:
            return corners[i,:]
        corner_avg[0] /= ncorner
        corner_avg[1] /= ncorner
        
        
        return corner_avg
    else:
        return np.array([0.0, 0.0])


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

        #if utils.point_in_rotated_rectangle(self.position, other.position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading) or utils.point_in_rotated_rectangle(other.position, self.position, 0.9*self.LENGTH, 0.9*self.WIDTH, self.heading):
        if utils.rotated_rectangles_intersect((self.position,self.LENGTH*.9,self.WIDTH*.9,self.heading),(other.position,other.LENGTH*.9,other.WIDTH*.9,other.heading)):
            # Determine who hit who (striker is the one with the smallest angle between the line connecting the two centers and their heading)
            POI=find_initial_impact(self,other)
            print(POI)

            pos_self=np.array(self.position)
            pos_other=np.array(other.position)
            CenterVectorSelf = (POI-pos_self)/np.linalg.norm(POI-pos_self)
            CenterVectorOther = (POI-pos_other)/np.linalg.norm(POI-pos_other)
            if self.speed != 0.0:
                SelfHVec = np.array(self.velocity,dtype=np.float32)/np.fabs(1.0*self.speed)
            else:
                SelfHVec = np.array([0,0],dtype=np.float32)
                SelfHVec[0] += CenterVectorSelf[1]
                SelfHVec[1] += -1.0*CenterVectorSelf[0]
            if other.speed != 0.0:
                OtherHVec = np.array(other.velocity,dtype=np.float32)/np.fabs(1.0*other.speed)
            else:
                OtherHVec = np.array([0,0],dtype=np.float32)
                OtherHVec[0] += CenterVectorOther[1]
                OtherHVec[1] += -1.0*CenterVectorOther[0]
            SelfCosAlpha = np.fabs(SelfHVec[0]*CenterVectorSelf[0]+SelfHVec[1]*CenterVectorSelf[1])
            OtherCosAlpha = np.fabs(OtherHVec[0]*CenterVectorOther[0]+OtherHVec[1]*CenterVectorOther[1]) #minus because the vector connecting two cars is pointed towards the "other" car

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

