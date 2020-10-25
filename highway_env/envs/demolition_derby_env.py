import numpy as np
from typing import Tuple
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle


class DemolitionDerbyEnv(AbstractEnv):
    """
    A demolition derby environment.
    
    Two vehicles are inclined to collide into the side of the other and
    avoid being struck on the side ("T-Boning").

    """

    RIGHT_LANE_REWARD: float = 0.1
    """The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""

    HIGH_SPEED_REWARD: float = 0.4
    """The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"]."""

    LANE_CHANGE_REWARD: float = 0
    """The reward received at each lane change action."""

    def default_config(self) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "Continuous",
            },
            "controlled_vehicles": 2,
            "duration": 100.,  # [s]
            "derby_radius": 20.,
            "time_steps": 100.
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """ circular road that acts as boundarys for derby """
        center = [0, 0]  # [m]
        alpha = 24  # [deg]
        radius = self.config["derby_radius"]

        net = RoadNetwork()
        radii = [radius, radius+4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        net.add_lane("e", "w", CircularLane(center, radius, 0, np.pi, clockwise=False, line_types=line[0], speed_limit=1000))
        net.add_lane("w", "e", CircularLane(center, radius, np.pi, 0, clockwise=False, line_types=line[0], speed_limit=1000))
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            XPos = self.np_random.rand()*self.config["derby_radius"]*1.5-self.config["derby_radius"]*.75
            YPos = self.np_random.rand()*self.config["derby_radius"]*1.5-self.config["derby_radius"]*.75
            Heading = 2*np.pi*self.np_random.rand()
            Speed = 0.
            vehicle = self.action_type.vehicle_class(self.road, [XPos, YPos], Heading, Speed)
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)

    def step(self) -> Tuple[np.ndarray, float, bool, dict]:
        """ 
        Step Forward in time
        Check for exiting boundary and collision 
        If exit boundary, set radial velocity to zero. 
          1) This is done by projecting position to radial coordinates 
               (e.g. x,y -> r,theta via r2 = x2+y2, theta = atan(abs(y/x)) * a + b,
               a = sign(x*y), b = 0 if x>0, pi if x<0).
          2) Then setting r = rmax, x = r cos(theta), y = r sin(theta)
          3) r' = 0, x' = -r sin(theta)*theta', y = r cos(theta) theta'
               where theta' = 1/r * (-x'sin(theta)+y'cos(theta))
        If collision, determine striker and award accordingly
          1) collisions are checked with _check_collision func.
          2) linear approximation can be used to "roll back" time before collision
           
        """
        dt = self.config["duration"]/self.config["time_steps"]

        # updating position
        for vehicle in self.road.vehicles:
            vehicle.step(dt)

        # checking if exits boundary
        for vehicle in self.road.vehicles:


        # checking for collisions
        for vehicle in self.road.vehicles:
            for other in self.road.vehicles:
                vehicle.check_collision(other)
            for other in self.road.objects:
                vehicle.check_collision(other)

        # update all positions

#        for iCar in range(self.config["controlled_vehicles"]):
#            Car = self.road.vehicles[iCar]
#            Cardd.step(dt)




    def _reward(self, action: Action) -> float:
        """
        Reward for hitting, and cost for being hit. +-Sin(heading difference)
        """
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


register(
    id='demolition_derby-v0',
    entry_point='highway_env.envs:DemolitionDerby',
)
