import numpy as np
from typing import Tuple
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.road import Road, LaneIndex
from highway_env.types import Vector
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle


class DemolitionDerbyEnv(AbstractEnv):
    """
    A demolition derby environment.
    
    Two vehicles are inclined to collide into the side of the other and
    avoid being struck on the side ("T-Boning").

    """

    CRASH_REWARD: float = 0.1
    """The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""

    GOT_CRASHED_REWARD: float = 0.4
    """The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"]."""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "Continuous",
            },
            "controlled_vehicles": 1,
            "duration": 100.,  # [s]
            "derby_radius": 20.
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
        path = "highway_env.envs.demolition_derby_env.DerbyCar"
        # changing to our vehicle
        change_vehicles(path)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """ 
        Step Forward in time
        Check for exiting boundary and collision 
        If exit boundary, set radial velocity to zero and fix position to radius. 
        """
        r = self.config["derby_radius"];
        #for iCar in range(self.config["controlled_vehicles"]):
        #    vehicle = self.road.vehicles[iCar]
        #    vehicle.act(action[iCar,:])
            
        # action is none as the steps above already fill the actions of the cars
        obs, reward, terminal, info = super().step(action)

        # checking if exits boundary
        for vehicle in self.road.vehicles:
            corners = corner_positions(vehicle)
            corners_r2 = np.dot(corners, np.transpose(corners))
            max_r2 = np.max(corners_r2)
            # if a corner is beyond the circle, fix position and velocity
            if max_r2 > r*r:
                max_r = np.sqrt(max_r2)
                unitC = corner/max_r
                dr = max_r-r
                indx = np.argmax(corners_r2)
                corner = corners[indx,:]
                vel = vehicle.velocity
                # position, movedin the direction of the unit vector of corner and magnitude dr
                vehicle.position = vehicle.position-unitC*dr

                # projection of velocity onto corner to center vector then setting radial velocity to zero
                radial_v = np.dot(unitC, vel)
                vehicle.velocity = vel - unitC*radial_v
        
        info["agents_rewards"] = self._agent_rewards(action, self.controlled_vehicles)



    @staticmethod
    def corner_positions(self, vehicle: "Vehicle" = None)->np.array:
        """
        This method computes the position of each corner with a rotated car.

        """
        if vehicle is None:
            return np.array([0,0],[0,0],[0,0],[0,0])
        c = self.position
        l = self.LENGTH
        w = self.WIDTH
        h = self.heading

        c, s = np.cos(heading), np.sin(heading)
        r = np.array([[c, -s], [s, c]])
        corners = np.array([[l*0.5,w*0.5],[-l*0.5,w*0.5],[-l*0.5,-w*0.5],[l*0.5,-w*0.5]])

        for i in range(4):
            corners[i,:]=r.dot(corners[i,:])+c

        return corners


    def _reward(self, action: np.ndarray) -> float:
        """
        Reward for hitting, and cost for being hit. +-Sin(heading difference)
        """
        return 0

    def _agent_rewards(self, action: int, vehicles: tuple) -> float:
        rewards = []
        for i, vehicle in enumerate(vehicles):
            reward = 0
            reward = self.config["did_crash_reward"][i] * vehicle.did_crash * abs(np.sin(vehicle.crash_angle)) * (vehicle.velocity / vehicle.MAX_SPEED) ** 2
            reward += self.config["got_crashed_reward"][i] * vehicle.got_crashed * abs(np.sin(vehicle.crash_angle))
            rewards.append(reward)
        return tuple(rewards)

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)

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
        
    
    def _is_colliding(self, other):
        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > self.LENGTH:
            return False
        # Accurate point-inside checks
        c = 0
        self.got_crashed = 0
        self.did_crash = 0
        if utils.point_in_rotated_rectangle(self.position, other.position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading):
            self.got_crashed = 1
            self.crash_angle = (self.heading - other.heading)
            c = 1
        if utils.point_in_rotated_rectangle(other.position, self.position, 0.9*self.LENGTH, 0.9*self.WIDTH, self.heading):
            self.did_crash = 1
            self.crash_angle = (self.heading - other.heading)
            c = 1
        return c
    

class MultiAgentDemolitionDerbyEnv(DemolitionDerbyEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                 "type": "MultiAgentAction",
                 "action_config": {
                     "type": "Continuous",
                 }
            },
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }
            },
            "controlled_vehicles": 2,
            "duration": 100.,  # [s]
            "derby_radius": 20.,
            "did_crash_rewards": [1.0, 1.0],
            "got_crashed_rewards": [1.0, 1.0]
        })
        return config


TupleMultiAgentDemolitionDerbyEnv = MultiAgentWrapper(MultiAgentDemolitionDerbyEnv)

register(
    id='demolition_derby-v0',
    entry_point='highway_env.envs:DemolitionDerby',
)
register(
    id='demolition_derby-multi-agent-v0',
    entry_point='highway_env.envs:MultiAgentDemolitionDerbyEnv',
)

register(
    id='demolition_derby-multi-agent-v1',
    entry_point='highway_env.envs:TupleMultiAgentDemolitionDerbyEnv',
)
