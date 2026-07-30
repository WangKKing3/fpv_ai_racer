import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FPVDroneRaceEnv(gym.Env):
    """
    Custom Environment for FPV Drone Racing that follows gymnasium interface.
    This environment simulates a drone racing scenario where the agent controls a drone to navigate through a race track.
    """

    def __init__(self):
        super(FPVDroneRaceEnv, self).__init__()
        # Define the action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        self.mass= 0.67             # 670g med batteri )
        self.max_thrust_N = 62.4    # Maximum thrust in Newtons. (basert på 5 inch 6s drone med 1880kv motor og gemfan Freestyle 3 blade  5226)
        self.drag_coeff = 0.1       # Drag coefficient
        self.dt = 0.02              # 50 Hz simuleringsfrekvens  

        # Define the race track layout (gates positions
        self.gates = np.array([
            [10.0,  0.0, 2.0],
            [20.0, 10.0, 3.5],
            [10.0, 20.0, 2.0],
            [ 0.0, 10.0, 1.5]
        ], dtype=np.float32)

        self.gate_radius = 2.5  # Radius of the gates for collision detection 2.5 meter

        self.reset()

    def reset(self, seed=None, options=None):
        # Reset the environment and return the initial observation
        super().reset(seed=seed)

        # Initialize the drone's state
        self.pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)  
        self.vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.angles = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.current_gate_index = 0
        self.step = 0
        self.mac_steps = 1500 

        return self._get_obs(), {}

    def _get_obs(self):
        target_gete = self.gates[self.current_gate_index]
        next_gate_idx = (self.current_gate_index + 1) % len(self.gates)
        next_target_gate = self.gates[next_gate_idx]
        rel_target = target_gete - self.pos      
        rel_next_target = next_target_gate - self.pos

        return np.concatenate([ 
            self.vel,
            self.angles, 
            self.ang_vel, 
            rel_target, 
            rel_next_target]).astype(np.float32)

    def step(self, action):
        self.step += 1
        roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd, throttle_cmd = action
        thrust = (throttle_cmd + 1) / 2 * self.max_thrust_N  # Scale throttle to [0, max_thrust_N]
        self.ang_vel = np.array([roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd], dtype=np.float32)
        self.angles += self.ang_vel * self.dt

        r, p, y = self.angles

        fx = thrust * (np.sin(y) * np.sin(r) + np.cos(y) * np.sin(p) * np.cos(r))
        fy = thrust * (np.cos(y) * np.sin(r) - np.sin(y) * np.sin(p) * np.cos(r))
        fz = thrust * (np.cos(p) * np.cos(r)) - self.mass * 9.81  # Subtract gravity

        acc = np.array([fx, fy, fz]) - (self.drag_coeff * self.vel)

        self.vel += acc * self.dt
        self.pos += self.vel * self.dt

        target_gate = self.gates[self.current_gate_index]
        dist_to_gate = np.linalg.norm(self.pos - target_gate)
        dir_to_gate = (target_gate - self.pos) / (dist_to_gate + 1e-6)  # Avoid division by zero
        vel_toward_gate = np.dot(self.vel, dir_to_gate)

        reward = vel_toward_gate * 0.1  # Reward for moving towards the gate
        reward -= 0.05

        terminated = False
        truncated = False

        if self.pos[2] < 0.0 or  dist_to_gate > 50.0:
            reward -= 100.0
            terminated = True
        if self.step >= self.mac_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}