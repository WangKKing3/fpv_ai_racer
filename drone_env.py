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

        return self._get_observation(), {}

    def step(self, action):
        # Execute one time step within the environment
        pass

    def render(self):
        # Render the environment
        pass