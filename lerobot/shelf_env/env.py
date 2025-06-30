import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.create_env import ShelfPullDataCollector 


# print("Importing ShelfEnv...")
class ShelfEnv(gym.Env):
    # This metadata is used by the render function. 'rgb_array' is needed for video recording.
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, **kwargs):
        super().__init__()
        
        # --- Your Custom Initialization ---
        # E.g., connect to a simulator, set up robot parameters
        print(f"Initializing MyCustomEnv with")
        self.my_simulator = self._connect_to_simulator()
        
        # --- Define Action and Observation Spaces ---
        # These must match the policy's expectations.
        
        # 1. Define the Action Space
        # This should match the output of your policy. For example, a 2D continuous action.
        # Use spaces.Box for continuous actions: (low, high, shape, dtype)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 2. Define the Observation Space
        # This MUST match the `obs_type="pixels_agent_pos"` from the PushT example.
        # It's a dictionary with keys "pixels" and "agent_pos".
        # The shapes and dtypes must also match what the policy was trained on.
        self.observation_space = spaces.Dict(
            {
                # The camera image from your environment
                "pixels": spaces.Box(low=0, high=255, shape=(96, 96, 3), dtype=np.uint8),
                # The state vector (e.g., robot joint positions, end-effector pose)
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            }
        )

    def _connect_to_simulator(self):
        # Placeholder for your simulator connection logic
        print("Connecting to custom simulator...")
        
        #TODO
        experiment_config = {
        "robot_mode": "floating",  # "normal" or "floating"
        "collect_data": False,
        "path_mode": "SE39D",     # "JOINT7D", "SE38D", or "SE39D"
        "simulate": True,         # Whether the RobotEnvironment runs in simulation
        "camera_name": "cameraStatic", # or "cameraWrist"
        
        "shelf_pos_xyz": [.7, 0.05, .25], # Custom shelf position
        "shelf_quaternion": [1, 0, 0, 1], # Identity quaternion (w,x,y,z) for shelf orientation
        "shelf_openings_small": [4, 11],   # Shelf structure params for generate_shelf
        "shelf_equidistant": False,
        
        "num_samples_books": 10,        # Number of different book arrangements to try
        "num_boxes_per_sample": 1,     # Number of books per arrangement (original script used 1 active book)
        "allow_book_yaw": False,       # Allow books to have a random yaw
        "box_size_ranges": {           # Define ranges for book dimensions [width, depth, thickness]
            'x': (.07, .1),          # Book width (along shelf width)
            'y': (.1, .15),          # Book depth (along shelf depth)
            'z': (.01, .025),        # Book thickness (height)
        },
        "h5_filename": "my_shelf_pull_data.h5" # Output HDF5 filename
    }

        self.collector = ShelfPullDataCollector(**experiment_config)
        self.collector.spawn_books_scene()
        self.collector.C.view(True)


        return "simulator_instance"

    def _get_obs(self):
        # Your logic to get the current observation from your simulator
        # This MUST return a dictionary matching self.observation_space
        pixels = np.random.randint(0, 256, size=(96, 96, 3), dtype=np.uint8) # Replace with your camera data
        #points = ...
        agent_pos = np.random.rand(4).astype(np.float32) # Replace with your robot state data
        return {"pixels": pixels, "agent_pos": agent_pos}

    def _get_info(self):
        # Returns auxiliary diagnostic information (optional)
        return {"distance_to_goal": 0.1}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Your logic to reset your environment to an initial state
        # E.g., move robot to starting position
        print("Resetting the environment.")
        
        # collector.C.setJointState(q0)
        # collector.C.delFrame("target_book_0")
        # collector.C.view(False)

        # collector.spawn_books_scene()
        # collector.C.view(True)

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        # Your logic to apply an action to the environment
        # `action` will be a numpy array matching `self.action_space`
        print(f"Executing action: {action}")
        
        # --- Apply action and get new state ---
        self.collector.C.setJointState([action[0], action[1], action[2], 1, 0, 0, 0])  # Assuming the first 7 values are joint angles

        # --- After action, get the new results ---
        observation = self._get_obs()
        reward = 1.0 # Your logic for calculating reward
        terminated = False # Your logic for whether the episode has ended (e.g., task success)
        truncated = False # Your logic for whether the episode was cut short (e.g., time limit)
        info = self._get_info()
        
        # The step function MUST return these five values in this order
        return observation, reward, terminated, truncated, info

    def render(self):
        # This is used to save the video. It must return an RGB image as a numpy array.
        # The shape should be (H, W, 3) and the dtype np.uint8.
        # This could be the same as the "pixels" observation.
        frame = self._get_obs()["pixels"]
        return frame

    def close(self):
        # Clean up any resources (e.g., close simulator connection)
        print("Closing the environment.")
        # self.my_simulator.disconnect()


# TODO maybe overload the BaseRobotEnv class to implement a environments for different robot actions.
# class Pos3DRobotEnv(BaseRobotEnv):
#     def __init__(self):
#         super().__init__(action_dim=3)

#     def _apply_action(self, action):
#         # Here, interpret action as x, y, z
#         print(f"Moving to position: {action}")
#         self.state[:3] = action  # dummy logic
