import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.create_env import ShelfPullDataCollector 
import robotic as ry
import h5py
from envs.shelf import generate_shelf
from envs.high_level_methods import RobotEnviroment
from envs.book_spawning import generate_random_box_params

# print("Importing ShelfEnv...")
class ShelfEnv(gym.Env):
    # This metadata is used by the render function. 'rgb_array' is needed for video recording.
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, 
                 robot_mode="floating",
                 collect_data=True,
                 path_mode="SE39D",
                 simulate=True,
                 camera_name="cameraStatic",
                 shelf_pos_xyz=None, # e.g. [.8, 0., .3]
                 shelf_quaternion=None, # e.g. [1, 0, 0, 1] (w,x,y,z) or as expected by generate_shelf
                 shelf_openings_small=None, # e.g. [4, 11]
                 shelf_equidistant=False,
                 num_boxes_per_sample=1,
                 allow_book_yaw=False,
                 box_size_ranges= {'x': (.1, .15), 'y': (.14, .23), 'z': (.009, .045)},
                 h5_filename="variable_demo.h5"):
        super().__init__()
        
        # --- Your Custom Initialization ---
        # E.g., connect to a simulator, set up robot parameters
        print(f"Initializing MyCustomEnv with")
        self.robot_mode = robot_mode
        self.collect_data = collect_data
        self.path_mode = path_mode
        self.simulate = simulate
        self.camera_name = camera_name
        self.h5_filename = h5_filename
        self.C = ry.Config()

        self.num_boxes_per_sample = num_boxes_per_sample
        self.allow_book_yaw = allow_book_yaw
        self.box_size_ranges = box_size_ranges
        self.books = []

        self._create_shelf_scene(shelf_pos_xyz, shelf_quaternion, shelf_openings_small, shelf_equidistant)
        self.q0 = self.C.getJointState()

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



    def _create_shelf_scene(self, shelf_pos_xyz, shelf_quaternion, shelf_openings_small, shelf_equidistant):
        # Placeholder for your simulator connection logic
        print("Connecting to custom simulator...")

        # Configure robot based on mode
        if self.robot_mode == "normal":
            self.C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))
            self.prefix = "l_"
            self.gripper_name = "l_gripper"
            self.palm_name = "l_palm"
            table = self.C.getFrame("table")
            if table: # Check if table frame exists
                table.setShape(ry.ST.ssBox, size=[.5, 1., .1, .005]).setColor(np.array([242, 240, 216]) / 255)
            
            # Safely delete frame if it exists
            coll_camera_wrist = self.C.getFrame("panda_collCameraWrist")
            if coll_camera_wrist:
                 self.C.delFrame("panda_collCameraWrist")

        elif self.robot_mode == "floating":
            self.C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaFloatingFixGripper.g'))
            self.gripper_name = "gripper"
            self.palm_name = "palm"
            self.prefix = ""
            
            current_q = self.C.getJointState()
            offset = np.array([.0, 0, .2, 0, 0, 0, 0]) 
            current_q[:len(offset)] += offset 
            self.C.setJointState(current_q)

        else:
            raise ValueError(f"Unknown ROBOT_MODE: {self.robot_mode}")

        # Gripper friction
        finger1 = self.C.getFrame(self.prefix + "finger1")
        finger2 = self.C.getFrame(self.prefix + "finger2")
        if finger1: finger1.setAttribute("friction", 1e5)
        if finger2: finger2.setAttribute("friction", 1e5)
        

        # Shelf setup
        self.shelf_pos = np.array(shelf_pos_xyz) if shelf_pos_xyz is not None else np.array([.8, 0., .3])
        _shelf_quaternion = shelf_quaternion if shelf_quaternion is not None else [1, 0, 0, 1]
        _shelf_openings_small = shelf_openings_small if shelf_openings_small is not None else [4, 11]
        
        generate_shelf(self.C, self.shelf_pos, base_quaternion=_shelf_quaternion,
                       openings_small=_shelf_openings_small, equidistant=shelf_equidistant)

        self.C.addFrame("cameraWP", self.camera_name).setShape(ry.ST.marker, [.1])

        # Shelf frame for book manipulations (consistent naming from generate_shelf is assumed)
        self.shelf_bottom_frame_name = "big_xy_bottom_0_1" 
        self.shelf_bottom_frame = self.C.getFrame(self.shelf_bottom_frame_name)
        if not self.shelf_bottom_frame:
            raise RuntimeError(f"Shelf bottom frame '{self.shelf_bottom_frame_name}' not found. Check shelf generation logic or name.")

        shelf_size_params = self.shelf_bottom_frame.getSize() # [width, depth, thickness, radius]
        self.shelf_width = shelf_size_params[0]
        self.shelf_depth = shelf_size_params[1]
        self.shelf_plate_thickness = shelf_size_params[2] # Thickness of the bottom plate itself

        # This is the size used for generate_random_box_params, interpreted as the spawning surface dimensions.
        # The Z component here is the thickness of the plate books are spawned on.
        self.shelf_dims_for_spawning = (self.shelf_width, self.shelf_depth, self.shelf_plate_thickness)

        # Shelf corner for book positioning logic (bottom-left corner, Z at center of plate)
        shelf_center_pos = self.shelf_bottom_frame.getPosition()[:3]
        self.shelf_corner_ref_point = shelf_center_pos + np.array([-self.shelf_width/2, -self.shelf_depth/2, 0])

        self.C.view(True)

    def _spawn_books_scene(self):
        sample = generate_random_box_params(
            shelf_size=self.shelf_dims_for_spawning, # (shelf_width, shelf_depth, shelf_plate_thickness)
            box_size_ranges=self.box_size_ranges,
            num_samples=1,
            num_boxes=self.num_boxes_per_sample,
            allow_yaw=self.allow_book_yaw
        )

        for i, book_params in enumerate(sample):
            print(sample)
            print(book_params)

            # book_params: [size_x, size_y, size_z, pos_x_on_shelf, pos_y_on_shelf, yaw_angle]
            b_size_x, b_size_y, b_size_z, b_pos_x, b_pos_y, b_pos_z, b_yaw = book_params[0]
            z_offset = (self.shelf_plate_thickness + b_size_z) / 2
            
            book_center_position = self.shelf_corner_ref_point + np.array([b_pos_x, b_pos_y, z_offset])
            
            q_orientation = ry.Quaternion().setRollPitchYaw([0, 0, b_yaw]) 
            
            frame_name = f"target_book_{i}"
            self.books.append(frame_name)
            self.C.addFrame(frame_name) \
                .setPosition(book_center_position) \
                .setQuaternion(q_orientation.asArr()) \
                .setShape(ry.ST.ssBox, size=[b_size_x, b_size_y, b_size_z, 0.005]) \
                .setColor(np.random.rand(3)) \
                .setContact(1) \
                .setMass(.1)

    def _get_obs(self):
        # Your logic to get the current observation from your simulator
        pixels = np.random.randint(0, 256, size=(96, 96, 3), dtype=np.uint8) # Replace with your camera data
        #points = ...
        agent_pos = self.C.getJointState()[:3]  
        return {"pixels": pixels, "agent_pos": agent_pos}

    def _get_info(self):
        # Returns auxiliary diagnostic information (optional)
        return {"distance_to_goal": 0.1}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Your logic to reset your environment to an initial state
        # E.g., move robot to starting position
        print("Resetting the environment.")
        
        self.C.setJointState(self.q0)
        for book in self.books:
            self.C.delFrame(book)
            self.C.view(False)
        self.books = []

        self._spawn_books_scene()

        self.C.view(True)
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
