"""Sawyer environment for opening and closing a door."""

import os
from gymnasium import spaces

from metaworld.envs.mujoco.utils import reward_utils
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_door_close_v2 import SawyerDoorCloseEnvV2

import numpy as np

initial_states = np.array([[0.00591636, 0.39968333, 0.19493164, 1.0,
                           0.01007495, 0.47104556, 0.10003595]])
goal_states = np.array([[0.29072163, 0.74286009, 0.10003595, 1.0,
                        0.29072163, 0.74286009, 0.10003595]])

class SawyerDoorV2(SawyerDoorCloseEnvV2):
  max_path_length = int(1e8)

  def __init__(self, reward_type='sparse', reset_at_goal=False):
    self._reset_at_goal = reset_at_goal
    goal_low = (-0.25, 0.45, 0.0999)
    goal_high = (0.35, 0.85, 0.1001)
    obj_low = np.full(3, -np.inf)
    obj_high = np.full(3, +np.inf)
    gripper_low = -1
    gripper_high = 1
    self.initial_states = initial_states
  

    super().__init__(
      render_mode='rgb_array',
      # camera_name='topview',
      camera_name='doorview',
    )

    ee_vel_low = np.full(3, -np.inf)
    ee_vel_high = np.full(3, +np.inf)
    obj_vel_low = np.full(3, -np.inf)
    obj_vel_high = np.full(3, +np.inf)
    
    self.proprioceptive_only = False
    self.custom_task_goal = None
    self.add_velocity_info = None
    
    if self.add_velocity_info == 'door': # 14 -> 7+3 +7 =17 
      self.observation_space = spaces.Box(
        np.hstack((self._HAND_SPACE.low, gripper_low, obj_low, obj_vel_low, self._HAND_SPACE.low, gripper_low, goal_low)),
        np.hstack((self._HAND_SPACE.high, gripper_high, obj_high, obj_vel_high, self._HAND_SPACE.high, gripper_high, goal_high))
      )  
    elif self.add_velocity_info == 'ee_door': # 14 -> 7+6 +7 =20
      self.observation_space = spaces.Box(
        np.hstack((self._HAND_SPACE.low, gripper_low, obj_low, ee_vel_low, obj_vel_low, self._HAND_SPACE.low, gripper_low, goal_low)),
        np.hstack((self._HAND_SPACE.high, gripper_high, obj_high, ee_vel_high, obj_vel_high, self._HAND_SPACE.high, gripper_high, goal_high))
      )  
    else: # default
      self.observation_space = spaces.Box(
        np.hstack((self._HAND_SPACE.low, gripper_low, obj_low, self._HAND_SPACE.low, gripper_low, goal_low)),
        np.hstack((self._HAND_SPACE.high, gripper_high, obj_high, self._HAND_SPACE.high, gripper_high, goal_high))
      )

    self.init_config = {
        'obj_init_angle': -np.pi / 3 if not self._reset_at_goal \
                          else 0,  # default initial angle
        # 'obj_init_angle': 0,  # reset initial angle
        'obj_init_pos': np.array([0.1, 0.95, 0.1], dtype=np.float32),
        'hand_init_pos': np.array(
            [0, 0.4, 0.2] if not self._reset_at_goal \
            else [0.29, 0.74, 0.1],
            dtype=np.float32),
    }

    # (end effector pos, handle pos)
    # -pi/3 initial state -> [0.00591636, 0.39968333, 0.19493164, 0.01007495, 0.47104556, 0.10003595]
    # 0 initial state -> [0.00591636, 0.39968333, 0.19493164, 0.29072163, 0.74286009, 0.10003595]

    self.goal = np.array([0.29072163, 0.74286009, 0.10003595, 1.0,
                          0.29072163, 0.74286009, 0.10003595])  # 0 angle state, goal for forward policy
    # self.goal = np.array([0.00591636, 0.39968333, 0.19493164, 1.0,
    #                       0.01007495, 0.47104556, 0.10003595]) # -pi/3 angle state, goal for reverse policy

    self.goal_states = goal_states.copy()
    self.obj_init_pos = self.init_config['obj_init_pos']
    self.obj_init_angle = self.init_config['obj_init_angle']
    self.hand_init_pos = self.init_config['hand_init_pos']

    
    self._partially_observable = False
    self._set_task_called = True
    self._target_pos = self.goal[4:]
    self._reward_type = reward_type

    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': int(np.round(1.0 / self.dt))
    }
    
  
  def set_proprioceptive_only(self, proprioceptive_only):
    self.proprioceptive_only = proprioceptive_only
  
  def set_custom_task_goal(self, custom_task_goal):
    self.custom_task_goal = custom_task_goal

  def set_velocity_info(self, add_velocity_info):
    self.add_velocity_info = add_velocity_info

  def _get_vel_objects(self):
    return self.data.get_geom_xvelp('handle').copy()
  
  def get_endeff_vel(self):
    return self.data.get_body_xvelp('hand').copy()
  @property
  def model_name(self):
    return os.path.join(
      os.path.dirname(os.path.realpath(__file__)), "metaworld_assets/sawyer_xyz", 'sawyer_door_pull.xml')
  
  def _preprocess_obs(self, obs): 
    # xyz and gripper distance for end effector
    ee_xyzg_obj_xyz = obs[:7]
    preprocessed_obs = np.concatenate([
        ee_xyzg_obj_xyz, self.goal,
    ])
    return preprocessed_obs # (14, ) -> then prev_obs assignment causes the problem 

  def step(self, action): 
    obs, reward, done, truncated, info = super().step(action)
    obs = self._preprocess_obs(obs)
    return obs, reward, done, truncated, info 
  
  def reset(self, *args, **kwargs): 
    obs, info = super().reset(*args, **kwargs) 
    obs = self._preprocess_obs(obs) 
    return obs, info


  # need to expose the default goal, useful for multi-goal settings
  def get_next_goal(self):
    return self.goal_states[0]

  def reset_goal(self, goal=None, add_noise = False):
    if goal is None:
      goal = self.get_next_goal()
    if add_noise: 
      goal = goal + np.random.normal(scale=0.002, size = goal.shape)

    self.goal = goal
    self._target_gripper_distance = goal[3]
    self._target_pos = self._handle_goal = goal[4:]
    self._end_effector_goal = goal[:3]

    site_id = self.data.site('goal').id
    self.model.site_pos[site_id] = self._handle_goal

  def reset_model(self):
    self._reset_hand()
    self.objHeight = self.data.geom('handle').xpos[2]

    if True:
    # if self.random_init:
      # add noise to the initial position of the door
      initial_position = self.obj_init_angle
      initial_position += np.random.uniform(0, np.pi / 20) if not self._reset_at_goal \
        else np.random.uniform(-np.pi / 20, 0)

      body_id = self.data.body('door').id
      self.model.body_pos[body_id] = self.obj_init_pos
      self._set_obj_xyz(initial_position)

    self.reset_goal()
    return self._get_obs()
  
  @SawyerDoorCloseEnvV2._Decorators.assert_task_is_set
  def evaluate_state(self, obs, action):
      reward, obj_to_target, in_place = self.compute_reward(obs, action)
      info = {
          'obj_to_target': obj_to_target,
          'in_place_reward': in_place,
          'success': float(obj_to_target <= 0.08),
          'near_object': 0.,
          'grasp_success': 1.,
          'grasp_reward': 1.,
          'unscaled_reward': reward,
      }
      return reward, info

  def compute_reward(self, obs, actions=None):
    _TARGET_RADIUS = 0.05
    tcp = obs[:3]
    obj = obs[4:7]
    target = obs[11:14]

    tcp_to_target = np.linalg.norm(tcp - target)
    tcp_to_obj = np.linalg.norm(tcp - obj)
    obj_to_target = np.linalg.norm(obj - target)

    in_place_margin = np.linalg.norm(self.obj_init_pos - target)
    in_place = reward_utils.tolerance(obj_to_target,
                                bounds=(0, _TARGET_RADIUS),
                                margin=in_place_margin,
                                sigmoid='gaussian',)

    hand_margin = np.linalg.norm(self.hand_init_pos - obj) + 0.1
    hand_in_place = reward_utils.tolerance(tcp_to_obj,
                                bounds=(0, 0.25*_TARGET_RADIUS),
                                margin=hand_margin,
                                sigmoid='gaussian',)

    reward = 3 * hand_in_place + 6 * in_place

    if obj_to_target < _TARGET_RADIUS:
        reward = 10

    if self._reward_type == 'sparse':
      reward = float(self.is_successful(obs=obs))
    
    return [reward, obj_to_target, hand_in_place]

  def is_successful(self, obs=None, proprioceptive_only=False):
    if obs is None:
      obs = self._get_obs()
    if self.proprioceptive_only or proprioceptive_only: # consider only gripper
      return np.linalg.norm(obs[:3] - obs[7:10]) <= 0.05
    else:
      return np.linalg.norm(obs[4:7] - obs[11:14]) <= self.TARGET_RADIUS


  def set_tstar_states(self, hand_pos=None, obj_pos=None):
    # self.sim.model.site_pos[self.model.site_name2id('goal')] = self._handle_goal
      if hand_pos is not None:
        for i in range(hand_pos.shape[0]):
          hand_id = self.data.site(f"hand_{i+1}").id
          self.model.site_pos[hand_id] = hand_pos[i]
      if obj_pos is not None:
        for i in range(obj_pos.shape[0]):
          obj_id = self.data.site(f"obj_{i+1}").id
          self.model.site_pos[obj_id] = obj_pos[i]

      
    
  def get_goal_states(self):
    return self.goal_states.copy()