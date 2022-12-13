import numpy as np
import gym
from gym import utils
# from gym.envs.mujoco import MujocoEnv
from gym_mujoco.envs.mujoco_env import MujocoEnv
from gym.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

class HopperEnv(MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="hopper.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range=(0.7, float("inf")),
        healthy_angle_range=(-0.2, 0.2),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        randomize_slope=False,
        randomize_gravity=False,
        randomize_mass=False,
        randomize_friction=False,
        native_dim=-1,
        encoder_model=None,
        latent_dim=-1,
        hidden_layer=-1,
        model_path=None,
    ):

        self.encoder_model = encoder_model
        self.native_dim = native_dim
        self.latent_dim = latent_dim
        self.ae_model = None
        ae_lower_bounds = {
            1: [-0.6775136586851793], 
            2: [-0.7569554168671038, -0.7175542704217606], 
            3: [-0.2406284703829334, -0.21336030107512796, -0.27406352862719946], 
        }

        ae_upper_bounds = {
            1: [0.5569091809726876], 
            2: [0.6330208509900311, 0.42095026020586734], 
            3: [0.2077969313034738, 0.23864482392285494, 0.20266757536536611], 
        }

        vae_lower_bounds = {
            1: [-0.571912646257999], 
            2: [-1.0, -0.41601168535086086], 
            3: [-0.24274929665123407, 0.11081124773206685, -0.36884726357820347], 
        }
        vae_upper_bounds = {
            1: [0.3090501925438861], 
            2: [1.0, 0.11339946071078572], 
            3: [-0.0009030168812078632, 1.0, 0.14116973879804856], 
        }
        
        if encoder_model == "OTNAE":
            lower_bounds = [-1.0, -0.6369213126722804, -0.23126210637225103] 
            upper_bounds = [0.9881102833932653, 0.3778417659764338, -0.033590182928439266] 
        elif encoder_model == "AE":
            lower_bounds = ae_lower_bounds[latent_dim]
            upper_bounds = ae_upper_bounds[latent_dim]
        elif encoder_model == "VAE":
            lower_bounds = vae_lower_bounds[latent_dim]
            upper_bounds = vae_upper_bounds[latent_dim]
        elif encoder_model == "OTNVAE":
            lower_bounds = [-0.7304396194343641, -0.49846530798684235, -0.20705838677352323] 
            upper_bounds = [0.7125029972878328, 0.6357674990025607, 0.2978712697419127]

        if self.encoder_model != None:
            self.init_ae_model(native_dim, latent_dim, hidden_layer,model_path)
            # self.action_space = gym.spaces.box.Box(np.array(lower_bounds[:latent_dim]), np.array(upper_bounds[:latent_dim]), shape=(latent_dim,))

        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        slope_bounds = None
        gravity_bounds = None
        mass_bounds = None
        friction_bounds = None
        if randomize_slope:
            slope_bounds = [-0.1, +0.1]
        if randomize_gravity:
            gravity_bounds = [0.8, 1.2]
        if randomize_mass:
            mass_bounds = [0.5, 1.5]
        if randomize_friction:
            friction_bounds = [0.5, 1.5]

        MujocoEnv.__init__(self, xml_file, 4, 
                           slope_bounds=slope_bounds,
                           gravity_bounds=gravity_bounds,
                           mass_bounds=mass_bounds,
                           friction_bounds=friction_bounds,)

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.sim.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(self.sim.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        # action = self.reconstruct(action)
        action = self.bottleneck(action, self.latent_dim)
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
        }

        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)