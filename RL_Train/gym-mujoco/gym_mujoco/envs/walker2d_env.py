import gym.spaces.box
import numpy as np
import torch
from gym.envs.mujoco import mujoco_env
from gym import utils
from gym_mujoco.envs.mujoco_env import MujocoEnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class Walker2dEnv(MujocoEnv, utils.EzPickle):
    def __init__(
            self,
            xml_file="walker2d.xml",
            forward_reward_weight=1.0,
            ctrl_cost_weight=1e-3,
            healthy_reward=1.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(0.8, 2.0),
            healthy_angle_range=(-1.0, 1.0),
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
            model_path=None
    ):
        self.encoder_model = encoder_model
        self.native_dim = native_dim
        self.latent_dim = latent_dim
        self.ae_model = None

        ae_lower_bounds = {
            1: [-1.0], 
            2: [-0.8166615134673153, -1.0], 
            3: [-0.831699790598429, -0.7992412637605554, -0.9975144285752414], 
            4: [-0.7926965772730471, -0.7465793762083448, -0.7813560252991852, -0.53092952628896], 
            5: [-0.7020813031008541, -0.8934733909044157, -0.3511555234800395, -0.5208407537893108, -0.6797616748922117], 
            6: [-0.3106281155408345, -0.4312993655832613, -0.5581967847863605, -0.46106950953380077, -0.438952551515732, -0.4084424627541305], 
        }

        ae_upper_bounds = {
            1: [0.8804036947876969], 
            2: [1.0, 0.8961136761107718], 
            3: [0.7278276770251397, 0.8633359704243104, 1.0], 
            4: [0.7267539452946687, 0.41986120558042084, 0.8311876075797271, 0.45792135812560786], 
            5: [0.6197293229684803, 0.44913645220925313, 0.39098875326526555, 0.5265536981132424, 0.4038073479931419], 
            6: [0.4205862222587844, 0.35669390904404036, 0.4010195051621915, 0.47900372672836766, 0.5081418383684979, 0.594142226036667], 
        }

        vae_lower_bounds = {
            1: [-0.69246943761765], 
            2: [-0.8308899170949078, -0.30019624059702454], 
            3: [-0.5542743737956722, -0.45196350246942646, -0.42569339312952525], 
            4: [-0.44816160253665926, -0.47306384423774633, -0.47629007739312573, -0.498808353832605], 
            5: [0.22269314568059062, -0.34934194165691385, -0.5848071370931756, -0.42229623221909174, -0.37691337270472614], 
            6: [-0.5063856067612486, -0.27798324777481437, -0.8640618222674488, -0.5138181853056142, -0.4131149039835662, -0.21916502012939942], 
        }
        vae_upper_bounds = {
            1: [1.0], 
            2: [1.0, 1.0], 
            3: [1.0, 0.5934053461191918, 1.0], 
            4: [0.5481796422841346, 0.46050395726114407, 0.9762913645035801, 1.0], 
            5: [1.0, 0.8086714784703106, 1.0, 0.5161584868543396, 1.0], 
            6: [1.0, 0.7946708741055732, 1.0, 0.475106600580006, 0.15541787782775726, 1.0], 
        }
        
        if encoder_model == "OTNAE":
            lower_bounds = [-1.0, -0.9310406493075205, -0.4403371173775308, 0.018616068977518357, -0.34970476521997296, -0.8298468489686179] 
            upper_bounds = [1.0, 0.7767702730041762, 0.25119124263850756, 0.3479325637522472, 0.16845778777195272, 0.8461558130147606] 
        elif encoder_model == "AE":
            lower_bounds = ae_lower_bounds[latent_dim]
            upper_bounds = ae_upper_bounds[latent_dim]
        elif encoder_model == "VAE":
            lower_bounds = vae_lower_bounds[latent_dim]
            upper_bounds = vae_upper_bounds[latent_dim]
        elif encoder_model == "OTNVAE":
            lower_bounds = [-1.0, -0.43607176472402587, -0.4378262302779989, -0.27309505904377795, -0.2581679585661167, -1.0]  
            upper_bounds = [0.8842934579801075, 0.8599148083755196, 0.42762637852421453, 0.48316382784450057, 0.7327615424321042, 1.0] 

        if self.encoder_model != None:
            self.init_ae_model(native_dim, latent_dim, hidden_layer,model_path)
            # self.action_space = gym.spaces.box.Box(np.array(lower_bounds[:latent_dim]), np.array(upper_bounds[:latent_dim]), shape=(latent_dim,))

        slope_bounds = None
        gravity_bounds = None
        mass_bounds = None
        friction_bounds = None
        if randomize_slope:
            slope_bounds = [-0.05, +0.05]
        if randomize_gravity:
            gravity_bounds = [0.5, 2]
        if randomize_mass:
            mass_bounds = [0.5, 2]
        if randomize_friction:
            friction_bounds = [0.5, 2]

        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        MujocoEnv.__init__(self, xml_file, 4,
                           slope_bounds=slope_bounds,
                           gravity_bounds=gravity_bounds,
                           mass_bounds=mass_bounds,
                           friction_bounds=friction_bounds, )

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
        z = z + self.sim.data.qpos[0]*np.tan(self.slope) # yslope > 0 --> downward slope


        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = healthy_z and healthy_angle

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
        x_velocity = (x_position_after - x_position_before) / self.dt / np.cos(self.slope)

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
