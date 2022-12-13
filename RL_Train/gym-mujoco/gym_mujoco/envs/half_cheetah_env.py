import gym
import numpy as np
from gym import utils
from gym_mujoco.envs.mujoco_env import MujocoEnv

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class HalfCheetahEnv(MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="half_cheetah.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
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

        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # self.action_space = gym.spaces.Box(-1, +1, shape=(6,))
        ae_lower_bounds = {
            1: [-1.0], 
            2: [-0.6878362873561662, -1.0], 
            3: [-0.6299724431959844, -0.7940601019586835, -0.738661409263256], 
            4: [-0.801098952467469, -0.6623090242048923, -0.8853558736871038, -0.3256557753349629], 
            5: [-0.3422210059193498, -0.47092918469482337, -0.509614377283623, -0.6527407343794264, -0.7292326067022499], 
            6: [-0.37225290013282103, -0.5993836652383006, -0.5283296189590955, -0.44339058668503006, -0.6620326679944072, -0.4211815972725573]
        }

        ae_upper_bounds = {
            1: [1.0], 
            2: [0.6757999750120496, 1.0], 
            3: [0.37349743886196274, 0.8664190326845, 0.9076243488098241], 
            4: [0.8614624585903168, 0.40381415243677876, 0.8399750081001028, 0.5701373969249], 
            5: [0.525667563138514, 0.9174549882149411, 0.5506703154599188, 0.6329439676323149, 0.6771950970128899], 
            6: [0.3625875292123069, 0.6756400722021794, 0.5499546693716874, 0.6135751974279501, 0.7662875328590285, 0.46428772084346154], 
        }

        vae_lower_bounds = {
            1: [-0.7879212109451487], 
            2: [-0.34872155715592285, -1.0], 
            3: [-0.18313958371478367, -0.7777137612047946, -0.444082771216424], 
            4: [-0.5316242925378526, -0.5909238301199264, -0.39524430383787007, -1.0], 
            5: [-0.5959499016682022, -0.36369769108388217, -0.4321557328940934, -0.48313852298348464, -0.6197605266756966], 
            6: [-0.03498700555654388, -0.515506230480326, -0.5739513661207029, -0.5470875967403768, -0.2904893693003335, -0.46446946314171994], 
        }

        vae_upper_bounds = {
            1: [1.0], 
            2: [0.9147157466991591, 1.0], 
            3: [1.0, 1.0, 0.2401838768528109], 
            4: [0.5263262572643302, 1.0, 0.5025449003930869, 1.0], 
            5: [1.0, 1.0, 1.0, 1.0, 1.0], 
            6: [1.0, 0.8494417299058943, 1.0, 1.0, 1.0, 0.3310894230685243], 
        }
        
        
        if encoder_model == "OTNAE":
            lower_bounds = [-1.0, -0.5259224085015698, -0.12280347937734581, -0.16911740909103723, -0.19963298810619026, -0.07600529523648802]
            upper_bounds = [1.0, 0.5451327220103123, 0.2510774558534449, 0.3347243073807257, 0.39357017008090245, 0.1413701680062115]
        elif encoder_model == "AE":
            lower_bounds = ae_lower_bounds[latent_dim]
            upper_bounds = ae_upper_bounds[latent_dim]
        elif encoder_model == "VAE":
            lower_bounds = vae_lower_bounds[latent_dim]
            upper_bounds = vae_upper_bounds[latent_dim]
        elif encoder_model == "OTNVAE":
            lower_bounds = [-1.0, -0.47697562065852417, -0.2355697120839988, -0.5766057955035719, -1.0, -1.0]
            upper_bounds = [1.0, 0.5219258152124051, 0.3294370186837132, 0.3652631218618496, 0.5456795124698071, 0.776484927244113]

        if self.encoder_model != None:
            self.init_ae_model(native_dim, latent_dim, hidden_layer,model_path)
            # self.action_space = gym.spaces.box.Box(np.array(lower_bounds[:latent_dim]), np.array(upper_bounds[:latent_dim]), shape=(latent_dim,))

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

        MujocoEnv.__init__(self, xml_file, 5,
                           slope_bounds=slope_bounds,
                           gravity_bounds=gravity_bounds,
                           mass_bounds=mass_bounds,
                           friction_bounds=friction_bounds,
                           )


    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        # action = self.reconstruct(action)
        action = self.bottleneck(action, self.latent_dim)
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt / np.cos(self.slope)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
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