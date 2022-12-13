import gym
import numpy as np
from gym import utils

from gym_mujoco.envs.mujoco_env import MujocoEnv


class SwimmerEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, 
                native_dim=-1,
                encoder_model=None,
                latent_dim=-1,
                hidden_layer=-1,
                model_path=None):
        self.encoder_model = encoder_model
        self.native_dim = native_dim
        self.latent_dim = latent_dim
        self.ae_model = None

        ae_lower_bounds = {
            1: [-1.0], 
            2: [-0.412925216577046, -0.3662324929024072], 
        }

        ae_upper_bounds = {
            1: [1.0], 
            2: [0.37415049382434756, 0.506397379543878], 
        }

        vae_lower_bounds = {
            1: [-0.6561781088527022], 
            2: [-0.40701780447024677, -0.5535011660353963], 
        }
        vae_upper_bounds = {
            1: [1.0], 
            2: [0.2863519134266721, 1.0],  
        }
        
        if encoder_model == "OTNAE":
            lower_bounds = [-1.0, -1.0] 
            upper_bounds = [1.0, 1.0] 
        elif encoder_model == "AE":
            lower_bounds = ae_lower_bounds[latent_dim]
            upper_bounds = ae_upper_bounds[latent_dim]
        elif encoder_model == "VAE":
            lower_bounds = vae_lower_bounds[latent_dim]
            upper_bounds = vae_upper_bounds[latent_dim]
        elif encoder_model == "OTNVAE":
            lower_bounds = [-1.0, -0.5156714104350745] 
            upper_bounds = [1.0, 0.759048547575201] 

        if self.encoder_model != None:
            self.init_ae_model(native_dim, latent_dim, hidden_layer,model_path)
            # self.action_space = gym.spaces.box.Box(np.array(lower_bounds[:latent_dim]), np.array(upper_bounds[:latent_dim]), shape=(latent_dim,))

        
        MujocoEnv.__init__(self, "swimmer.xml", 4)

        utils.EzPickle.__init__(self)

    def step(self, a):
        # a = self.reconstruct(a)
        a = self.bottleneck(a, self.latent_dim)
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = -ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv),
        )
        return self._get_obs()