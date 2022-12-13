import copy
import os

import gym
import mujoco_py
import numpy as np
from gym import spaces
from gym.envs.mujoco.mujoco_env import convert_observation_to_space, DEFAULT_SIZE
from gym.utils import seeding
import torch
from ae_model.regular import *
from ae_model.otnae import *
from ae_model.vae import *
from ae_model.otnvae import *

class MujocoEnv(gym.Env):
    """
    Superclass for all MuJoCo environments.

    User can pass an optional 'local_xml' argument to the constructor. If local_xml=True,
    we search for the xml file from the local xml directory.
    """

    def __init__(self, model_path, frame_skip,
                 slope_bounds=None,
                 gravity_bounds=None,
                 mass_bounds=None,
                 friction_bounds=None,
                 density_bounds=None,
                 damping_bounds=None):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()
        self.slope = 0

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()

        # Env parameters
        self.slope_bounds = slope_bounds
        self.gravity_bounds = gravity_bounds
        self.mass_bounds = mass_bounds
        self.friction_bounds = friction_bounds
        self.density_bounds = density_bounds
        self.damping_bounds = damping_bounds

        print('slope:', self.slope_bounds)
        print('gravity:', self.gravity_bounds)
        print('mass', self.mass_bounds)
        print('friction', self.friction_bounds)
        print('density', self.density_bounds)
        print('damping', self.damping_bounds)

        # Not every model has a torso
        try:
            self.torso_index = list(self.model.body_names).index('torso')
        except:
            self.torso_index = None
        self.original_slope = 0
        self.original_density = 0
        self.original_damping = copy.deepcopy(self.model.dof_damping)
        self.original_gravity = copy.deepcopy(self.model.opt.gravity[-1])
        self.original_masses = copy.deepcopy(self.model.body_mass)
        self.original_friction = copy.deepcopy(self.model.geom_friction)


    def _set_action_space(self):
        if self.action_space is None:
            bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
            low, high = bounds.T
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        self.randomize_env_parameters()
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(
        self,
        mode="human",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):
        if mode == "rgb_array" or mode == "depth_array":
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == "rgb_array":
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "depth_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])


    def randomize_env_parameters(self):

        if self.slope_bounds is not None:
            slope = np.random.uniform(*self.slope_bounds)
            self.slope = slope # save slope because some envs end an episode if the torso falls below a certain height.
            self.model.geom_quat[0] = [np.cos(slope/2), 0 , np.sin(slope/2), 0]

        if self.gravity_bounds is not None:
            gravity_coef = np.random.uniform(*self.gravity_bounds)
            self.model.opt.gravity[-1] = gravity_coef * self.original_gravity

        if self.mass_bounds is not None:
            mass_coef = np.random.uniform(*self.mass_bounds)
            if self.torso_index is None:
                self.model.body_mass[:] = mass_coef * self.original_masses[:]
            else:
                self.model.body_mass[self.torso_index] = mass_coef * self.original_masses[self.torso_index]
            # print(self.model.body_mass[self.torso_index])

        if self.friction_bounds is not None:
            friction_coef = np.random.uniform(*self.friction_bounds)
            self.model.geom_friction[:,:] = friction_coef * self.original_friction

        if self.damping_bounds is not None:
            damping_coef = np.random.uniform(*self.damping_bounds)
            self.model.dof_damping[:-2] = damping_coef

        if self.density_bounds is not None:
            density_coef = np.random.uniform(*self.density_bounds)
            self.model.opt.density = density_coef

        # self.model.opt.wind[:] = np.random.uniform(-100, 100, size=(3,))

            # 5: back foot
            # 8: front foot

        # print(slope)
        # print(new_gravity)
        # print(new_masses)
        # print(new_friction)
        # print(density_coef)
        # print(self.model.dof_damping)

    def set_param(self, name, val):

        if name == 'damping':
            self.model.dof_damping[:-2] = val
        elif name == 'mass':
            self.model.body_mass[self.torso_index] = val * self.original_masses[self.torso_index]
        elif name == 'mass_torso':
            self.model.body_mass[:] = val * self.original_masses[:]
        elif name == 'friction':
            self.model.geom_friction[:,:] = val * self.original_friction
        elif name == 'slope':
            self.slope = val
            self.model.geom_quat[0] = [np.cos(val/2), 0 , np.sin(val/2), 0]
        elif name == 'gravity':
            self.model.opt.gravity[-1] = val * self.original_gravity

    def init_ae_model(self, native_dim, latent_dim, hidden_layer, model_path):
        if self.encoder_model != None:
            assert native_dim != -1 and latent_dim != -1 and hidden_layer != -1
            if self.encoder_model == "AE":
                self.ae_model = VanillaAE(native_dim, latent_dim, hidden_layer)
            elif self.encoder_model == "VAE":     
                self.ae_model = VAE(native_dim, latent_dim, hidden_layer) 
            elif self.encoder_model == "OTNAE":
                self.ae_model = OTNAE(native_dim, hidden_layer)
            elif self.encoder_model == "OTNVAE":
                self.ae_model = OTNVAE(native_dim, hidden_layer)
            
            self.ae_model.load_state_dict(torch.load(model_path), strict=False)
            self.ae_model.eval()
            # self.action_space = gym.spaces.box.Box(-1, +1, shape=(latent_dim,))
            # self.action_space = gym.spaces.box.Box(np.array(lower_bounds[:latent_dim]), np.array(upper_bounds[:latent_dim]), shape=(latent_dim,))

    def reconstruct(self, a):
        if self.encoder_model != None:
            a = torch.from_numpy(a)
            if self.encoder_model == "AE" or self.encoder_model == "VAE":
                a = self.ae_model.decoder(a.float())
            elif self.encoder_model == "OTNAE" or self.encoder_model == "OTNVAE":
                a = a.float()
                output = torch.zeros(1, self.native_dim)
                for i in range(self.latent_dim):
                    decoder = self.ae_model.decoders[i]
                    output += decoder(a[i].reshape(-1, 1))
                a = self.ae_model.tanh(output) 
            a = a.detach().numpy()
        return a
    def bottleneck(self, a, latent_dim):
        if self.encoder_model != None:
            a = torch.from_numpy(a).reshape(1, -1)
            if self.encoder_model == "AE" or self.encoder_model == "VAE":
                a = self.ae_model(a.float())
            elif self.encoder_model == "OTNAE" or self.encoder_model == "OTNVAE":
                output = self.ae_model(a.float())
                a = output[latent_dim-1, :, :]
                # a = a.float()
                # output = torch.zeros(1, self.latent_dim)
                # for i in range(self.native_dim):
                #     encoder = self.ae_model.encoders[i]
                #     output += encoder(a[i].reshape(-1, 1))
                # a = output
            a = a.detach().numpy()
        return a
    

