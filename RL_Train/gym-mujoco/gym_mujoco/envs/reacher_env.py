import gym
import numpy as np
from gym import utils
from gym_mujoco.envs.mujoco_env import MujocoEnv
from stable_baselines3.common.utils import load_pca_transformation_numpy


class ReacherEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, num_links=10, goal=None):
        self.num_links = num_links
        self.goal = np.array(goal)
        self.randomize_goal = goal is None

        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(self, model_path=f"reacher_{num_links}dof.xml", frame_skip=2)

    def step(self, a):

        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)*self.num_links
        reward_ctrl = -np.square(a).sum()
        # print(a)
        # print(reward_ctrl, reward_dist, sep='\t')
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False # I think the max_steps in teh registration handles the horizon.

        info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        # print(info)

        return ob, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.2, high=0.2, size=self.model.nq)
            + self.init_qpos
        )

        if self.randomize_goal:
            while True:
                r = self.num_links*0.1
                self.goal = np.random.uniform(-r, r, size=(2,))
                if np.linalg.norm(self.goal) < 1: break
            qpos[-2:] = self.goal
        else:
            qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[self.num_links:],
                self.sim.data.qvel.flat[:self.num_links],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ]
        )