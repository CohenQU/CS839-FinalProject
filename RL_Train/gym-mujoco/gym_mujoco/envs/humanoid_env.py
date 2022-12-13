import numpy as np
from gym import utils
from gym_mujoco.envs.mujoco_env import MujocoEnv


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidEnv(MujocoEnv, utils.EzPickle):
    def __init__(self,
                 randomize_slope=False,
                 randomize_gravity=False,
                 randomize_mass=False,
                 randomize_friction=False,
                 ):

        slope_bounds = None
        gravity_bounds = None
        mass_bounds = None
        friction_bounds = None
        if randomize_slope:
            slope_bounds = [-0.05, +0.05]
        if randomize_gravity:
            gravity_bounds = [0.75, 1.25]
        if randomize_mass:
            mass_bounds = [0.75, 1.25]
        if randomize_friction:
            friction_bounds = [0.75, 1.25]

        MujocoEnv.__init__(self, "humanoid.xml", 5,
                           slope_bounds=slope_bounds,
                           gravity_bounds=gravity_bounds,
                           mass_bounds=mass_bounds,
                           friction_bounds=friction_bounds,
                           )
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate(
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt / np.cos(self.slope)
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos

        # qpos[0:2] = x,y,z positions
        # qpos[3:6] = quaternions postions
        # tan(slope) = z/y --> z = y*tan(slope)
        # Assumes slope is only changed in the y plane!
        height_above_ground = qpos[2] + qpos[0]*np.tan(self.slope) # yslope > 0 --> downward slope
        # print('height', height_above_ground)
        done = bool((height_above_ground < 1.0) or (height_above_ground > height_above_ground))
        return (
            self._get_obs(),
            reward,
            done,
            dict(
                reward_linvel=lin_vel_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_alive=alive_bonus,
                reward_impact=-quad_impact_cost,
            ),
        )

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(
                low=-c,
                high=c,
                size=self.model.nv,
            ),
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20