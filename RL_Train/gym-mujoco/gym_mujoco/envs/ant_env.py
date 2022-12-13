import numpy as np
from gym import utils
import gym.spaces.box
from gym_mujoco.envs.mujoco_env import MujocoEnv
import torch

class AntEnv(MujocoEnv, utils.EzPickle):
    def __init__(
            self,
            slope=None,
            gravity=None,
            mass=None,
            friction=None,
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
            2: [-0.3441139020555455, -0.4607266667465467],
            3: [-0.6003824512252076, -0.08413776876290441, -0.5302562284144169],
            4: [-0.45770354425256765, -0.6420795601788057, -0.4121610273762347, -0.09040944689079716],
            5: [-0.21236915542158413, -0.39366383443230535, -0.6893768885312384, -0.709060962750114, -0.5377486119060733],
            6: [-0.1362287561744565, -0.20326953351983748, -0.6159330634405104, -0.49687731313147854, -0.379869711895674, -0.12669776216860631],
            7: [-0.23400133313018695, -0.25407412166934074, -0.509806028699185, -0.1536737177419646, -0.4339598233735781, -0.16440835035421958, -0.4178426180928765],
            8: [-0.20738737986531755, -0.2979311515295984, -0.26452054095509175, -0.19068204153331178, -0.1617573536594778, -0.3177981839934494, -0.10473315901791483, -0.3466360777673768]
        }

        ae_upper_bounds = {
            1: [0.7231701975890638], 
            2: [0.607843528805627, 0.6385682759774481], 
            3: [0.386736040710265, 0.8307408601133097, 0.38639389912202715], 
            4: [0.28082778468394026, 0.360859482993441, 0.294301297690199, 0.7067624120094755], 
            5: [0.5200677309105943, 0.5163677799189648, 0.29266790798052, 0.024581281906009722, 0.23864441193748784], 
            6: [0.6424271217458912, 0.41781687339293094, 0.16067927752078975, 0.23820267352045585, 0.5144247829012979, 0.5332562292876283], 
            7: [0.3921037402415378, 0.3898045914947053, 0.2439671793971029, 0.42771248385978733, 0.24977509242858353, 0.38966193516564684, 0.3471904324041844], 
            8: [0.24459128255234608, 0.2623461344155514, 0.22365936484026716, 0.21728791790986787, 0.29008966306590556, 0.010612909003657411, 0.38569987816206436, -0.01171766495295487], 
        }

        vae_upper_bounds = {
            1: [0.7676020702966748], 
            2: [1.0, 0.8188617786798897], 
            3: [0.43087027000106815, 0.12350040782173656, 0.3660734995367459], 
            4: [0.4321033429623835, 0.20450929210218238, 1.0, 0.9988473403752084], 
            5: [0.1919082864241452, 0.31422792558571955, 1.0, 0.874085959956759, 0.9567588033161398], 
            6: [0.13474625484047262, 0.36313916212215014, 0.26983835504010306, 0.9043961579654716, 1.0, 0.5233175790296505], 
            7: [0.7519591626226327, 0.5466830819197634, 0.4305431702986771, 0.02906384270859219, 0.9386436732248729, 0.19312107760207364, 0.7731597765832543], 
            8: [0.6821806879155495, 0.8411360689488988, 0.09966254332692373, 1.0, 0.07174201049421187, 1.0, 1.0, 0.7933484047852837],
        }
        vae_lower_bounds = {
            1: [-0.014802729164542838], 
            2: [-0.221815727652238, -0.39766183308122083], 
            3: [-0.3032912254431216, -0.33275297999525266, -0.2651874603057956], 
            4: [-0.503344934794962, -0.23640958017471211, 0.3997946075941114, 0.05870667769676813], 
            5: [-0.372102027187855, -0.510335443647422, 0.25595236966670215, -0.15129264901569073, 0.19560452154763847], 
            6: [-0.19352962966884923, -0.1687830086161532, -0.3254690964410593, 0.3383801015579201, -0.25978451531725455, 0.027339571671267715], 
            7: [0.0784248660048491, -0.04796380811231582, 0.049802587368186824, -0.3293411366301755, 0.15257384758647768, -0.3530174216305214, -0.19909415125139235], 
            8: [0.03712289758115017, 0.18947922671256773, -0.20177460196081579, 0.18647329056998097, -0.2248823084076484, 0.32781948524866444, 0.40711175955606427, 0.02704737592433637], 
        }
        
        if encoder_model == "OTNAE":
            lower_bounds = [-0.333696117352243, -0.6356164588633815, -0.7602210673643633, -0.2623440861969074, -0.13129290712977584, -0.1150177820494275, -0.42479584796923453, -0.19872508083379967]
            upper_bounds = [0.8064970468687969, 0.3521026380494128, -0.2858587228418902, 0.22592066701080704, 0.24661963100041087, 0.33463563373252914, -0.14145650139403287, 0.1835418637099038]
        elif encoder_model == "AE":
            lower_bounds = ae_lower_bounds[latent_dim]
            upper_bounds = ae_upper_bounds[latent_dim]
        elif encoder_model == "VAE":
            lower_bounds = vae_lower_bounds[latent_dim]
            upper_bounds = vae_upper_bounds[latent_dim]
        elif encoder_model == "OTNVAE":
            lower_bounds = [-0.48091410799660606, 0.25650983998576793, -0.6517028585850968, -0.14445354715309966, -0.22297263951633897, -0.4244389913768983, -0.18027939782204216, -0.7407406558519789]
            upper_bounds = [0.9409952355200373, 0.7315455598464935, -0.07107305028292982, 0.43479749594031825, 0.5566640160704374, 0.34123620545855643, 0.30982147765291135, 0.13165474599537857]

        if self.encoder_model != None:
            self.init_ae_model(native_dim, latent_dim, hidden_layer,model_path)

            # self.action_space = gym.spaces.box.Box(np.array(lower_bounds[:latent_dim]), np.array(upper_bounds[:latent_dim]), shape=(latent_dim,))


        MujocoEnv.__init__(self, "ant.xml", 5,
                           slope_bounds=slope,
                           gravity_bounds=gravity,
                           mass_bounds=mass,
                           friction_bounds=friction,
                           )

        utils.EzPickle.__init__(self)

    def step(self, a):
        # a = self.reconstruct(a)
        a = self.bottleneck(a, self.latent_dim)
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt / np.cos(self.slope)
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()

        height_above_ground = state[2] + state[0]*np.tan(self.slope) # yslope > 0 --> downward slope

        notdone = np.isfinite(state).all() and height_above_ground >= 0.2 and height_above_ground <= 1.0
        done = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
    
    