import os

import gym
import numpy as np

class Toy2dEnv(gym.Env):

    def __init__(self, use_latent=False):
        super().__init__()

        self.step_counter = 0

        self.observation_space = gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(2,))
        self.action_space = gym.spaces.Box(low=-1, high=+1, shape=(2,))

        self.state_curr = np.array([0.0,0.0])
        self.goal = np.array([1,0.0])
        self.step_size = 0.1

        print(os.getcwd())

        self.use_latent = use_latent
        if self.use_latent:
            self.action_space = gym.spaces.Box(-1, 1, shape=(1,))
            self.W = np.load('./pca/pca_results/Toy2d-v0/W_squashed.npy')[:1,:]
            self.mu = np.load('./pca/pca_results/Toy2d-v0/mu_squashed.npy')
            print(self.W)
            print(self.mu)

        self.X = np.arange(0, 1.1, 0.1)

    def is_good(self):

        x, y = self.state_curr
        d = 0.1

        if (y > -d and y < d):
            for val in self.X:
                if np.allclose(x, val, rtol=1e-20, atol=1e-20):
                    print(x, val)
                    return True
        elif (y < -d or y > d):
            return True
        else:
            return False


    def step(self, a):

        # a = np.clip(a, -1+1e-7, 1-1e-7)
        # a = np.arctanh(a) # unsquash so latent space is a line'

        # W = np.array([[1, 0]])
        # mu =np.array([1,0])
        # a = W.T.dot(a) + mu
        if self.use_latent:
            a = self.W.T.dot(a) + self.mu
        # a = np.tanh(a)
        a = np.clip(a, -1, 1)

        # if self.step_counter > 10000:
        #     print(a)

        self.state_curr += self.step_size * a
        # self.state_curr = np.clip(self.state_curr, -1, 1)

        if self.is_good():
            reward = -np.linalg.norm(self.state_curr - self.goal)
            # print(self.state_curr)
        else:
            reward = -1

        done = False
        # print(self.state_curr, reward)

        self.step_counter += 1
        return self.state_curr, reward, done, {}

    def reset(self):
        self.state_curr = np.array([0.0,0.0])
        return self.state_curr


if __name__ == "__main__":

    X = np.arange(0.1, 1.1, 0.1)
    s = 0
    for x in X:
        s += np.sqrt((1-x)**2)
        print(1-x)
    print(X)
    print(s)