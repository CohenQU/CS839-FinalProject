import argparse
import os
import sys

import gym, gym_mujoco
import numpy as np
from utils import StoreDict, ALGOS

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

def simulate(env, model, num_actions, random=False, seed=0):
    states = []
    actions = []
    rewards = []
    next_states = []
    next_actions =[]
    ep_returns = []
    dones = []

    action_dim = env.action_space.shape[-1]

    env.seed(seed)
    np.random.seed(seed)

    action_count = 0
    while action_count < num_actions:

        # s,a,r,s',a',done
        ep_actions = []

        state = env.reset()
        done = False

        step_count = 0
        ep_return = 0
        while not done:
            if random:
                action = np.random.uniform(-1, 1, size=(action_dim,))
                # action = np.array([1*(1-step_count/50) + 0.05])
            else:
                action, _ = model.predict(state, deterministic=True)

            states.append(state)
            actions.append(action)
            ep_actions.append(action)

            state, reward, done, _ = env.step(action)
            ep_return += reward

            rewards.append(reward)
            next_states.append(state)
            dones.append(False)

            step_count += 1

        action_count += step_count
        dones[-1] = True
        next_actions.extend(ep_actions[1:])
        next_actions.append(ep_actions[-1])
        ep_returns.append(ep_return)
        print(f'action_count = {action_count}, return = {ep_return}', step_count)
    print("Average return = {}".format(np.average(np.array(ep_returns))))
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    next_actions = np.array(next_actions)
    dones = np.array(dones)

    return states, actions, rewards, next_states, next_actions, dones

if __name__ == "__main__":

    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default=f"Walker2d-v3")
    parser.add_argument("--env-kwargs", type=str, nargs="*", action=StoreDict, default={})
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument("--time-feature", type=bool, default=False)
    parser.add_argument("--algo", type=str, default="td3")
    parser.add_argument("--num-actions", type=int, default=int(10e3))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-threshold", type=float, default=-np.inf)
    parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--model_path')
    args = parser.parse_args()
    print(args)

    env_id = args.env_id
    env = gym.make(env_id, **args.env_kwargs)
    env.seed(args.seed)

    # NECESSARY
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    algo_class = ALGOS[args.algo]
    model_path = args.model_path

    model = algo_class.load(model_path, env=env, custom_objects=custom_objects)
    states, actions, rewards, next_states, next_actions, dones = simulate(
        env=env, model=model, num_actions=args.num_actions, random=args.random)

    save_dir = f'./data/{env_id}'
    if args.random:
        save_path = f'{save_dir}/random.npz'
    else:
        save_path = f'{save_dir}/trained.npz'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print(f'Saving to {save_dir}')
    
    np.savez(save_path, states=states, actions=actions, rewards=rewards, dones=dones)