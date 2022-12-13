import numpy as np
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)

    args = parser.parse_args()
    env_id = args.env_id
    agent = args.agent
    data_name = args.data_name

    path = "./data/{}/{}.npz".format(env_id, agent)
    raw_data = np.load(path)
    data = raw_data[data_name]
    new_path = "./data/{}/{}_{}".format(env_id,agent, data_name)
    np.save(new_path, data)
