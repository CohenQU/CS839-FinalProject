# CS839-FinalProject

## Python environment configuration

```
conda create -n <env_name> python=3.9
conda activate <env_name>
cd RL_Train

pip install -r requirement.txt
cd gym_mujoco
pip install -e .

```

## Supported variables

- Environments (\[env_id\] (\[native_dim\])): Hopper-v3 (3), HalfCheetah-v3 (6), Walker2d-v3 (6), Ant-v3 (8), Swimmer20-v3 (19), ReacherTracker20-v3 (20)
- Encoder models (\[encoder_model\]): Vanilla Auto-encoder (AE), One-to-N Auto-encoder (OTNAE)
- Agents (\[agent\]): random, trained
- Data in .npz (\[data_name\]): states, actions, rewards, next_states, next_actions, dones

## Learning process

### Train native agent

```
cd RL_Train
python3 train.py --algo td3 --gym-packages gym-mujoco --env [env_id] --seed [random-seed] --n-timesteps 2000000
```

agents will be saved at `RL_Train/logs/td3/[env_id]_[seed]/[env_id].zip`

### Generate demonstrations

#### Expert

```
cd RL_Train/utils
python3 generate_demo.py --env-id [env_id] --model_path [model_path] --num-actions 1000000
```

data will be saved at `RL_Train/utils/data/[env_id]/trained.npz`

#### Random

```
cd RL_Train/utils
python3 generate_demon.py --env-id [env_id] --model_path [model_path] --num-actions 1000000 --random
```

data will be saved at `RL_Train/utils/data/[env_id]/random.npz`

### Decompose .npz

```
cd RL_Train/utils
python3 split.py --env-id [env_id] --agent [agent] --data_name [data_name]
```

data will be saved at `RL_Train/utils/data/[env_id]/[agent]_[data_name].npy`

### Train Auto-encoder

```
cd AE_Train
python3 train.py --env_id [env_id] --native_dim [native_dim] --latent_dim [latent_dim] --encoder_model [encoder_model] --hidden_layer 64 --batch_size 64 --data_path [data_path] --save_dir [save_dir] --epoch 100 --lr 1e-4 --num_actions 100000
```

models with a visualization of the running loss will be saved in the directory `AE_train/[save_dir]`

### Train RL agents with learned latent action space in source environment

```
cd RL_Train
python3 train.py --algo td3 --gym-packages gym-mujoco --env [env_id] --env-kwargs latent_dim:[latent_dim] encoder_model:\"[encoder_model]\" native_dim:[native_dim] hidden_layer:64 model_path:\"[model_path]\" --seed [seed] --n-timesteps 1000000
```

### Train RL agents with learned latent action space in target environment

```
cd RL_Train
```

- Ant-v3: `python3 train.py --algo td3 --gym-packages gym-mujoco --env Ant-v3 --env-kwargs latent_dim:[latent_dim] encoder_model:[encoder_model] native_dim:[native_dim] hidden_layer:64 model_path:[model_path] mass:0.5,2.0 friction:0.5,1.5 slope:-0.1,0.1 gravity:0.5,2.0 --seed [seed] --n-timesteps 1000000`
- ReacherTracker20-v3: `python3 train.py --algo td3 --gym-packages gym-mujoco --env ReacherTracker20-v3 --env-kwargs latent_dim:[latent_dim] encoder_model:[encoder_model] native_dim:[native_dim] hidden_layer:64 model_path:[model_path] mass:0.5,2.0 damping:1,10 --seed [seed] --n-timesteps 1000000`
- Swimmer20-v3: `python3 train.py --algo td3 --gym-packages gym-mujoco --env Swimmer20-v3 --env-kwargs latent_dim:[latent_dim] encoder_model:[encoder_model] native_dim:[native_dim] hidden_layer:64 model_path:[model_path] density:2000,6000 damping:1,5 --seed [seed] --n-timesteps 1000000`
- Other environments (Hopper-v3, HalfCheetah-v3, Walker2d-v3): `python3 train.py --algo td3 --gym-packages gym-mujoco --env [env_id] --env-kwargs latent_dim:[latent_dim] encoder_model:[encoder_model] native_dim:[native_dim] hidden_layer:64 model_path:[model_path] randomize_slope:True randomize_gravity:True randomize_mass:True randomize_friction:True --seed [seed] --n-timesteps 1000000`

python3 train.py --algo td3 --gym-packages gym-mujoco --env Ant-v3 --env-kwargs latent_dim:4 encoder_model:\"AE\" native_dim:8 hidden_layer:64 model_path:\"/Users/yxqu/Desktop/Class/COMPSCI839/FinalProject/code/CS839-FinalProject/AE_Train/expert/models/Ant-v3_AE/AE_4.pt\" --seed 1 --n-timesteps 10000
