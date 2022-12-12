# CS839-FinalProject

## Python environment configuration

## Supported variables

### Environments (\[env_id\] (\[native_dim\])):

- Hopper-v3 (3)
- HalfCheetah-v3 (6)
- Walker2d-v3 (6)
- Ant-v3 (8)
- Swimmer20-v3 (19)
- ReacherTracker20-v3 (20)

### Encoder models (\[encoder_model\]):

- Vanilla Auto-encoder (AE)
- One-to-N Auto-encoder (OTNAE)

## Learning process

### Train native agent

```
cd RL_Train
python3 train.py --algo td3 --gym-packages gym-mujoco --env [env_id] --seed [random-seed] --n-timesteps 2000000
```

agents will be saved in the directory `RL_Train/logs/`

### Generate demonstrations

#### Expert

```
cd RL_Train/utils
python3 generate_demon.py --env-id [env_id] --model_path [model_path] --num-actions 1000000
```

#### Random

```
cd RL_Train/utils
python3 generate_demon.py --env-id [env_id] --model_path [model_path] --num-actions 1000000 --random
```

demonstrations will be save in the directory `RL_train/utils/data/`

### Train Auto-encoder

```
cd AE_Train
python3 train.py --env_id [env_id] --native_dim [native_dim] --latent_dim [native_dim] --encoder_model [encoder_model] --hidden_layer 64 --batch_size 64 --data_path [data_path] --save_dir [save_dir] --epoch 100 --lr 1e-4 --num_actions 100000
```

models with a visualization of the running loss will be saved in the directory `RL_train/[save_dir]`

### Train RL agents with learned latent action space in source environment

```
cd RL_Train
python3 train.py --algo td3 --gym-packages gym-mujoco --env [env_id] --env-kwargs latent_dim:[latent_dim] encoder_model:[encoder_model] native_dim:[native_dim] hidden_layer:64 model_path:[model_path] --seed [seed]
```

### Train RL agents with learned latent action space in target environment

```
cd RL_Train
```

- Ant-v3: `python3 train.py --algo td3 --gym-packages gym-mujoco --env Ant-v3 --env-kwargs latent_dim:[latent_dim] encoder_model:[encoder_model] native_dim:[native_dim] hidden_layer:64 model_path:[model_path] mass:0.5,2.0 friction:0.5,1.5 slope:-0.1,0.1 gravity:0.5,2.0 --seed [seed]`
- ReacherTracker20-v3: `python3 train.py --algo td3 --gym-packages gym-mujoco --env ReacherTracker20-v3 --env-kwargs latent_dim:[latent_dim] encoder_model:[encoder_model] native_dim:[native_dim] hidden_layer:64 model_path:[model_path] mass:0.5,2.0 damping:1,10 --seed [seed]`
- Swimmer20-v3: `python3 train.py --algo td3 --gym-packages gym-mujoco --env Swimmer20-v3 --env-kwargs latent_dim:[latent_dim] encoder_model:[encoder_model] native_dim:[native_dim] hidden_layer:64 model_path:[model_path] density:2000,6000 damping:1,5 --seed [seed]`
- Other environments (Hopper-v3, HalfCheetah-v3, Walker2d-v3): `python3 train.py --algo td3 --gym-packages gym-mujoco --env [env_id] --env-kwargs latent_dim:[latent_dim] encoder_model:[encoder_model] native_dim:[native_dim] hidden_layer:64 model_path:[model_path] randomize_slope:True randomize_gravity:True randomize_mass:True randomize_friction:True --seed [seed]`
