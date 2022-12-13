# CS839-FinalProject

## 1. Python environment configuration

### 1.1. For MacOS with M1 chip (thanks [geyang](https://github.com/openai/mujoco-py/issues/682), and [wookayin](https://github.com/openai/mujoco-py/issues/662))

#### 1.1.1. Background

Until `mujoco-py` gets updated to officially support DeepMind's MuJoCo 2.1+, you can try the following as a hacky workaround.
First, make sure your python is running as arm64 (NOT x86_64 under Rosetta 2). For instance, you can use `miniforge3`.

```
$ which python3
/Users/$ID/.miniforge3/bin/python3
$ lipo -archs $(which python3)
arm64
```

#### 1.1.2. Pre-requisits

- Use `Miniforge` as your Conda environment
  - If you want to keep both Miniforge and Anaconda/Miniconda, you can refer to this [tutorial](https://youtu.be/w2qlou7n7MA).
- Install `glfw` via `brew install glfw`. Note the location for the installation
- Download [MuJoCo2.1.1](https://github.com/deepmind/mujoco/releases/tag/2.1.1) image that ends with a \*.dmg. The new mujoco2.1.1 is released as a Framework. You can copy the MuJoCo.app into /Applications/ folder.

#### 1.1.3. Installation Script

Make a file locally called `install-mujoco.sh`, and put the following into it.

```
mkdir -p $HOME/.mujoco/mujoco210         # Remove existing installation if any
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/Headers/ $HOME/.mujoco/mujoco210/include
mkdir -p $HOME/.mujoco/mujoco210/bin
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.*.dylib $HOME/.mujoco/mujoco210/bin/libmujoco210.dylib
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.*.dylib /usr/local/lib/

# For M1 (arm64) mac users:
# The released binary doesn't ship glfw3, so need to install on your own
brew install glfw
ln -sf /opt/homebrew/lib/libglfw.3.dylib $HOME/.mujoco/mujoco210/bin

# Please make sure /opt/homebrew/bin/gcc-11  exists: install gcc if you haven't already
# brew install gcc
export CC=/opt/homebrew/bin/gcc-11         # see https://github.com/openai/mujoco-py/issues/605

pip install mujoco-py && python -c 'import mujoco_py'
```

### 1.2. For MacOS and Win

```
conda create --name <env> --file requirements.txt
conda activate <env_name>
cd RL_Train
pip install -r requirement.txt
cd gym_mujoco
pip install -e .
```

## 2. Learning process

### 2.1. Supported variables

- Environments (\[env_id\] (\[native_dim\])): Hopper-v3 (3), HalfCheetah-v3 (6), Walker2d-v3 (6), Ant-v3 (8), Swimmer20-v3 (19), ReacherTracker20-v3 (20)
- Encoder models (\[encoder_model\]): Vanilla Auto-encoder (AE), One-to-N Auto-encoder (OTNAE)
- Agents (\[agent\]): random, trained
- Data in .npz (\[data_name\]): states, actions, rewards, next_states, next_actions, dones

### 2.2. Train native agent

```
cd RL_Train
python3 train.py --algo td3 --gym-packages gym-mujoco --env [env_id] --seed [random-seed] --n-timesteps 2000000
```

agents will be saved at `RL_Train/logs/td3/[env_id]_[seed]/[env_id].zip`

### 2.3. Generate demonstrations

#### 2.3.1. Expert

```
cd RL_Train/utils
python3 generate_demo.py --env-id [env_id] --model_path [model_path] --num-actions 1000000
```

data will be saved at `RL_Train/utils/data/[env_id]/trained.npz`

#### 2.3.2. Random

```
cd RL_Train/utils
python3 generate_demon.py --env-id [env_id] --model_path [model_path] --num-actions 1000000 --random
```

data will be saved at `RL_Train/utils/data/[env_id]/random.npz`

### 2.4. Decompose .npz

```
cd RL_Train/utils
python3 split.py --env-id [env_id] --agent [agent] --data_name [data_name]
```

data will be saved at `RL_Train/utils/data/[env_id]/[agent]_[data_name].npy`

### 2.5. Train Auto-encoder

```
cd AE_Train
python3 train.py --env_id [env_id] --native_dim [native_dim] --latent_dim [latent_dim] --encoder_model [encoder_model] --hidden_layer 64 --batch_size 64 --data_path [data_path] --save_dir [save_dir] --epoch 100 --lr 1e-4 --num_actions 100000
```

models with a visualization of the running loss will be saved in the directory `AE_train/[save_dir]`

### 2.6. Train RL agents with learned latent action space in source environment

```
cd RL_Train
python3 train.py --algo td3 --gym-packages gym-mujoco --env [env_id] --env-kwargs latent_dim:[latent_dim] encoder_model:\"[encoder_model]\" native_dim:[native_dim] hidden_layer:64 model_path:\"[model_path]\" --seed [seed] --n-timesteps 1000000
```

### 2.7. Train RL agents with learned latent action space in target environment

```
cd RL_Train
```

- Ant-v3: `python3 train.py --algo td3 --gym-packages gym-mujoco --env Ant-v3 --env-kwargs latent_dim:[latent_dim] encoder_model:[encoder_model] native_dim:[native_dim] hidden_layer:64 model_path:[model_path] mass:0.5,2.0 friction:0.5,1.5 slope:-0.1,0.1 gravity:0.5,2.0 --seed [seed] --n-timesteps 1000000`
- ReacherTracker20-v3: `python3 train.py --algo td3 --gym-packages gym-mujoco --env ReacherTracker20-v3 --env-kwargs latent_dim:[latent_dim] encoder_model:[encoder_model] native_dim:[native_dim] hidden_layer:64 model_path:[model_path] mass:0.5,2.0 damping:1,10 --seed [seed] --n-timesteps 1000000`
- Swimmer20-v3: `python3 train.py --algo td3 --gym-packages gym-mujoco --env Swimmer20-v3 --env-kwargs latent_dim:[latent_dim] encoder_model:[encoder_model] native_dim:[native_dim] hidden_layer:64 model_path:[model_path] density:2000,6000 damping:1,5 --seed [seed] --n-timesteps 1000000`
- Other environments (Hopper-v3, HalfCheetah-v3, Walker2d-v3): `python3 train.py --algo td3 --gym-packages gym-mujoco --env [env_id] --env-kwargs latent_dim:[latent_dim] encoder_model:[encoder_model] native_dim:[native_dim] hidden_layer:64 model_path:[model_path] randomize_slope:True randomize_gravity:True randomize_mass:True randomize_friction:True --seed [seed] --n-timesteps 1000000`
