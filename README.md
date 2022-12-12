# CS839-FinalProject

## Python environment configuration

## \[env_id\] supported by current implementations are

- Ant-v3
- HalfCheetah-v3
- Hopper-v3
- Walker2d-v3
- Swimmer20-v3
- ReacherTracker20-v3

## Learning process

### Train native agent

```
cd RL_Train
python3 train.py --algo td3 --gym-packages gym-mujoco --env \[env_id\] --seed \[random-seed\] --n-timesteps 2000000
tar -czf \[env_id\]_native_\[seed\].tar.gz logs/
```
