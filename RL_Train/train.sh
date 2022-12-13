cd /Users/quyuxiao/Desktop/SummerResearch/SALAS/repo/AERL/gym-mujoco
python3 -m pip uninstall gym_mujoco
python3 -m pip install -e .
cd /Users/quyuxiao/Desktop/SummerResearch/SALAS/repo/AERL
python3 train.py --algo td3 --gym-packages gym-mujoco --env Ant-v3 --env-kwargs latent_dim:4 encoder_model:\"OTNAE\" native_dim:8 hidden_layer:128 model_path:\"/Users/quyuxiao/Desktop/SummerResearch/SALAS/repo/SALAS/sanity_check/models/Ant-v3_OTNAE_0.0001/OTNAE_8.pt\"