import gym
from gym.envs.registration import register

# unregister gym's env so I can use the same name
envs_to_unregister = [
	'Ant-v3', 
	'HalfCheetah-v3', 
	'Walker2d-v3',
	'Hopper-v3',
	'Swimmer-v3',
	'Humanoid-v3',
	]
for env_id in envs_to_unregister:
	if env_id in gym.envs.registry.env_specs:
		del gym.envs.registry.env_specs[env_id]

# register(
# 	id='Walker2d-v3',
# 	entry_point='gym_mujoco.envs:Walker2dEnv',
# 	max_episode_steps=1000,
# )

register(
	id='Humanoid-v3',
	entry_point='gym_mujoco.envs:HumanoidEnv',
	max_episode_steps=1000,
)

register(
	id='Ant-v3',
	entry_point='gym_mujoco.envs:AntEnv',
	max_episode_steps=1000,
)

register(
	id='HalfCheetah-v3',
	entry_point='gym_mujoco.envs:HalfCheetahEnv',
	max_episode_steps=1000,
)

register(
	id='Walker2d-v3',
	entry_point='gym_mujoco.envs:Walker2dEnv',
	max_episode_steps=1000,
)

register(
	id='Hopper-v3',
	entry_point='gym_mujoco.envs:HopperEnv',
	max_episode_steps=1000,
)

register(
	id='Swimmer-v3',
	entry_point='gym_mujoco.envs:SwimmerEnv',
	max_episode_steps=1000,
)

# register(
# 	id='Reacher10-v3',
# 	entry_point='gym_mujoco.envs:ReacherEnv',
# 	kwargs={'num_links': 10},
# 	max_episode_steps=100,
# )

register(
	id='ReacherTracker10-v3',
	entry_point='gym_mujoco.envs:ReacherTrackerEnv',
	kwargs={'num_links': 10},
	max_episode_steps=200,
)

# register(
# 	id='Reacher20-v3',
# 	entry_point='gym_mujoco.envs:ReacherEnv',
# 	kwargs={'num_links': 20},
# 	max_episode_steps=200,
# )

register(
	id='ReacherTracker20-v3',
	entry_point='gym_mujoco.envs:ReacherTrackerEnv',
	kwargs={'num_links': 20},
	max_episode_steps=200,
)


register(
	id='Swimmer6-v3',
	entry_point='gym_mujoco.envs:SwimmerOldEnv',
	kwargs={'num_links': 6},
	max_episode_steps=1000,
)

register(
	id='Swimmer10-v3',
	entry_point='gym_mujoco.envs:SwimmerOldEnv',
	kwargs={'num_links': 10},
	max_episode_steps=1000,
)

register(
	id='Swimmer20-v3',
	entry_point='gym_mujoco.envs:SwimmerOldEnv',
	kwargs={'num_links': 20},
	max_episode_steps=1000,
)

# register(
# 	id='Swimmer20-1k-v3',
# 	entry_point='gym_mujoco.envs:SwimmerEnv',
# 	kwargs={'num_links': 20},
# 	max_episode_steps=1000,
# )

# register(
# 	id='Toy2d-v0',
# 	entry_point='gym_mujoco.envs:Toy2dEnv',
# 	max_episode_steps=20,
# )

