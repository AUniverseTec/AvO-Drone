from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv

env = DummyVecEnv([lambda: DroneEnv()])

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000)

model.save("drone_rl_model")
