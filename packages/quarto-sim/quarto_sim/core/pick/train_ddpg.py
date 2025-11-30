import numpy as np
from stable_baselines3 import DDPG
import gym
from stable_baselines3.common.noise import NormalActionNoise

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from gym_wrapper import PickEnvGym


# Create and check the environment

# Wrap environment with normalization for observations
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv([lambda: PickEnvGym()])
env = VecNormalize(env, norm_obs=True, norm_reward=False)
check_env(env.envs[0], warn=True)

# Stage 1: DDPG hyperparameter tuning for exploration and stability
buffer_size = 200_000
batch_size = 128
learning_starts = 5000

# Action noise for exploration
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
)

learning_rate = 1e-4
policy_kwargs = dict(net_arch=[400, 300])

model = DDPG(
    "MlpPolicy",
    env,
    verbose=1,
    buffer_size=buffer_size,
    learning_starts=learning_starts,
    batch_size=batch_size,
    tau=0.005,
    gamma=0.99,
    train_freq=(1, "episode"),
    gradient_steps=1,
    action_noise=action_noise,
    learning_rate=learning_rate,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./ddpg_tensorboard/",
)


class SaveEveryNEpisodesCallback(BaseCallback):
    def __init__(
        self, save_freq=10, save_path="ddpg_pickenv", preview_steps=50, verbose=0
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.episode_count = 0
        self.preview_steps = preview_steps

    def _on_step(self) -> bool:
        # Check if episode is done
        if self.locals.get("dones") is not None:
            # For vectorized envs, dones is an array
            if any(self.locals["dones"]):
                self.episode_count += 1
                if self.episode_count % self.save_freq == 0:
                    # Save a video of the agent

                    video_folder = f"videos/ep{self.episode_count}"
                    video_length = 1_00
                    eval_env = PickEnvGym(render_mode="rgb_array")
                    eval_env = gym.wrappers.RecordVideo(
                        eval_env,
                        video_folder=video_folder,
                        episode_trigger=lambda x: True,
                    )
                    obs, _ = eval_env.reset()
                    for _ in range(video_length):
                        action, _ = self.model.predict(obs, deterministic=False)
                        obs, reward, terminated, truncated, info = eval_env.step(action)
                        done = terminated or truncated
                        if done:
                            obs, _ = eval_env.reset()
                    print(f"Video saved to {video_folder}/")
        return True


# Evaluation callback for best model saving
eval_env = DummyVecEnv([lambda: PickEnvGym(render_mode="rgb_array")])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./ddpg_best_model/",
    log_path="./ddpg_eval_logs/",
    eval_freq=5000,
    deterministic=True,
    render=False,
)

# Train the agent with callbacks (video + evaluation)
save_callback = SaveEveryNEpisodesCallback(save_freq=1_000, save_path="ddpg_pickenv")
model.learn(total_timesteps=800_000, callback=[save_callback, eval_callback])

# Save the model and normalization statistics
model.save("ddpg_pickenv")
env.save("ddpg_pickenv_vecnormalize.pkl")
