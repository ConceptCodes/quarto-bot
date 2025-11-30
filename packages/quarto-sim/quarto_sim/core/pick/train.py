import os
import numpy as np
from env import PickEnv
import mujoco.viewer
import time


class SimplePolicy:
    """Simple random policy with noise decay for basic RL"""

    def __init__(self, action_dim=6):
        self.action_dim = action_dim
        self.noise_level = 1.0
        self.noise_decay = 0.995

    def get_action(self, obs):
        # Simple policy: random actions with decaying noise
        action = np.random.randn(self.action_dim) * self.noise_level
        return np.clip(action, -1, 1)

    def update_noise(self):
        self.noise_level *= self.noise_decay


def main():
    env = PickEnv()
    policy = SimplePolicy(action_dim=6)

    n_episodes = 1_000
    steps_per_episode = 200

    episode_rewards = []
    success_count = 0

    # Use the blocking viewer (no threading)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(n_episodes):
            obs = env.reset()
            episode_reward = 0

            for t in range(steps_per_episode):
                # Get action from policy in [-1, 1]
                raw_action = policy.get_action(obs)
                # Map action to actuated joint limits only
                actuator_joint_ids = env.model.actuator_trnid[:, 0]
                jnt_range = np.array(env.model.jnt_range)[
                    actuator_joint_ids
                ]  # shape: (n_actuators, 2)
                action = (
                    0.5 * (raw_action + 1) * (jnt_range[:, 1] - jnt_range[:, 0])
                    + jnt_range[:, 0]
                )
                action = np.clip(action, jnt_range[:, 0], jnt_range[:, 1])

                # Step environment
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                viewer.sync()
                time.sleep(0.01)  # Small delay for visualization

                if done:
                    # Only count as success if gripper reached the piece
                    if info.get("distance", 1.0) < 0.03:
                        success_count += 1
                        print(f"Episode {ep+1}: SUCCESS! Reward: {episode_reward:.2f}")
                    else:
                        print(
                            f"Episode {ep+1}: DONE (not success). Reward: {episode_reward:.2f}"
                        )
                    break

            if not done:
                print(f"Episode {ep+1}: Reward: {episode_reward:.2f}")

            episode_rewards.append(episode_reward)
            policy.update_noise()

            # Print progress every 10 episodes
            if (ep + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(
                    f"Episodes {ep-8}-{ep+1}: Avg Reward: {avg_reward:.2f}, Success Rate: {success_count/(ep+1)*100:.1f}%"
                )

    print(f"\nTraining Complete!")
    print(f"Total Success Rate: {success_count/n_episodes*100:.1f}%")
    print(f"Final Average Reward: {np.mean(episode_rewards[-10:]):.2f}")


if __name__ == "__main__":
    main()
