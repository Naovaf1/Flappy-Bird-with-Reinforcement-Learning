import argparse
import os

import flappy_bird_gymnasium
import gymnasium as gym
import matplotlib.pyplot as plt

from agent import DQNAgent


def train(num_episodes=500, render=False):
    """Main training loop for the Flappy Bird DQN agent."""
    print(f"Starting training for {num_episodes} episodes...")

    env_mode = "human" if render else None
    env = gym.make("FlappyBird-v0", render_mode=env_mode, use_lidar=False)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size)

    scores = []
    best_score = float("-inf")

    os.makedirs("models", exist_ok=True)

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

        scores.append(total_reward)

        if episode % 10 == 0:
            agent.update_target_network()

        if total_reward > best_score:
            best_score = total_reward
            agent.save("models/dqn_flappy_best.pth")
            print(f"[New Best] Episode {episode}: reward={total_reward:.1f} saved to models/dqn_flappy_best.pth")

        if episode % 10 == 0:
            avg_score = sum(scores[-10:]) / min(10, len(scores))
            print(
                f"Episode {episode:4d}/{num_episodes} | "
                f"Reward: {total_reward:6.1f} | "
                f"Avg(10): {avg_score:6.1f} | "
                f"Best: {best_score:6.1f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    agent.save("models/dqn_flappy_final.pth")
    env.close()
    return scores


def plot_scores(scores, output_path="training_progress.png"):
    """Save a training curve plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(scores, alpha=0.5, color="blue", label="Reward per episode")

    window = 50
    if len(scores) >= window:
        moving_avg = [sum(scores[i : i + window]) / window for i in range(len(scores) - window + 1)]
        plt.plot(
            range(window - 1, len(scores)),
            moving_avg,
            color="red",
            linewidth=2,
            label=f"Moving Avg ({window})",
        )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Progress - Flappy Bird")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved training plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent for Flappy Bird.")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the game window during training. Slower but useful for demos.",
    )
    parser.add_argument(
        "--plot",
        default="training_progress.png",
        help="Output file for the training plot.",
    )
    args = parser.parse_args()

    scores = train(num_episodes=args.episodes, render=args.render)
    plot_scores(scores, output_path=args.plot)
