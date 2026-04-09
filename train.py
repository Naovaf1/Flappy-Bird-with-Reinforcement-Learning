import argparse
import os
import random

import flappy_bird_gymnasium
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import DQNAgent, load_checkpoint


def parse_hidden_sizes(hidden_sizes_text):
    return tuple(int(part.strip()) for part in hidden_sizes_text.split(",") if part.strip())


def make_env(render=False, use_lidar=False, score_limit=None):
    env_mode = "human" if render else None
    return gym.make(
        "FlappyBird-v0",
        render_mode=env_mode,
        use_lidar=use_lidar,
        score_limit=score_limit,
    )


def shape_reward(next_state, reward, done, use_lidar=False, stability_bonus=0.0):
    if done or stability_bonus <= 0 or use_lidar:
        return reward

    player_y = float(next_state[9])
    next_pipe_top = float(next_state[4])
    next_pipe_bottom = float(next_state[5])
    next_pipe_x = float(next_state[3])
    gap_center = (next_pipe_top + next_pipe_bottom) / 2.0
    gap_half = max((next_pipe_bottom - next_pipe_top) / 2.0, 1e-6)

    # Reward staying near the center of the upcoming gap when the bird approaches it.
    centered = max(0.0, 1.0 - abs(player_y - gap_center) / gap_half)
    approach_weight = max(0.0, 1.0 - min(next_pipe_x, 1.0))
    return reward + stability_bonus * centered * approach_weight


def evaluate_agent(
    agent,
    use_lidar=False,
    num_games=10,
    seed_base=10_000,
    score_limit=200,
    failure_threshold=10,
):
    eval_env = make_env(render=False, use_lidar=use_lidar, score_limit=score_limit)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    scores = []
    rewards = []

    for game_idx in range(num_games):
        state, info = eval_env.reset(seed=seed_base + game_idx)
        done = False
        total_reward = 0.0
        score = int(info.get("score", 0))

        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
            score = int(info.get("score", 0))

        scores.append(score)
        rewards.append(total_reward)

    agent.epsilon = original_epsilon
    eval_env.close()

    stats = {
        "scores": scores,
        "rewards": rewards,
        "failures": sum(score < failure_threshold for score in scores),
        "min_score": min(scores) if scores else 0,
        "median_score": float(np.median(scores)) if scores else 0.0,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "max_score": max(scores) if scores else 0,
        "score_limit_hits": sum(score >= score_limit for score in scores),
    }
    stats["metric"] = (
        -stats["failures"],
        stats["min_score"],
        stats["median_score"],
        stats["mean_score"],
        stats["max_score"],
        stats["score_limit_hits"],
    )
    return stats


def build_agent(env, args):
    return DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        hidden_sizes=args.hidden_sizes,
        gamma=args.gamma,
        epsilon=args.start_epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        memory_capacity=args.memory_capacity,
        use_double_dqn=args.double_dqn,
        loss_type=args.loss_type,
        grad_clip=args.grad_clip,
    )


def train(args):
    print(f"Starting training for {args.episodes} episodes...")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(render=args.render, use_lidar=args.use_lidar)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    agent = build_agent(env, args)

    if args.resume_from:
        state_dict, checkpoint_config, checkpoint_metadata = load_checkpoint(args.resume_from, map_location=agent.device)
        if checkpoint_config:
            print(f"Resume checkpoint metadata: {checkpoint_config}")
        if checkpoint_metadata:
            print(f"Resume checkpoint eval summary: {checkpoint_metadata.get('eval_stats', {})}")
        agent.load(args.resume_from, epsilon=args.start_epsilon)
        print(f"Resumed training from: {args.resume_from}")

    os.makedirs("models", exist_ok=True)

    scores = []
    training_rewards = []
    best_eval_metric = None
    best_eval_stats = None
    global_step = 0

    for episode in range(1, args.episodes + 1):
        state, info = env.reset()
        done = False
        total_reward = 0.0
        total_training_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            shaped_reward = shape_reward(
                next_state,
                reward,
                done,
                use_lidar=args.use_lidar,
                stability_bonus=args.stability_bonus,
            )

            agent.memory.push(state, action, shaped_reward, next_state, done)
            global_step += 1

            if global_step >= args.learning_starts and global_step % args.train_frequency == 0:
                agent.learn()
            if global_step % args.target_update_steps == 0:
                agent.update_target_network()

            state = next_state
            total_reward += reward
            total_training_reward += shaped_reward

        scores.append(int(info.get("score", 0)))
        training_rewards.append(total_training_reward)

        if episode % args.log_interval == 0 or episode == 1:
            avg_score = float(np.mean(scores[-args.log_interval :]))
            avg_reward = float(np.mean(training_rewards[-args.log_interval :]))
            print(
                f"Episode {episode:4d}/{args.episodes} | "
                f"Score: {scores[-1]:4d} | "
                f"AvgScore({min(args.log_interval, len(scores))}): {avg_score:6.2f} | "
                f"TrainReward: {total_training_reward:7.2f} | "
                f"AvgReward: {avg_reward:7.2f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

        should_evaluate = episode % args.eval_interval == 0 or episode == args.episodes
        if should_evaluate:
            eval_stats = evaluate_agent(
                agent,
                use_lidar=args.use_lidar,
                num_games=args.eval_games,
                seed_base=args.eval_seed_base + episode * 10,
                score_limit=args.eval_score_limit,
                failure_threshold=args.failure_threshold,
            )
            print(
                f"[Eval] Episode {episode}: failures<{args.failure_threshold}={eval_stats['failures']} | "
                f"min={eval_stats['min_score']} | median={eval_stats['median_score']:.1f} | "
                f"mean={eval_stats['mean_score']:.1f} | max={eval_stats['max_score']} | "
                f"scores={eval_stats['scores']}"
            )

            if best_eval_metric is None or eval_stats["metric"] > best_eval_metric:
                best_eval_metric = eval_stats["metric"]
                best_eval_stats = eval_stats
                metadata = {
                    "use_lidar": args.use_lidar,
                    "label": args.checkpoint_label,
                    "eval_stats": eval_stats,
                }
                agent.save(args.best_model, metadata=metadata)
                print(f"[New Best Eval] Saved stable checkpoint to {args.best_model}")

    final_metadata = {
        "use_lidar": args.use_lidar,
        "label": args.checkpoint_label,
        "best_eval_stats": best_eval_stats or {},
    }
    agent.save(args.final_model, metadata=final_metadata)
    print(f"Saved final checkpoint to {args.final_model}")
    env.close()
    return scores, training_rewards, best_eval_stats


def plot_scores(scores, output_path="training_progress.png", label="Score per episode"):
    plt.figure(figsize=(10, 5))
    plt.plot(scores, alpha=0.5, color="blue", label=label)

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
    plt.ylabel("Score")
    plt.title("DQN Training Progress - Flappy Bird")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved training plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent for Flappy Bird.")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes.")
    parser.add_argument("--render", action="store_true", help="Render the game window during training.")
    parser.add_argument("--use-lidar", action="store_true", help="Train with lidar observations for stronger awareness.")
    parser.add_argument("--plot", default="training_progress.png", help="Output file for the training plot.")
    parser.add_argument("--best-model", dest="best_model", default="models/dqn_flappy_best.pth")
    parser.add_argument("--final-model", dest="final_model", default="models/dqn_flappy_final.pth")
    parser.add_argument("--resume-from", help="Optional checkpoint to continue training from.")
    parser.add_argument("--hidden-sizes", type=parse_hidden_sizes, default=(64, 64), help="Comma-separated hidden sizes.")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-capacity", type=int, default=10000)
    parser.add_argument("--start-epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--double-dqn", action="store_true", help="Use Double DQN target selection.")
    parser.add_argument("--loss-type", choices=("mse", "huber"), default="mse")
    parser.add_argument("--grad-clip", type=float, help="Optional gradient clipping value.")
    parser.add_argument("--learning-starts", type=int, default=1000, help="Warm-up steps before learning starts.")
    parser.add_argument("--train-frequency", type=int, default=1, help="Gradient update frequency in environment steps.")
    parser.add_argument("--target-update-steps", type=int, default=500, help="Sync target network every N environment steps.")
    parser.add_argument("--eval-interval", type=int, default=100, help="Evaluate every N episodes.")
    parser.add_argument("--eval-games", type=int, default=12, help="Number of deterministic evaluation games.")
    parser.add_argument("--eval-score-limit", type=int, default=200, help="Cap evaluation games to keep runs bounded.")
    parser.add_argument("--failure-threshold", type=int, default=10, help="Scores below this count as unstable failures.")
    parser.add_argument("--stability-bonus", type=float, default=0.0, help="Extra reward scale for staying near the next gap center.")
    parser.add_argument("--checkpoint-label", default="default")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible runs.")
    parser.add_argument("--eval-seed-base", type=int, default=50_000, help="Base seed for deterministic evaluation episodes.")
    args = parser.parse_args()

    scores, training_rewards, best_eval_stats = train(args)
    plot_scores(scores, output_path=args.plot)
    if best_eval_stats:
        print(f"Best eval stats: {best_eval_stats}")
