import argparse
import os
import time

import flappy_bird_gymnasium
import gymnasium as gym
import pygame

from agent import DQNAgent


DEFAULT_MODEL_PATH = "models/dqn_flappy_1000_best.pth"


def resolve_model_path(model_path):
    if model_path and os.path.exists(model_path):
        return model_path

    for candidate in (
        DEFAULT_MODEL_PATH,
        "models/dqn_flappy_300_best.pth",
        "models/dqn_flappy_50_best.pth",
        "models/dqn_flappy_best.pth",
        "models/dqn_flappy_final.pth",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def play(model_path=DEFAULT_MODEL_PATH, num_games=3, render=True, fps=30):
    """Load a model and watch the agent play."""
    resolved_model_path = resolve_model_path(model_path)
    if resolved_model_path is None:
        print(f"Model not found: {model_path}")
        print("Run train.py first so a checkpoint is created in the models folder.")
        return

    print(f"Loading model from: {resolved_model_path}")

    render_mode = "human" if render else None
    env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    agent.load(resolved_model_path)

    for game_idx in range(num_games):
        state, info = env.reset()
        total_reward = 0.0
        done = False
        score = int(info.get("score", 0))
        manual_stop = False

        while not done:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        manual_stop = True
                        done = True
                        break
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        manual_stop = True
                        done = True
                        break

            if manual_stop:
                break

            action = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            score = int(info.get("score", 0))

            if render:
                time.sleep(1 / max(fps, 1))

        if manual_stop:
            print("Playback stopped by user.")
            break

        print(f"Game {game_idx + 1}: score={score} | total_reward={total_reward:.1f}")
        if render:
            time.sleep(1)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a trained DQN agent play Flappy Bird.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to model checkpoint.")
    parser.add_argument("--games", type=int, default=3, help="Number of games to run.")
    parser.add_argument("--no-render", action="store_true", help="Run without opening the game window.")
    parser.add_argument("--fps", type=int, default=30, help="Playback speed when rendering.")
    args = parser.parse_args()

    play(args.model, num_games=args.games, render=not args.no_render, fps=args.fps)
