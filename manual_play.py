import time

import flappy_bird_gymnasium
import gymnasium as gym
import pygame


def manual_play(fps=30):
    """Open Flappy Bird and let a human play with Space or Up."""
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)

    print("Manual play mode")
    print("Press Space or Up Arrow to flap")
    print("Press R to restart after game over")
    print("Press Esc or close the game window to quit")

    try:
        state, info = env.reset()
        done = False
        score = int(info.get("score", 0))

        while True:
            action = 0
            restart_requested = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_SPACE, pygame.K_UP):
                        action = 1
                    elif event.key == pygame.K_ESCAPE:
                        return
                    elif event.key == pygame.K_r and done:
                        restart_requested = True

            if restart_requested:
                state, info = env.reset()
                done = False
                score = int(info.get("score", 0))
                continue

            if not done:
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                score = int(info.get("score", 0))
                pygame.display.set_caption("Flappy Bird | Space/Up = flap | Esc = quit")
            else:
                pygame.display.set_caption(f"Flappy Bird | Game Over | score={score} | press R to restart")

            time.sleep(1 / max(fps, 1))
    finally:
        env.close()


if __name__ == "__main__":
    manual_play()
