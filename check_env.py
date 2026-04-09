import gymnasium as gym
import flappy_bird_gymnasium


def main():
    env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)
    state, info = env.reset()
    print("Environment check passed")
    print(f"Observation shape: {env.observation_space.shape}")
    print(f"Action count: {env.action_space.n}")
    print(f"Initial state sample: {state}")

    for step_idx in range(5):
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(
            f"Step {step_idx + 1}: reward={reward:.2f}, "
            f"done={terminated or truncated}, score={info.get('score', 0)}"
        )
        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
