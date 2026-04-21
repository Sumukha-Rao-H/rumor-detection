# train.py - Simulated training loop that calls reward.get_reward()

from reward import get_reward


def main():
    """Run a simulated training loop for 100 iterations."""
    total_reward = 0.0
    num_iterations = 100

    for _ in range(num_iterations):
        total_reward += get_reward()

    avg_reward = total_reward / num_iterations
    print(f"AVG_REWARD={avg_reward}")


if __name__ == "__main__":
    main()
