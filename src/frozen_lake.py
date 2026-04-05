from pathlib import Path

import gymnasium as gym
import numpy as np

NUM_EPISODES = 15_000
MODELS_DIR = Path(__file__).parents[1] / "models"
Q_TABLE_FILENAME = (MODELS_DIR / "q_table_frozen_lake.txt").resolve()


def run(render: bool = False, is_training: bool = False):
    env = gym.make(
        "FrozenLake-v1",
        map_name="8x8",
        is_slippery=False,
        render_mode="human" if render else None,
    )

    if is_training:
        q = np.zeros((64, 4))
    else:
        with open(Q_TABLE_FILENAME, "r") as f:
            q = np.loadtxt(f)

    learning_rate = 0.9  # alpha
    discount_factor = 0.9  # gamma

    epsilon = 1
    epsilon_decay_rate = 0.0001  # will take 10'000 episodes until epsilon reaches 0
    rng = np.random.default_rng()

    for _ in range(NUM_EPISODES):
        state = env.reset()[0]  # 0-63, 0=top left, 63=bottom right
        terminated = False  # True when if fell in hole or reached goal
        truncated = False  # True when num actions >200

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                # Random action
                action = (
                    env.action_space.sample()
                )  # actions: 0=left, 1=down, 2=right, 3=up
            else:
                # Use q table
                # As episodes progress, we pick this path more often
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                q[state, action] = q[state, action] + learning_rate * (
                    reward + discount_factor * np.max(q[new_state, :]) - q[state, action]  # type: ignore
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Stabilize q values after we are no longer exploring
        if epsilon == 0:
            learning_rate = 0.0001

        env.close()

    if is_training:
        with open(Q_TABLE_FILENAME, "w+") as f:
            np.savetxt(f, q)


if __name__ == "__main__":
    is_training = False
    render = not is_training
    run(render=render, is_training=is_training)
