# 2048 Gym Environment

## Introduction

This repository contains a custom Gym environment for the popular game 2048.
It is designed to be compatible with the Gymnasium API (formerly OpenAI Gym),
allowing for the integration of reinforcement learning algorithms.
The environment is implemented in Python and is suitable for anyone interested
in experimenting with RL agents for 2048.

## Installation

### Using Poetry
This project uses [Poetry](https://python-poetry.org/) for dependency management.

To install the environment, follow these steps:

**Install Poetry**: Follow the [instructions](https://python-poetry.org/docs/#installation) on the official Poetry website.

**Clone the Repository**

**Install**

TODO TODO TODO


## Basic Usage

To use the 2048 Gym environment in your Python project, you can follow this simple example:

TODO TODO TODO: revisit this example

```python
import gymnasium as gym
import gym_2048

# Create the environment
env = gym.make('Gym2048-v0')

# Reset the environment to start a new game
state = env.reset()

# Game loop
done = False
while not done:
    # Take a random action
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    # Render the current state (optional)
    env.render()
```



## License
This project is licensed under the MIT License.

