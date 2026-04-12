# README Theory Backup

This file stores the theory-first README sections that were removed from the main workshop README so they can still be reused later for slides or other projects.

## Introduction to Reinforcement Learning

Reinforcement Learning (RL) is a branch of machine learning where an agent learns by interacting with an environment. Instead of learning from labeled examples, the agent receives rewards or penalties and gradually improves its behavior to maximize long-term reward.

## Core Concepts

| Concept | Meaning in this project |
|---|---|
| Agent | The controller that decides when the bird should flap |
| Environment | The Flappy Bird game world |
| State | The observation returned by the game at the current step |
| Action | Flap or do nothing |
| Reward | Positive or negative feedback after each action |
| Policy | The strategy the agent uses to choose actions |
| Q-value | The estimated long-term value of taking an action in a state |
| Discount factor | How much future rewards matter compared with immediate rewards |

## The RL Loop

```text
  +-------+    action     +-------------+
  | Agent | ----------->  | Environment |
  +-------+  <----------- +-------------+
            state, reward
```

The loop is simple:

1. the environment gives the current state
2. the agent chooses an action
3. the environment returns the next state and reward
4. the agent updates its behavior from repeated experience

## Why Flappy Bird

Flappy Bird is a strong RL teaching example because:

- the action space is small and easy to explain
- the game is visually clear even to people who do not know RL
- the timing challenge is still hard enough to be interesting
- weak and strong models look very different on screen
- it is a good bridge between theory, code, and live demo

## Why DQN

This project uses Deep Q-Network (DQN), a reinforcement learning method that replaces a Q-table with a neural network. That lets the agent handle a continuous or structured state instead of memorizing every possible situation exactly.

## Q-Learning Foundation

At a high level, Q-learning tries to estimate how good an action is in a given state:

```text
Q(s, a) = r + gamma * max Q(s', a')
```

Where:

- `r` is the immediate reward
- `gamma` is the discount factor
- `s'` is the next state
- `max Q(s', a')` is the best expected future value from the next state

Traditional Q-learning works well for small state spaces, but it becomes impractical when the state is larger or continuous.

## From Q-Learning to DQN

DQN replaces the lookup table with a neural network that approximates Q-values. In this repository, the network learns from many game transitions and gradually improves how it values flap vs no-flap in different situations.

The repository mainly focuses on:

- experience replay
- target networks
- epsilon-greedy exploration
- Double DQN for the stronger tuned model

The goal here is practical understanding rather than a full research implementation. The code is meant to be readable enough for demos, classroom explanation, and follow-up experiments.

## DQN Building Blocks

### Experience Replay

The agent stores past transitions and samples them later in random mini-batches. This helps reduce the strong correlation between consecutive frames and makes learning more stable.

### Target Network

A separate target network is used when computing the learning target. This slows down moving-target effects and improves training stability.

### Epsilon-Greedy Exploration

The agent does not always choose the current best action. Early in training it explores more, and later it gradually shifts toward exploitation as epsilon decays.

### Double DQN

The stronger tuned model uses Double DQN to reduce overestimation. One network chooses the next action, while the target network evaluates it.

## Project Architecture

```text
                    +--------------------+
                    |  train.py config   |
                    +----------+---------+
                               |
                               v
+-----------+    state    +---------+    action    +-------------------+
| Flappy    | --------->  | Agent   | ---------->  | Flappy Bird Env   |
| Bird Game | <---------  | (DQN)   | <----------  | (Gymnasium)       |
+-----------+    reward   +---------+   next state +-------------------+
                               |
                               v
                    +--------------------+
                    | Replay Memory      |
                    +--------------------+
                               |
                               v
                    +--------------------+
                    | Policy Network     |
                    | Target Network     |
                    +--------------------+
                               |
                               v
                    +--------------------+
                    | Models / Plots     |
                    | Demo Checkpoints   |
                    +--------------------+
```
