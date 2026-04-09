# Flappy Bird with Reinforcement Learning

This repository is a workshop-friendly Flappy Bird project built around one simple comparison:

- a normal game controlled by a human
- the same game controlled by a trained Reinforcement Learning agent

The goal is not to build Flappy Bird from scratch. The goal is to start from an existing game environment and add a DQN agent on top of it, then use that result as both a demo and a guided workshop example.

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

## What is inside

- `manual_play.py`: play the game yourself with the keyboard
- `train.py`: train a DQN agent
- `play.py`: load a trained model and watch the AI play
- `check_env.py`: quick environment check
- `agent.py`: DQN model and agent logic
- `skeleton/agent_skeleton.py`: incomplete workshop version for students
- `skeleton/train_skeleton.py`: incomplete workshop version for students
- `models/dqn_flappy_50_best.pth`: weak demo model
- `models/dqn_flappy_500_best.pth`: medium demo model
- `models/dqn_flappy_1000_best.pth`: strong score-base demo model
- `models/dqn_flappy_strong_best.pth`: tuned stability-first demo model

## Project structure

```text
Flappy bird/
|-- agent.py
|-- train.py
|-- play.py
|-- manual_play.py
|-- check_env.py
|-- requirements.txt
|-- models/
|-- skeleton/
|-- notes/
```

Main roles:

- `manual_play.py`: human-controlled baseline
- `play.py`: AI-controlled demo playback
- `train.py`: training and evaluation flow
- `agent.py`: DQN model, replay memory, checkpoint loading and saving
- `skeleton/`: classroom starter files for students
- `models/`: prepared checkpoints for quick comparison

## Requirements

- Windows PowerShell or Command Prompt
- Python 3.10+ recommended

An IDE is optional. You can complete the workshop with just Python and a terminal.

## Setup

1. Download this repository as ZIP or clone it.
2. Open PowerShell or Command Prompt in this folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Check that the Flappy Bird environment works:

```bash
python check_env.py
```

## Human vs AI demo

One of the clearest parts of this repository is that the environment stays the same while the decision-maker changes.

- In `manual_play.py`, the player decides when to flap.
- In `play.py`, the trained model decides when to flap.

That makes it easy to explain the core idea of RL in class: the game itself does not change, only the policy that chooses the next action.

## Path A: Quick demo

Use this path if you only want to try the project and see the final result.
This path is only for demo, not for training.

1. Run the game yourself:

```bash
python manual_play.py
```

Controls:

- `Space` or `Up Arrow`: flap
- `R`: restart after game over
- `Esc`: quit

2. Watch the backup AI model play.

The numbers `50`, `500`, and `1000` are training episode counts.
They represent how long each AI brain was trained before saving the checkpoint.
In general, a model trained for more episodes should perform better, but the goal here is to let you compare weak, medium, strong, and tuned-stable behavior side by side.

Choose the AI brain first:

```bash
python play.py --model .\models\dqn_flappy_50_best.pth --games 1
```

```bash
python play.py --model .\models\dqn_flappy_500_best.pth --games 1
```

```bash
python play.py --model .\models\dqn_flappy_1000_best.pth --games 1
```

```bash
python play.py --model .\models\dqn_flappy_strong_best.pth --games 1
```

Default AI playback:

```bash
python play.py --games 1
```

Notes:

- `models/dqn_flappy_50_best.pth` is a weak model for comparison
- `models/dqn_flappy_500_best.pth` is a medium model for comparison
- `models/dqn_flappy_1000_best.pth` is the strong score-base model in this repository
- `models/dqn_flappy_strong_best.pth` is a separately tuned stability-first model
- `python play.py --games 1` now defaults to the strongest AI model automatically
- the first four commands let you choose the AI brain explicitly before watching
- `dqn_flappy_1000_best.pth` can spike to a higher score in some runs
- `dqn_flappy_strong_best.pth` was tuned to reduce early failures and be more stable across runs
- in Quick demo mode, you do not need to run `train.py`
- if you already trained a weak model and overwrote files locally, download the latest ZIP from GitHub again to restore the backup models

## Workshop flow

Recommended classroom sequence:

1. `python check_env.py`
2. `python manual_play.py`
3. `python play.py --model .\models\dqn_flappy_50_best.pth --games 1`
4. `python play.py --model .\models\dqn_flappy_500_best.pth --games 1`
5. `python play.py --model .\models\dqn_flappy_1000_best.pth --games 1`
6. `python play.py --model .\models\dqn_flappy_strong_best.pth --games 1`
7. explain the difference between `manual_play.py` and `play.py`
8. open and complete `skeleton/agent_skeleton.py`
9. open and complete `skeleton/train_skeleton.py`
10. compare with `agent.py` and `train.py`
11. train a short model and play it
12. train a longer model and compare the results

## Path B: Workshop mode

Use this path if students are going to complete missing logic and train their own models.

1. Run the game yourself first:

```bash
python manual_play.py
```

2. Open and complete:

- `skeleton/agent_skeleton.py`
- `skeleton/train_skeleton.py`

3. Compare with the reference files if needed:

- `agent.py`
- `train.py`

4. Train and compare multiple models.

Short training example:

```bash
python train.py --episodes 50 --best-model .\models\dqn_flappy_50_best.pth --final-model .\models\dqn_flappy_50_final.pth --plot training_progress_50.png
python play.py --model .\models\dqn_flappy_50_best.pth --games 1
```

Medium training example:

```bash
python train.py --episodes 500 --best-model .\models\dqn_flappy_500_best.pth --final-model .\models\dqn_flappy_500_final.pth --plot training_progress_500.png
python play.py --model .\models\dqn_flappy_500_best.pth --games 1
```

Strong score-base training example:

```bash
python train.py --episodes 1000 --best-model .\models\dqn_flappy_1000_best.pth --final-model .\models\dqn_flappy_1000_final.pth --plot training_progress_1000.png
python play.py --model .\models\dqn_flappy_1000_best.pth --games 1
```

Strong tuned training example:

```bash
python train.py --episodes 1500 --resume-from .\models\dqn_flappy_1000_best.pth --best-model .\models\dqn_flappy_strong_best.pth --final-model .\models\dqn_flappy_strong_final.pth --plot training_progress_strong.png --checkpoint-label strong_base --hidden-sizes 64,64 --start-epsilon 0.05 --epsilon-min 0.01 --epsilon-decay 0.99997 --learning-rate 0.0003 --batch-size 128 --memory-capacity 50000 --double-dqn --loss-type huber --grad-clip 5 --learning-starts 128 --target-update-steps 1000 --eval-interval 100 --eval-games 12 --eval-score-limit 150 --failure-threshold 10 --stability-bonus 0.05
python play.py --model .\models\dqn_flappy_strong_best.pth --games 1
```

This makes it easy to compare a weaker model and a stronger model without confusion.

Important:

- by default, `train.py` overwrites `models/dqn_flappy_best.pth` and `models/dqn_flappy_final.pth`
- use `--best-model` and `--final-model` if you want to keep multiple checkpoints
- if your local copy says `unrecognized arguments: --best-model`, your ZIP is from an older version of the repository and you should download the latest ZIP again

## 1000 vs Strong

Quick tuning notes for classroom discussion after students compare the training results:

- `dqn_flappy_1000_best.pth` is the higher-risk, higher-peak score-base model
- `dqn_flappy_strong_best.pth` is a separate tuned model that focuses more on stability
- the tuned Strong model uses Double DQN, Huber loss, a lower learning rate, a larger batch size, a larger replay buffer, gradient clipping, and multi-game evaluation when selecting the best checkpoint
- in our tests, Strong reduced early low-score failures compared with `1000`, but it still cannot guarantee a perfect run every time because the pipe patterns are still variable and DQN is still an approximate policy

## Training notes

This repository includes multiple checkpoints so the class can compare different behaviors without retraining everything from scratch.

- `50` is a weak comparison point
- `500` is a medium comparison point
- `1000` is a strong score-oriented model
- `Strong` is a tuned follow-up model that prioritizes stability more than peak score

The tuned `Strong` version keeps the same general project structure but changes how the model is trained and selected. Compared with the `1000` version, it adds a more stability-focused training setup with Double DQN, Huber loss, a lower learning rate, a larger batch size, a larger replay buffer, gradient clipping, and multi-game evaluation for checkpoint selection.

This does not guarantee a perfect run every time. Flappy Bird still contains variation, and DQN is still an approximate method. The practical goal is to reduce early failures and make the demo behavior more reliable.

## References and background

If you want to extend the presentation or explain the theory in more depth, useful topics to mention are:

- Q-learning and the Bellman equation
- experience replay
- target networks
- epsilon-greedy exploration
- Double DQN

Useful references:

- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.*
- Van Hasselt, H. et al. (2016). *Deep Reinforcement Learning with Double Q-learning.*
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Flappy Bird Gymnasium](https://github.com/markub3327/flappy-bird-gymnasium)

## Backup material

- `models/dqn_flappy_50_best.pth`: weak checkpoint for comparison
- `models/dqn_flappy_500_best.pth`: medium checkpoint for comparison
- `models/dqn_flappy_1000_best.pth`: strong score-base checkpoint for comparison
- `models/dqn_flappy_strong_best.pth`: separately tuned stability-first checkpoint for comparison
- `training_progress_1000.png`: sample training curve

## Troubleshooting

If `play.py` says the model is missing, or the backup model was accidentally overwritten, download the latest ZIP from GitHub again.

If a game window does not open, verify the environment first:

```bash
python check_env.py
```
