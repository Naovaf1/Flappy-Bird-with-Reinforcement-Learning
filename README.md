# Flappy Bird with Reinforcement Learning

This repository is a workshop-friendly Flappy Bird project that shows the difference between:

- a normal game controlled by a human
- the same game controlled by a trained Reinforcement Learning agent

The goal is not to build Flappy Bird from scratch. The goal is to start from an existing game environment and add a DQN agent on top of it.

## What is inside

- `manual_play.py`: play the game yourself with the keyboard
- `train.py`: train a DQN agent
- `play.py`: load a trained model and watch the AI play
- `check_env.py`: quick environment check
- `agent.py`: DQN model and agent logic
- `skeleton/agent_skeleton.py`: incomplete workshop version for students
- `skeleton/train_skeleton.py`: incomplete workshop version for students
- `models/dqn_flappy_best.pth`: backup trained model for demos

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

## Run the game yourself

```bash
python manual_play.py
```

Controls:

- `Space` or `Up Arrow`: flap
- `R`: restart after game over
- `Esc`: quit

## Train the AI

Short demo run:

```bash
python train.py --episodes 50
```

Longer run:

```bash
python train.py --episodes 1000 --plot training_progress_1000.png
```

This saves model checkpoints into the `models/` folder.

## Watch the AI play

```bash
python play.py --model .\models\dqn_flappy_best.pth --games 1
```

Controls while watching:

- close the window to stop
- `Esc` to stop

## Workshop flow

Recommended classroom sequence:

1. `python check_env.py`
2. `python manual_play.py`
3. open and complete `skeleton/agent_skeleton.py`
4. open and complete `skeleton/train_skeleton.py`
5. `python train.py --episodes 50`
6. `python play.py --model .\models\dqn_flappy_best.pth --games 1`

## Human vs AI version

The two important files to compare are:

- `manual_play.py`: action comes from keyboard input
- `play.py`: action comes from `agent.select_action(state)`

The game environment is the same in both. The main difference is who chooses the action.

## Backup material

- `models/dqn_flappy_best.pth`: trained checkpoint for live demos
- `README_WORKSHOP.md`: extra workshop notes
- `training_progress_1000.png`: sample training curve

## Troubleshooting

If `play.py` says the model is missing, run:

```bash
python train.py --episodes 50
```

If a game window does not open, verify the environment first:

```bash
python check_env.py
```
