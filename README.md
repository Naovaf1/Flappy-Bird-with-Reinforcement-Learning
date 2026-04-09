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

## Path A: Quick demo

Use this path if you only want to try the project and see the final result.
This path is only for demo, not for training.

1. Run the game yourself:

```bash
python manual_play.py
```

2. Watch the backup AI model play:

```bash
python play.py --model .\models\dqn_flappy_best.pth --games 1
```

Notes:

- `models/dqn_flappy_best.pth` is the backup trained model included in the repository
- in Quick demo mode, you do not need to run `train.py`
- if you already trained a weak model and overwrote the backup file, download the latest ZIP from GitHub again to restore the original backup model

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

Longer training example:

```bash
python train.py --episodes 1000 --best-model .\models\dqn_flappy_1000_best.pth --final-model .\models\dqn_flappy_1000_final.pth --plot training_progress_1000.png
python play.py --model .\models\dqn_flappy_1000_best.pth --games 1
```

This makes it easy to compare a weaker model and a stronger model without confusion.

Important:

- by default, `train.py` overwrites `models/dqn_flappy_best.pth` and `models/dqn_flappy_final.pth`
- use `--best-model` and `--final-model` if you want to keep multiple checkpoints
- if your local copy says `unrecognized arguments: --best-model`, your ZIP is from an older version of the repository and you should download the latest ZIP again

## Run the game yourself

```bash
python manual_play.py
```

Controls:

- `Space` or `Up Arrow`: flap
- `R`: restart after game over
- `Esc`: quit

## Train the AI

These commands use the completed reference files `agent.py` and `train.py`.

If you are running the workshop exercise, finish the skeleton files first.

Warning:

- by default, training overwrites `models/dqn_flappy_best.pth`
- if you want to compare multiple trained models, use `--best-model` and `--final-model` with different filenames

Simple run:

```bash
python train.py --episodes 50
```

Longer run:

```bash
python train.py --episodes 1000 --plot training_progress_1000.png
```

Safer comparison examples:

```bash
python train.py --episodes 50 --best-model .\models\dqn_flappy_50_best.pth --final-model .\models\dqn_flappy_50_final.pth --plot training_progress_50.png
python train.py --episodes 1000 --best-model .\models\dqn_flappy_1000_best.pth --final-model .\models\dqn_flappy_1000_final.pth --plot training_progress_1000.png
```

## Watch the AI play

Backup model:

```bash
python play.py --model .\models\dqn_flappy_best.pth --games 1
```

Specific trained models:

```bash
python play.py --model .\models\dqn_flappy_50_best.pth --games 1
python play.py --model .\models\dqn_flappy_1000_best.pth --games 1
```

Controls while watching:

- close the window to stop
- `Esc` to stop

## Workshop flow

Recommended classroom sequence:

1. `python check_env.py`
2. `python manual_play.py`
3. `python play.py --model .\models\dqn_flappy_best.pth --games 1`
4. explain the difference between `manual_play.py` and `play.py`
5. open and complete `skeleton/agent_skeleton.py`
6. open and complete `skeleton/train_skeleton.py`
7. compare with `agent.py` and `train.py`
8. train a short model and play it
9. train a longer model and play it
10. compare the results

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

If `play.py` says the model is missing, or the backup model was accidentally overwritten, download the latest ZIP from GitHub again.

If a game window does not open, verify the environment first:

```bash
python check_env.py
```
