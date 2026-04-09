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
- `models/dqn_flappy_50_best.pth`: weak demo model
- `models/dqn_flappy_500_best.pth`: medium demo model
- `models/dqn_flappy_1000_best.pth`: strong score-base demo model
- `models/dqn_flappy_strong_best.pth`: tuned stability-first demo model

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

## Human vs AI version

The two important files to compare are:

- `manual_play.py`: action comes from keyboard input
- `play.py`: action comes from `agent.select_action(state)`

The game environment is the same in both. The main difference is who chooses the action.

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
