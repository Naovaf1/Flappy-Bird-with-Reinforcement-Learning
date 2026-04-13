# Flappy Bird with Reinforcement Learning

This repository is a workshop-friendly Flappy Bird project built around one simple comparison:

- a normal game controlled by a human
- the same game controlled by a trained Reinforcement Learning agent

The goal is not to build Flappy Bird from scratch. The goal is to start from an existing game environment and add a DQN agent on top of it, then use that result as both a demo and a guided workshop example.

## What is inside

Core files used in the current teaching plan:

- `manual_play.py`: play the game yourself with the keyboard
- `play.py`: load a prepared model and watch the AI play
- `train.py`: training and graph-generation flow
- `agent.py`: DQN model and agent logic
- `check_env.py`: quick environment check
- `requirements.txt`: project dependencies
- `tuning_lab.ipynb`: Colab notebook for hyperparameter tuning and graph generation
- `models/`: prepared demo checkpoints for `50`, `500`, `1000`, and `strong`
- `training_progress_50.png`: weak baseline graph
- `training_progress_500.png`: medium baseline graph
- `training_progress_3000.png`: long-training graph that shows instability clearly

## Project structure

```text
Flappy bird/
|-- agent.py
|-- train.py
|-- play.py
|-- manual_play.py
|-- check_env.py
|-- requirements.txt
|-- tuning_lab.ipynb
|-- models/
|-- training_progress_50.png
|-- training_progress_500.png
|-- training_progress_3000.png
```

Main roles:

- `manual_play.py`: human-controlled baseline
- `play.py`: AI-controlled demo playback
- `train.py`: training and graph-generation flow
- `agent.py`: DQN model, replay memory, checkpoint loading and saving
- `tuning_lab.ipynb`: Colab workspace for trying hyperparameter changes and generating graphs
- `models/`: prepared checkpoints for quick comparison during the demo

## Requirements

- Windows PowerShell or Command Prompt
- Python 3.10+ recommended

An IDE is optional. You can complete the workshop with just Python and a terminal.

## Setup

Use this section first if you want to get the workshop running as quickly as possible.

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

## Workshop goal

The classroom goal is simple:

- verify the environment works
- compare human play and AI play
- compare weaker and stronger checkpoints
- explain the core RL ideas behind the project
- use Colab to tune hyperparameters and generate comparison graphs

## Demo in class

Use this section for the live classroom presentation.
This is the local demo flow, not the tuning flow.

1. Run the game yourself:

```bash
python manual_play.py
```

Controls:

- `Space` or `Up Arrow`: flap
- `R`: restart after game over
- `Esc`: quit

2. Watch the backup AI model play.

The prepared checkpoints are there for quick comparison during the demo.
For the presentation graphs, the main baseline story is:

- `50`: weak baseline
- `500`: medium baseline
- `3000`: longer training that still becomes unstable

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
- only these four checkpoints are kept in `models/` for the public teaching repository
- `python play.py --games 1` now defaults to the strongest AI model automatically
- the first four commands let you choose the AI brain explicitly before watching
- in Quick demo mode, you do not need to run `train.py`
- if you already trained a weak model and overwrote files locally, download the latest ZIP from GitHub again to restore the backup models

## Hyperparameter tuning in Colab

Use this section when you want to experiment with hyperparameters and generate training-progress graphs for comparison.

Important separation:

- local PowerShell / CMD is used for the live class demo
- Google Colab is used for tuning experiments and graph generation

Recommended flow:

1. Download this repository as ZIP from GitHub.
2. Open Google Colab.
3. Upload the full project ZIP, not only `tuning_lab.ipynb`.
4. Open `tuning_lab.ipynb` in Colab.
5. Run the notebook cells from top to bottom:
   - extract the ZIP if needed
   - set `PROJECT_ROOT`
   - verify `train.py`, `agent.py`, and `requirements.txt`
   - install dependencies with `pip install -r requirements.txt`
   - run tuning experiments

What the notebook does:

- calls `train.py` with the hyperparameters you choose
- saves graphs to `notebook_outputs/`
- shows the generated graph inside Colab
- stores temporary checkpoints in `models/notebook_temp/` only for the duration of the experiment

Good classroom use cases:

- compare multiple `learning_rate` values with the same episode count
- compare different `epsilon_decay` values
- compare reward shaping on versus off
- compare baseline runs against tuned runs

Important note:

- the notebook is a tuning workspace, not a standalone trainer
- if you upload only the notebook without the rest of the project, it will fail because `train.py` and `agent.py` are required
- graphs or temporary checkpoints created during tuning should stay in ignored local folders, not in the main repository root

## Classroom flow

Recommended classroom sequence:

1. `python check_env.py`
2. `python manual_play.py`
3. `python play.py --model .\models\dqn_flappy_50_best.pth --games 1`
4. `python play.py --model .\models\dqn_flappy_500_best.pth --games 1`
5. show `training_progress_50.png`
6. show `training_progress_500.png`
7. show `training_progress_3000.png`
8. explain the difference between `manual_play.py` and `play.py`
9. explain why more episodes alone are not enough
10. move to Google Colab and open `tuning_lab.ipynb`
11. upload the full project ZIP to Colab
12. run the setup cells in `tuning_lab.ipynb`
13. change one hyperparameter at a time
14. compare the generated training-progress graphs

## Training notes and model comparison

This repository includes multiple checkpoints so the class can compare different behaviors immediately before moving into tuning experiments.

- `50` is a weak comparison point
- `500` is a medium comparison point
- `3000` is the most useful long-training graph for showing that more episodes do not automatically mean more stability
- extra strong or tuned checkpoints can still be used as supporting examples during discussion, but they are not the main three-graph baseline story

This does not guarantee a perfect run every time. Flappy Bird still contains variation, and DQN is still an approximate method. The practical goal is to reduce early failures and make the demo behavior more reliable.

## Colab tuning workflow

Use `tuning_lab.ipynb` when the goal is to generate graphs, not to prepare final demo checkpoints.

Recommended tuning mindset:

1. start from one baseline configuration
2. change one hyperparameter at a time
3. keep the episode count the same across runs
4. save each generated graph with a clear label
5. compare the resulting curves side by side

Good first experiments:

- learning rate comparison
- epsilon decay comparison
- reward shaping on versus off
- replay buffer size comparison
- discount factor comparison

Why Colab is used here:

- students can edit values quickly
- the graphs appear in the notebook immediately
- this is more convenient for parameter exploration than re-running local demo commands

## Graphs used in slides

The main three graphs for the presentation are:

- `training_progress_50.png`
- `training_progress_500.png`
- `training_progress_3000.png`

These three are used to tell the simplest classroom story:

- weak learning
- partial learning
- longer training with visible instability

Other graphs such as tuned or stable runs should be treated as supporting material, not the main baseline set.

## Repository cleanup policy

To keep the teaching repository simple, the public version keeps only:

- the code needed to run the demo
- the Colab notebook used for tuning
- four prepared demo checkpoints in `models/`
- the three baseline graphs used in slides

Older graphs, extra tuning runs, temporary checkpoints, and draft workshop files should stay outside the public repo or inside ignored local folders.

## Troubleshooting

If `play.py` says the model is missing, or the backup model was accidentally overwritten, download the latest ZIP from GitHub again.

If a game window does not open, verify the environment first:

```bash
python check_env.py
```
