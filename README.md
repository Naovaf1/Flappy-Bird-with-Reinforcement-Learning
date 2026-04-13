# Flappy Bird with Reinforcement Learning

This repository is a workshop-friendly Flappy Bird project built around one simple comparison:

- a normal game controlled by a human
- the same game controlled by a trained Reinforcement Learning agent

The goal is not to build Flappy Bird from scratch. The goal is to start from an existing game environment and add a DQN agent on top of it, then use that result as both a demo and a guided workshop example.

## What is inside

- `manual_play.py`: play the game yourself with the keyboard
- `train.py`: train a DQN agent
- `play.py`: load a trained model and watch the AI play
- `check_env.py`: quick environment check
- `agent.py`: DQN model and agent logic
- `tuning_lab.ipynb`: Colab notebook for hyperparameter tuning and graph generation
- `models/dqn_flappy_50_best.pth`: weak demo model
- `models/dqn_flappy_500_best.pth`: medium demo model
- `models/dqn_flappy_1000_best.pth`: strong score-base demo model
- `models/dqn_flappy_strong_best.pth`: tuned stability-first demo model
- `compare_checkpoints.py`: deterministic benchmark helper for workshop comparison slides
- `workshop_benchmark_summary.md`: slide-ready benchmark summary with speaker notes

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
```

Main roles:

- `manual_play.py`: human-controlled baseline
- `play.py`: AI-controlled demo playback
- `train.py`: training and evaluation flow
- `agent.py`: DQN model, replay memory, checkpoint loading and saving
- `tuning_lab.ipynb`: Colab workspace for trying hyperparameter changes and generating graphs
- `models/`: prepared checkpoints for quick comparison
- `old_graphs/`: archived training plots from previous experiments
- `compare_checkpoints.py`: evaluate checkpoints with the same seeds and export a slide-ready summary
- `workshop_benchmark_summary.md`: ready-to-copy presentation summary based on benchmark output

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

## Colab tuning lab

Use this path when you want to experiment with hyperparameters and generate training-progress graphs for comparison.

Important separation:

- local PowerShell / CMD is the recommended path for the live class demo
- Google Colab is the recommended path for tuning experiments and graph generation

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

## Classroom flow

Recommended classroom sequence:

1. `python check_env.py`
2. `python manual_play.py`
3. `python play.py --model .\models\dqn_flappy_50_best.pth --games 1`
4. `python play.py --model .\models\dqn_flappy_500_best.pth --games 1`
5. `python play.py --model .\models\dqn_flappy_1000_best.pth --games 1`
6. `python play.py --model .\models\dqn_flappy_strong_best.pth --games 1`
7. explain the difference between `manual_play.py` and `play.py`
8. move to Google Colab and open `tuning_lab.ipynb`
9. upload the full project ZIP to Colab
10. run the setup cells in `tuning_lab.ipynb`
11. change one hyperparameter at a time
12. compare the generated training-progress graphs

## Training notes and model comparison

This repository includes multiple checkpoints so the class can compare different behaviors immediately before moving into tuning experiments.

- `50` is a weak comparison point
- `500` is a medium comparison point
- `1000` is a strong score-oriented model
- `Strong` is a tuned follow-up model that prioritizes stability more than peak score

The most important classroom comparison is `1000` versus `Strong`:

- `dqn_flappy_1000_best.pth` is the higher-risk, higher-peak score-base model
- `dqn_flappy_strong_best.pth` is a separate tuned model that focuses more on stability
- compared with `1000`, the tuned `Strong` version adds Double DQN, Huber loss, a lower learning rate, a larger batch size, a larger replay buffer, gradient clipping, and multi-game evaluation for checkpoint selection

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

Why Colab is used here:

- students can edit values quickly
- the graphs appear in the notebook immediately
- this is more convenient for parameter exploration than re-running local demo commands

## Benchmark for slides

If you want reproducible numbers for PowerPoint instead of eyeballing only the plots, run:

```bash
python compare_checkpoints.py --games 100 --csv-out notes\checkpoint_benchmark.csv --markdown-out notes\checkpoint_benchmark.md
```

What this gives you:

- deterministic evaluation with the same seed pattern for every checkpoint
- failure rate for "dies too early" comparisons
- quartiles, median, mean, and max score for consistency analysis
- a Markdown summary you can copy into presentation notes

The default benchmark set includes the weak, medium, score-peak, stable, and candidate checkpoints that already exist in `models/`.

## Backup material

- `models/dqn_flappy_50_best.pth`: weak checkpoint for comparison
- `models/dqn_flappy_500_best.pth`: medium checkpoint for comparison
- `models/dqn_flappy_1000_best.pth`: strong score-base checkpoint for comparison
- `models/dqn_flappy_strong_best.pth`: separately tuned stability-first checkpoint for comparison
- `old_graphs/training_progress_1000.png`: sample training curve

## Troubleshooting

If `play.py` says the model is missing, or the backup model was accidentally overwritten, download the latest ZIP from GitHub again.

If a game window does not open, verify the environment first:

```bash
python check_env.py
```
