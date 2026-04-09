# AI Flappy Bird Workshop

This workshop follows the plan we discussed: use an existing Flappy Bird environment, then add Reinforcement Learning on top with a DQN agent. The scope is designed for a 2-hour guided session, not a full from-scratch game build.

## Workshop flow

1. Confirm the environment works.
2. Let participants fill in `skeleton/agent_skeleton.py`.
3. Let participants complete `skeleton/train_skeleton.py`.
4. If time runs short, switch to the completed reference files `agent.py`, `train.py`, and `play.py`.
5. Show a trained or partially trained model playing the game.

## Files

- `agent.py`: complete DQN agent reference
- `train.py`: complete training loop reference
- `play.py`: load a saved model and watch it play
- `check_env.py`: quick environment check
- `skeleton/agent_skeleton.py`: exercise version with TODOs
- `skeleton/train_skeleton.py`: exercise version with TODOs

## Setup

```bash
pip install -r requirements.txt
python check_env.py
```

`check_env.py` is the safest quick test because it does not depend on the `flappy_bird_gymnasium` executable being on PATH.

## Guided workshop commands

Participant version:

```bash
python skeleton/train_skeleton.py
```

Reference version:

```bash
python train.py --episodes 50
python play.py --games 3
```

## Suggested 2-hour timeline

- 0:00-0:15 setup and environment check
- 0:15-0:35 explain state, action, reward, and DQN idea
- 0:35-1:05 complete `agent_skeleton.py`
- 1:05-1:35 complete `train_skeleton.py`
- 1:35-1:50 run short training and inspect reward logs
- 1:50-2:00 run `play.py` and discuss results

## Notes for teaching

- Early rewards can be negative. That is normal.
- `train.py` now saves the best checkpoint even when the first few episodes are bad.
- `play.py` can fall back to `models/dqn_flappy_final.pth` if `models/dqn_flappy_best.pth` is missing.
- If training is too slow in class, use the reference files and a pre-trained checkpoint as Plan B.

## Useful commands

```bash
python train.py --episodes 200
python train.py --episodes 50 --render
python play.py --games 1
python play.py --games 1 --no-render
```
