import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def load_checkpoint(path, map_location=None):
    """Load a checkpoint and normalize both old and new checkpoint formats."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        config = payload.get("agent_config", {})
        metadata = payload.get("metadata", {})
    else:
        state_dict = payload
        config = {}
        metadata = {}

    if "fc1.weight" in state_dict:
        state_dict = {
            "network.0.weight": state_dict["fc1.weight"],
            "network.0.bias": state_dict["fc1.bias"],
            "network.2.weight": state_dict["fc2.weight"],
            "network.2.bias": state_dict["fc2.bias"],
            "network.4.weight": state_dict["fc3.weight"],
            "network.4.bias": state_dict["fc3.bias"],
        }
        config = {**config, "hidden_sizes": [64, 64]}

    return state_dict, config, metadata


class DQN(nn.Module):
    """Simple configurable feed-forward DQN."""

    def __init__(self, state_size, action_size, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        input_size = state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ReplayMemory:
    """Experience replay buffer."""

    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """Configurable DQN agent used for training and playback."""

    def __init__(
        self,
        state_size=12,
        action_size=2,
        hidden_sizes=(64, 64),
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.001,
        batch_size=64,
        memory_capacity=10000,
        use_double_dqn=False,
        loss_type="mse",
        grad_clip=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = tuple(hidden_sizes)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        self.loss_type = loss_type
        self.grad_clip = grad_clip

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Agent] Using device: {self.device}")

        self.policy_net = DQN(state_size, action_size, hidden_sizes=self.hidden_sizes).to(self.device)
        self.target_net = DQN(state_size, action_size, hidden_sizes=self.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(capacity=memory_capacity)
        self.loss_fn = nn.SmoothL1Loss() if loss_type == "huber" else nn.MSELoss()

    def get_config(self):
        return {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "hidden_sizes": list(self.hidden_sizes),
            "gamma": self.gamma,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "use_double_dqn": self.use_double_dqn,
            "loss_type": self.loss_type,
            "grad_clip": self.grad_clip,
        }

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path, metadata=None):
        checkpoint = {
            "state_dict": self.policy_net.state_dict(),
            "agent_config": self.get_config(),
            "metadata": metadata or {},
        }
        torch.save(checkpoint, path)

    def load(self, path, epsilon=0.0):
        state_dict, _, _ = load_checkpoint(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = epsilon
