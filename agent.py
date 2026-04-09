import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    """Neural Network ที่รับ state แล้วทำนาย Q-value ของแต่ละ action"""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # State ของ Flappy Bird (Gymnasium) มี 12 ค่า
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    """เก็บประสบการณ์ไว้สำหรับ train (Experience Replay)"""
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """บันทึกประสบการณ์ใหม่ลง memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """สุ่มดึงประสบการณ์ตาม batch size เพื่อนำไป train"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    """AI Agent หลักที่คอยตัดสินใจและเรียนรู้"""
    def __init__(self, state_size=12, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        
        # 📌 [TODO: ลองปรับค่า Hyperparameters พวกนี้ได้เลย!] 📌
        self.gamma = 0.99          # Discount factor (ความสำคัญของ reward ในอนาคต)
        self.epsilon = 1.0         # Exploration rate (ยิ่งสูงยิ่งสุ่มเยอะ เริ่มที่ 100%)
        self.epsilon_min = 0.01    # ค่าลดหลั่นสุดท้ายของ epsilon
        self.epsilon_decay = 0.995 # อัตราการลด exploration ในแต่ละ step
        self.learning_rate = 0.001 # ความเร็วในการเรียนรู้ (ลองเปลี่ยนเป็น 0.01, 0.0001)
        self.batch_size = 64       # ขนาดของข้อมูลที่จะสุ่มมาเรียนรู้ต่อครั้ง
        
        # Network setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Agent] Using device: {self.device}")
        
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net ไม่ได้ใช้ backprop, ปรับเป็น eval mode
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory()
        self.loss_fn = nn.MSELoss()
    
    def select_action(self, state):
        """เลือก action: จะสุ่ม (Explore) หรือเลือกตาม Model (Exploit)"""
        # อัตราการสุ่ม (Explore)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # คาดคะเนจาก Model (Exploit)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def learn(self):
        """เรียนรู้จาก memory ปรับปรุง weights"""
        # ถ้า memory ยังน้อยกว่า batch size ให้ข้ามไปก่อน
        if len(self.memory) < self.batch_size:
            return
        
        # สุ่มชุดข้อมูลจาก memory
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # แปลงเป็น Tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # คำนวณ Q-value ปัจจุบัน
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # คำนวณ Target Q-value จาก target network
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # หา Loss ระหว่าง current กับ target
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ลดอัตราการสุ่มลงเรื่อยๆ
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_network(self):
        """Sync policy weights เข้า target weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        """Save model"""
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path, epsilon=0.0):
        """Load model weights and optionally set epsilon."""
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = epsilon # ไม่สุ่มอีกต่อไป เพราะจะเล่นจริง
