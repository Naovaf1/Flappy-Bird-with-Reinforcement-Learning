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
        # TODO: 1. สร้าง Layer ของ Neural Network (เช่น nn.Linear)
        # ตัวใบ้: state ของ Flappy Bird มีขนาด = state_size
        #         ผลลัพธ์ของ Network ควรมีขนาด = action_size (2 ค่า: โดด/ไม่โดด)
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None
    
    def forward(self, x):
        # TODO: 2. นำ Output ของแต่ละ layer มารับ Activation Function (เช่น torch.relu)
        pass

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
        
        # 📌 Hyperparameters เอาไว้ปรับเล่นตอนหลังได้
        self.gamma = 0.99          
        self.epsilon = 1.0         
        self.epsilon_min = 0.01    
        self.epsilon_decay = 0.995 
        self.learning_rate = 0.001 
        self.batch_size = 64       
        
        # Network setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Agent] Using device: {self.device}")
        
        # TODO: 3. สร้าง policy_net และ target_net จากคลาส DQN ด้านบน
        self.policy_net = None # แทนที่จุดนี้
        self.target_net = None # แทนที่จุดนี้
        
        # Copy โครงสร้าง weight ตอนเริ่มต้นให้ target_net มีค่าเท่ากับ policy_net
        if self.policy_net is not None and self.target_net is not None:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
            
        self.memory = ReplayMemory()
        self.loss_fn = nn.MSELoss()
    
    def select_action(self, state):
        """เลือก action: จะสุ่ม (Explore) หรือเลือกตาม Model (Exploit)"""
        # TODO: 4. เขียนโค้ดสุ่มแอคชันถ้าค่า random.random() < self.epsilon
        # ถ้าสุ่ม ให้ return ค่า 0 หรือ 1 แบบสุ่ม
        if random.random() < self.epsilon:
            pass # ลบ pass แล้วใส่โค้ดของคุณ
            
        # ถ้าไม่สุ่ม ให้ใช้ Neural Network (policy_net) ตัดสินใจ
        if self.policy_net is not None:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        return 0 # Fallback 
    
    def learn(self):
        """เรียนรู้จาก memory ปรับปรุง weights"""
        # เตรียมชุดข้อมูลจาก Memory (ข้ามตอนนี้ถ้ายังไม่ได้ทำข้อก่อนหน้า)
        if len(self.memory) < self.batch_size or self.policy_net is None:
            return
            
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # TODO: 5. คำนวณหาค่า loss เอาไปอัปเดต network (โค้ดส่วนนี้อาจจะลึกหน่อย ในห้องอาจจะให้ก๊อปวาง)
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
            
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ลดอัตรา exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_network(self):
        if self.policy_net is not None and self.target_net is not None:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        if self.policy_net is not None:
            torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path):
        if self.policy_net is not None and self.target_net is not None:
            self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.epsilon = 0.0 
