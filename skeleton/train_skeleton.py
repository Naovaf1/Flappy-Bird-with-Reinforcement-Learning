import gymnasium as gym
import flappy_bird_gymnasium
from agent_skeleton import DQNAgent # <- เปลี่ยนไปใช้ไฟล์ agent ของเพื่อนคุณ
import os
import matplotlib.pyplot as plt

def train(num_episodes=500, render=False):
    """
    Main training loop สำหรับ Flappy Bird Agent
    """
    print(f"กำลังเริ่ม Training สำหรับ {num_episodes} Episodes...")
    
    # 1. สร้าง environment 
    env_mode = "human" if render else None
    env = gym.make("FlappyBird-v0", render_mode=env_mode, use_lidar=False)
    
    # 2. สร้าง Agent
    state_size = env.observation_space.shape[0]  
    action_size = env.action_space.n             
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    scores = []
    best_score = 0
    os.makedirs("models", exist_ok=True)
    
    for episode in range(num_episodes):
        # TODO: 1. Reset Environment เพื่อเริ่มรอบใหม่ (ควรรับค่า state และ info)
        state = None # แก้ตรงนี้

        total_reward = 0
        done = False
        
        while not done:
            # TODO: 2. สั่งให้ agent เลือก action
            action = 0 # แก้โค้ดส่วนนี้เพื่อเลือก action จริงๆ
            
            # TODO: 3. นำ action ไปรัน 1 step ใน environment 
            # (จะได้รับ next_state, reward, terminated, truncated, info คืนมา)
            # แก้โค้ด 2 บรรทัดข้างล่างนี้
            next_state, reward = None, 0
            done = True 
            
            if agent.policy_net is not None:
                # TODO: 4. พอได้ reward มาแล้ว นำไปให้ agent บันทึกประสบการ์ (push ลง memory)
                # agent.memory.push(?, ?, ?, ?, ?)
                
                # TODO: 5. สั่งให้ agent เรียนรู้ (learn)
                # agent.learn()
                pass
            
            state = next_state
            total_reward += reward
        
        scores.append(total_reward)
        
        # Sync Weights ให้ Target Network 
        if episode % 10 == 0:
            agent.update_target_network()
        
        # บันทึก Model ถ้าได้คะแนนใหม่เยอะกว่าเดิม
        if total_reward > best_score and agent.policy_net is not None:
            best_score = total_reward
            agent.save("models/dqn_flappy_best.pth")
            print(f"🌟 [New Best] Episode {episode}: Score {total_reward:.1f} - Model saved!")
        
        # พิมพ์สรุปผล 
        if episode % 10 == 0:
            avg_score = sum(scores[-10:]) / min(10, len(scores))
            print(f"Episode {episode:4d}/{num_episodes} | Score: {total_reward:5.1f} | Avg(10): {avg_score:5.1f} | Epsilon: {agent.epsilon:.3f}")
    
    if agent.policy_net is not None:
        agent.save("models/dqn_flappy_final.pth")
    env.close()
    return scores

if __name__ == "__main__":
    # สามารถเปลี่ยน render=True เพื่อให้หน้าจอเกมเด้งขึ้นมาดูได้เลย
    scores = train(num_episodes=150, render=False)
