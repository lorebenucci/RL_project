import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

def train_agent(env, episodes=1000, batch_size=64, lr=1e-3, device='cpu'):
    input_dim = 2048 
    output_dim = env.action_space_size
    
    policy_net = DuelingDQN(input_dim, output_dim).to(device)
    target_net = DuelingDQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer(10000)
    
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    gamma = 0.99
    
    print("Starting Dueling DQN training...")
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, output_dim - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_tensor).argmax(dim = 1).item()
            
            next_state, reward, done, info = env.step(action)
            memory.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if len(memory) > batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                curr_Q = policy_net(states).gather(1, actions)
                #next_Q = target_net(next_states).max(1)[0].unsqueeze(1)
                #expected_Q = rewards + (1 - dones) * gamma * next_Q

                with torch.no_grad():

                    next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)   # selezione azione
                    next_Q = target_net(next_states).gather(1, next_actions)            # valutazione
                    expected_Q = rewards + (1 - dones) * gamma * next_Q


                loss = F.mse_loss(curr_Q, expected_Q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if ep % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Episode {ep}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

    return policy_net