import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from config import *
from generator_hybrid.utils import *
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU()
        )
        self.value_stream = nn.Linear(hidden_dim // 2, 1)
        self.advantage_stream = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

class WeightedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta=0.4
        self.beta_increment=0.005

    # METODO AGGIUNTO: Permette di usare len(memory)
    def __len__(self):
        return len(self.buffer)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))
        self.rewards.append(r)

    def sample(self, batch_size):
        weights = (np.abs(self.rewards) + 1e-5) ** self.alpha
        probs = weights / weights.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # 3. Importance Sampling Weights (Correzione del Bias)
        # w_i = (N * P(i)) ^ -beta
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max() # Normalizza a 1 max
        
        # Incrementa beta (annealing verso 1.0)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.buffer[i] for i in indices]
        s, a, r, ns, d = zip(*batch)
        return np.array(s), a, r, np.array(ns), d, weights

def train_agent(env, smiles_list, episodes=1000, batch_size=64, lr=1e-4,gamma=0.99,hidden_channels=1024, device='cpu', path="checkpoint.pth"):
    policy_net = DuelingDQN(LATENT_DIM, env.action_space_size,hidden_dim=hidden_channels).to(device)
    target_net = DuelingDQN(LATENT_DIM, env.action_space_size,hidden_dim=hidden_channels).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    #steps_per_epoch = len(smiles_list)
    #total_steps = EPOCHS_AGENT * steps_per_epoch
    #warmup_steps = int(total_steps * 0.20) 
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = WeightedReplayBuffer(10000)
    
    #scheduler
    #scheduler = create_lr_scheduler(optimizer, total_steps, warmup_steps)
    
    epsilon, epsilon_decay, tau = 1.0, 0.996, 0.005 
    best_avg_reward = -float('inf')
    reward_history = []

    for ep in range(episodes):
        state = env.reset(random.choice(smiles_list))
        total_reward, done = 0, False
        
        while not done:
            if random.random() < epsilon: 
                action = random.randint(0, env.action_space_size - 1)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_t).argmax(1).item()
            
            next_state, reward, done, info = env.step(action)
            memory.push(state, action, reward, next_state, done)
            
            # Oversampling per successi (MEG logic)
           # if done and info.get('flipped', False):
            #    for _ in range(5): memory.push(state, action, reward, next_state, done)
            
            state, total_reward = next_state, total_reward + reward

            if len(memory) > batch_size:
                s, a, r, ns, d, weights = memory.sample(batch_size)
                s_t, ns_t = torch.FloatTensor(s).to(device), torch.FloatTensor(ns).to(device)
                a_t = torch.LongTensor(a).view(-1, 1).to(device)
                r_t = torch.FloatTensor(r).view(-1, 1).to(device)
                d_t = torch.FloatTensor(d).view(-1, 1).to(device)
                weights_tensor = torch.FloatTensor(weights).to(device).unsqueeze(1)
                
                # Double DQN logic
                curr_Q = policy_net(s_t).gather(1, a_t)
                with torch.no_grad():
                    next_a = policy_net(ns_t).argmax(1, keepdim=True)
                    next_Q = target_net(ns_t).gather(1, next_a)
                    expected_Q = r_t + (1 - d_t) * gamma * next_Q

                loss = F.smooth_l1_loss(curr_Q, expected_Q)
                loss = (loss * weights_tensor).mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                
                #add scheduler
                #scheduler.step()

                # Soft update Target Net
                for tp, lp in zip(target_net.parameters(), policy_net.parameters()):
                    tp.data.copy_(tau * lp.data + (1.0 - tau) * tp.data)

        epsilon = max(0.01, epsilon * epsilon_decay)
        reward_history.append(total_reward)

        if ep % 10 == 0:
            avg_rew = np.mean(reward_history[-50:]) if len(reward_history) > 0 else 0
            if ep > 100 and avg_rew > best_avg_reward:
                best_avg_reward = avg_rew
                torch.save(policy_net.state_dict(), path)
                print(f"New Best Agent Saved! Reward: {avg_rew:.2f}")
            print(f"Ep {ep} | Avg(50): {avg_rew:.2f} | Eps: {epsilon:.2f} | Dir: {info.get('direction', '')}")
            
    return policy_net