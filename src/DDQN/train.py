import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from config import *
from utils import *
from src.DDQN.model import DuelingDQN, WeightedReplayBuffer
from config import EPOCHS_AGENT, BATCH_SIZE_RL_AGENT, LR_GENERATOR, GAMMA, DEVICE, AGENT_PATH

def create_lr_scheduler(optimizer, num_train_steps, warmup_steps):
    
    
    # Scheduler for the linear warmup phase
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    
    # Scheduler for the cosine decay phase
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(num_train_steps - warmup_steps)
    )
    
    # Chain them together
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps]
    )
    return lr_scheduler




def train_agent(p_model,t_model, env, smiles_list, batch_size=64, lr=1e-4,gamma=0.99):
    
    save_path = AGENT_PATH
    device = DEVICE

    policy_net = p_model.to(device)
    target_net = t_model.to(device)

    target_net.load_state_dict(policy_net.state_dict())
    
    #steps_per_epoch = len(smiles_list)
    #total_steps = EPOCHS_AGENT * steps_per_epoch
    #warmup_steps = int(total_steps * 0.20) 
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = WeightedReplayBuffer(10000)
    

    #scheduler = create_lr_scheduler(optimizer, total_steps, warmup_steps)
    
    epsilon, epsilon_decay, tau = 1.0, 0.996, 0.005 
    best_avg_reward = -float('inf')
    reward_history = []

    for ep in range(EPOCHS_AGENT):
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
                torch.save(policy_net.state_dict(), save_path)
                print(f"New Best Agent Saved! Reward: {avg_rew:.2f}")
            print(f"Ep {ep} | Avg(50): {avg_rew:.2f} | Eps: {epsilon:.2f} | Dir: {info.get('direction', '')}")
            
    return policy_net