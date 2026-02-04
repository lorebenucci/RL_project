import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from config import *

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(DuelingDQN, self).__init__()
        
        #feature extractor "pyramid" NN
        # we keep more amplitude to combine all bit descriptors
        self.feature_layer = nn.Sequential(
            
            nn.Linear(input_dim, hidden_dim*2),  #Step 1:
            nn.LayerNorm(hidden_dim*2), #LayerNorm is more stable than BatchNorm
            nn.SiLU(), #better approach of SiLU
            nn.Dropout(0.2), #light Dropout
            
            nn.Linear(hidden_dim*2, hidden_dim), #Step 2: Compressione a 512
            nn.LayerNorm(hidden_dim),
            nn.SiLU(), 
            nn.Dropout(0.2)
        )
        
        # Value stream (it evaluates how much a state is good )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2 , 1)
        )
        
        #Advantage Stream (it evaluates the goodness of action)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2 ),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2 , output_dim)
        )

    def forward(self, x):
        
        #features
        features = self.feature_layer(x)
        
        #values step
        values = self.value_stream(features)
        
        #advantages step 
        advantages = self.advantage_stream(features)
        
        #Dueling Logic: Q = V + (A - mean(A))
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_vals



class WeightedReplayBuffer:
    def __init__(self,capacity=10000,alpha=0.6):
        self.buffer=deque(maxlen=capacity) #normal buffer
      
        self.rewards=deque(maxlen=capacity) # Teniamo traccia solo delle reward per velocità
        self.alpha=alpha # priority (1= uniform sampling ,0=uniform )
        
    def __len__(self):    
        return len(self.buffer)
    
    def push(self,state,action,reward,next_state,done):
        transition=(state,action,reward,next_state,done)
        
        #put always into buffer ans save rewards
        self.buffer.append(transition)
        self.rewards.append(reward)
        
            
    def sample(self,batch_size):
    
        #1  compute the weigths for (Priority)
        # Use abs of reward to give importance
        # add epsilon (1e-5) to guarantee also reward=0 has a possibility of being exctact
        curr_len = len(self.buffer)
        rewards_arr = np.array(self.rewards)
        weights=(np.abs(rewards_arr) + 1e-5) ** self.alpha
        
        # Normalization to obtain probability:
        probs=weights / weights.sum()
        
        #Weighted sampling
        #choice from each batch size a sample given by weighted probabilities
        indices=np.random.choice(curr_len,batch_size,p=probs,replace=False)
        
        batch=[self.buffer[i] for i in indices]
        
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), action, reward, np.array(next_state), done
        
        
def train_agent(env,train_smiles_list, episodes=1000, batch_size=64, lr=1e-3, device='cpu'):
    
    output_dim = env.action_space_size
    
    policy_net = DuelingDQN(DIM_DESCRIPTORS, output_dim).to(device)
    target_net = DuelingDQN(DIM_DESCRIPTORS, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = WeightedReplayBuffer(MEMORY_SIZE_BUFFER,ALPHA)
    
    epsilon = EPSILON_START
    epsilon_decay = EPSILON_DECAY
    epsilon_min = EPSILON_MIN
    gamma = GAMMA
    
    tau = TAU_START  #soft update... of target
    tau_min = TAU_MIN     
    tau_decay = TAU_DECAY  
    print("Starting Dueling DQN training...")
    
    reward_history = []
    
    for ep in range(episodes):
        
        #Observer randomly a different molecule
        current_smile = random.choice(train_smiles_list)
        state = env.reset(specific_smiles=current_smile)
        #state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, output_dim - 1)
            else:
                policy_net.eval() #so we deactivate dropout
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_tensor).argmax(dim = 1).item()
                policy_net.train() 
            
                
            next_state, reward, done, info = env.step(action)
            memory.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if len(memory) > batch_size:
                
                #weighted sampling
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                #evaluate current Q from policy net
                curr_Q = policy_net(states).gather(1, actions)
                #next_Q = target_net(next_states).max(1)[0].unsqueeze(1)
                #expected_Q = rewards + (1 - dones) * gamma * next_Q

                
                with torch.no_grad():

                    next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)   # selezione azione by policy net
                    next_Q = target_net(next_states).gather(1, next_actions)            # valutazione by target net
                    expected_Q = rewards + (1 - dones) * gamma * next_Q


                #loss = F.mse_loss(curr_Q, expected_Q)
                #We use a smooth L1 LOSS more stable for higher reward
                loss = F.smooth_l1_loss(curr_Q, expected_Q)

                optimizer.zero_grad()
                loss.backward()
                
                #clip gradient (avoid explosion gradient caused by high reward)
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()
                
                # a risk....
                #SOFT UPDATE del Target Network because otherwise this can provoke numerical instability
                for target_param, local_param in zip(target_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
        #epsilon and tau decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        tau = max(tau_min, tau * tau_decay)
        
        reward_history.append(total_reward)
        
            
        if ep % 10 == 0:
            #last mean reward of 10 episodes
            avg_rew = np.mean(reward_history[-10:])
            #target_net.load_state_dict(policy_net.state_dict())
            print(f"Ep {ep} | Avg Reward: {avg_rew:.2f} | Eps: {epsilon:.2f} | Last Info: {info['direction'] if 'direction' in info else ''}")

    return policy_net