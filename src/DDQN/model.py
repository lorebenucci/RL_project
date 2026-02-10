import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque



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