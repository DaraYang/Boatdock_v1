import os
import click
import time
import gym
import gym_platform
from gym.wrappers import Monitor
# from common import ClickPythonLiteralOption
# from common.platform_domain import PlatformFlattenedActionWrapper
import argparse
import numpy as np
import torch
# from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
# from agents.hppo_noshare import PPO
# from agents.utils.ppo_utils import ReplayBufferPPO
from pamdp_env import boatEnv
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from sklearn.utils import shuffle
from torch.distributions import Categorical



class Actor(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim, max_action, ):
        super(Actor, self).__init__()

        self.l1_1 = nn.Linear(state_dim, 256)
        self.l1_2 = nn.Linear(state_dim, 256)

        self.l2_1 = nn.Linear(256, 256)
        self.l2_2 = nn.Linear(256, 256)

        self.l3_1 = nn.Linear(256, discrete_action_dim)
        self.l3_2 = nn.Linear(256, parameter_action_dim)

        self.max_action = max_action
        self.log_std = nn.Parameter(torch.zeros([10, parameter_action_dim]).view(-1, parameter_action_dim))

    def forward(self, x):
        # 共享部分
        x_1 = F.relu(self.l1_1(x))
        x_2 = F.relu(self.l1_2(x))

        x_1 = F.relu(self.l2_1(x_1))
        x_2 = F.relu(self.l2_2(x_2))

        # 离散
        discrete_prob = F.softmax(self.l3_1(x_1),dim=1)
        # 连续
        mu = torch.tanh(self.l3_2(x_2))
        log_std = self.log_std.sum(dim=0).view(1, -1) - 0.5
        std = torch.exp(log_std)

        return discrete_prob, mu, std, log_std


mytensor = torch.tensor([
    [0.5780, 0.5750, 0.5660, 0.4580, 0.6400, 0.5120, 0.1200, 0.0000, 0.0010,
     0.5780,0.575]
])