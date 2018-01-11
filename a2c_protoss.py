from itertools import count
from collections import namedtuple
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.maps.lib import Map
from torch.autograd import Variable
from torch.distributions import Categorical
from protoss_units import protoss_units_array
from absl import flags
import sys

gamma = 0.99
log_interval = 1

# Functions
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_TRAIN_ZEALOT = actions.FUNCTIONS.Train_Zealot_quick.id
_TRAIN_STALKER = actions.FUNCTIONS.Train_Stalker_quick.id
_TRAIN_DARK = actions.FUNCTIONS.Train_DarkTemplar_quick.id
_TRAIN_IMMORTAL = actions.FUNCTIONS.Train_Immortal_quick.id
_TRAIN_COLOSSUS = actions.FUNCTIONS.Train_Colossus_quick.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_PROTOSS_GATEWAY = 62
_PROTOSS_ROBOTICSFACILITY = 71
_PROTOSS_STARGATE = 67
# Parameters
_PLAYER_SELF = 1
_SUPPLY_USED = 3
_SUPPLY_MAX = 4
_NOT_QUEUED = [0]
_QUEUED = [1]

# player state id
_MINERALS_ID = 1


class Net(nn.Module):
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.affine(x)
        policy_result = self.policy(self.activation(x))
        value_result = self.value(self.activation(x))
        return self.softmax(policy_result), value_result

    def conv_layers(self, x):
        x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.maxpool3(self.relu3(self.bn2(self.conv3(x))))
        return x

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=17, out_channels=36, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=36)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=36, out_channels=36, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=36)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=36, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(num_features=36)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.affine = nn.Linear(in_features=1296, out_features=128)
        self.policy = nn.Linear(in_features=128, out_features=16)
        self.value = nn.Linear(in_features=128, out_features=1)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.saved_actions = []
        self.rewards = []


class A2CAgent(base_agent.BaseAgent):

    def __init__(self, net):
        super(A2CAgent, self).__init__()
        self.model = net
        self.building_unit = None
        self.building_selected = False
        self.total_reward = 0

    def setup(self, obs_spec, action_spec):
        '''
        setup the agent, not to setup the environment!
        :param obs_spec:
        :param action_spec:
        :return:
        '''
        super(A2CAgent, self).setup(obs_spec, action_spec)

    def reset(self):
        '''
        reset the agent, not to reset the environment!
        :return: None
        '''
        super(A2CAgent, self).reset()
        self.building_unit = None
        self.building_selected = None
        self.total_reward = 0

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self.model(Variable(state))
        m = Categorical(probs)
        action = m.sample()
        self.model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.data[0]

    def step(self, obs):
        '''
        To decide what to do this step.
        if the the unit to build is not decided  yet, the decide function is called
        and decide which unit to build. Then, the corresponding building is selected.
        If the unit to build is decided and the building is selected,
        then return the building function to build the corresponding unit.
        :param obs: the env observation : o(s, t)
        :return: the action function : a(s, t)
        '''
        obs = obs[0]
        super(A2CAgent, self).step(obs)
        self.total_reward += obs.reward

        if self.building_unit is None:
            unit_id = self.select_action(obs.observation['screen'])
            unit = protoss_units_array[unit_id]
            self.building_unit = unit
            unit_type = obs.observation["screen"][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == unit.build_id).nonzero()

            if unit_y.any():
                target = [int(unit_x.mean()), int(unit_y.mean())]
                self.building_selected = True
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])


        elif self.building_selected:
            building_unit = self.building_unit
            self.building_unit = None
            self.building_selected = False
            if building_unit.train_id in obs.observation['available_actions']:
                return actions.FunctionCall(building_unit.train_id, [_QUEUED])
        return actions.FunctionCall(_NOOP, [])


FLAGS = flags.FLAGS
FLAGS(sys.argv)

train_map = Map()
train_map.directory = '/home/wangjian/StarCraftII/Maps'
train_map.filename = 'DRLTrain'
env = sc2_env.SC2Env(
    map_name=train_map,
    visualize=True,
    agent_race='P',
    score_index=0,
    game_steps_per_episode=500,
    difficulty=8,
    step_mul=2,
    # save_replay_episodes=1,
    # replay_dir='/home/wangjian/StarCraftII/Maps'
)

model = Net()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
agent = A2CAgent(model)
agent.setup(env.action_spec(), env.observation_spec())


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.data[0, 0]
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
    optimizer.zero_grad()
    loss = torch.cat(policy_losses).sum() + torch.cat(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = agent.step(state)
            state = env.step([action])
            model.rewards.append(state[0].reward)
            if state[0].last():
                break
        final_reward = finish_episode()
        if i_episode % log_interval == 0:
            print('Episode {}\tfinal reward: {:.2f}'.format(
                i_episode, final_reward))


if __name__ == '__main__':
    main()
