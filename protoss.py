from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.tests import utils
from absl import flags
from absl.testing import absltest
from pysc2.maps.lib import Map
import sys

# GAFT
from gaft import GAEngine
from gaft.components import GAIndividual, GAPopulation
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation

# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

# Functions
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_TRAIN_ZEALOT = actions.FUNCTIONS.Train_Zealot_quick.id
_TRAIN_STALKER = actions.FUNCTIONS.Train_Stalker_quick.id
_TRAIN_DARK = actions.FUNCTIONS.Train_DarkTemplar_quick.id
# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_PROTOSS_GATEWAY = 62

# Parameters
_PLAYER_SELF = 1
_SUPPLY_USED = 3
_SUPPLY_MAX = 4
_NOT_QUEUED = [0]
_QUEUED = [1]

# player state id
_MINERALS_ID = 1

combination_queue = []
combination_queue.append((30, 0, 0))

class SimpleAgent(base_agent.BaseAgent):
    build_queue = []
    building_unit = None
    building_selected = False
    total_reward = 0

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)

    def reset(self):
        super().reset()
        self.total_reward = 0

    def step(self, obs):
        super(SimpleAgent, self).step(obs)
        self.total_reward += obs.reward
        print(self.total_reward)

        if obs.observation['player'][_MINERALS_ID] >= 2500:
            for i in range(0, 10):
                self.build_queue.append('zealot')
                self.build_queue.append('stalker')
                self.build_queue.append('dark')
        if len(self.build_queue) != 0 and self.building_unit is None:
            unit = self.build_queue.pop()
            if unit == 'zealot' or unit == 'stalker' or unit == 'dark':
                self.building_unit = unit
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _PROTOSS_GATEWAY).nonzero()

                if unit_y.any():
                    target = [int(unit_x.mean()), int(unit_y.mean())]
                    self.building_selected = True
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

        elif self.building_selected:
            building_unit = self.building_unit
            self.building_unit = None
            self.building_selected = False
            if building_unit == 'zealot' and _TRAIN_ZEALOT in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_ZEALOT, [_QUEUED])
            elif building_unit == 'stalker' and _TRAIN_STALKER in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_STALKER, [_QUEUED])
            elif building_unit == 'dark' and _TRAIN_DARK in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_DARK, [_QUEUED])
        return actions.FunctionCall(_NOOP, [])


FLAGS = flags.FLAGS
FLAGS(sys.argv)


class TestScripted(utils.TestCase):
    def test(self):
        train_map = Map()
        train_map.directory = '/home/wangjian/StarCraftII/Maps'
        train_map.filename = 'Train'
        with sc2_env.SC2Env(
                map_name=train_map,
                visualize=True,
                agent_race='T',
                score_index=0,
                game_steps_per_episode=1000) as env:
            agent = SimpleAgent()
            run_loop.run_loop([agent], env)


if __name__ == '__main__':
    absltest.main()
