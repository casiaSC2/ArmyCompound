# GAFT
from gaft import GAEngine
# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore
from gaft.components import GAIndividual
from gaft.components import GAPopulation
from gaft.operators import FlipBitMutation
from gaft.operators import RouletteWheelSelection
from gaft.operators import UniformCrossover
# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
# Analysis plugin base class.
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.maps.lib import Map
from absl import flags
from run_loop import run
import sys
# Analysis plugin base class.

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


class SimpleAgent(base_agent.BaseAgent):
    build_queue = []
    indv = None
    building_unit = None
    building_selected = False
    total_reward = 0

    def __init__(self, indv):
        super(SimpleAgent, self).__init__()
        self.indv = indv

    def set_up_build_queue(self):
        zealot_num, stalker_num, dark_num = self.indv.variants
        print('zealot num: {zealot}, stalker num:{stalker} dark num:{dark}'.format(
            zealot = zealot_num,
            stalker = stalker_num,
            dark = dark_num
        ))
        for i in range(0, int(zealot_num)):
            self.build_queue.append('zealot')
        for i in range(0, int(stalker_num)):
            self.build_queue.append('stalker')
        for i in range(0, int(dark_num)):
            self.build_queue.append('dark')

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)
        self.build_queue.clear()

    def reset(self):
        super().reset()
        self.build_queue.clear()

    def step(self, obs):
        super(SimpleAgent, self).step(obs)
        self.total_reward += obs.reward

        if obs.observation['player'][_MINERALS_ID] >= 2500:
            self.set_up_build_queue()
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


def test(indv):
    train_map = Map()
    train_map.directory = '/home/wangjian/StarCraftII/Maps'
    train_map.filename = 'Train'
    with sc2_env.SC2Env(
            map_name=train_map,
            visualize=True,
            agent_race='T',
            score_index=0,
            game_steps_per_episode=1000) as env:
        agent = SimpleAgent(indv)
        run([agent], env)
        return agent.total_reward


indv_template = GAIndividual(ranges=[(0, 10), (0, 10), (0, 10)], encoding='binary', eps=1.0)
population = GAPopulation(indv_template=indv_template, size=10).init()
# Use built-in operators here.
selection = RouletteWheelSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[FitnessStore])


@engine.fitness_register
def fitness(indv):
    zealot_num, stalker_num, dark_num = indv.variants
    fit = float(test(indv))
    fit = fit - 100 * zealot_num - 200 * stalker_num - 250 * dark_num
    print('fit :{fit}'.format(fit = fit))
    return fit


# Define on-the-fly analysis.
@engine.analysis_register
class ConsoleOutputAnalysis(OnTheFlyAnalysis):
    interval = 1
    master_only = True

    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        msg = 'Generation: {}, best fitness: {:.3f}'.format(g, engine.fitness(best_indv))
        self.logger.info(msg)

    def finalize(self, population, engine):
        best_indv = population.best_indv(engine.fitness)
        x = best_indv.variants
        y = engine.fitness(best_indv)
        msg = 'Optimal solution: ({}, {})'.format(x, y)
        self.logger.info(msg)


if __name__ == '__main__':
    engine.run(ng=4)
