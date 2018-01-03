# GAFT
from gaft import GAEngine
# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore
from gaft.components import binary_individual
from gaft.components import population
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
import random
import protoss_units

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


class SimpleAgent(base_agent.BaseAgent):
    build_queue = []
    indv = None
    building_unit = None
    building_selected = False
    total_reward = 0

    def __init__(self):
        super(SimpleAgent, self).__init__()
        # self.indv = indv

    def set_up_build_queue(self):
        self.build_queue.clear()
        self.build_queue = protoss_units.get_building_queue(self.indv.solution)
        random.shuffle(self.build_queue)

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)
        self.build_queue.clear()

    def reset(self):
        super().reset()
        self.build_queue.clear()
        self.indv = global_indv
        self.building_unit = None
        self.building_selected = None
        self.total_reward = 0

    def step(self, obs):
        super(SimpleAgent, self).step(obs)
        self.total_reward += obs.reward

        if obs.observation['player'][_MINERALS_ID] >= 3500:
            self.set_up_build_queue()
        if len(self.build_queue) != 0 and self.building_unit is None:
            unit = self.build_queue.pop()
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
train_map.directory = 'D:\\StarcraftAI\\Maps'
train_map.filename = 'Train'
env = sc2_env.SC2Env(
            map_name=train_map,
            visualize=False,
            agent_race='P',
            score_index=0,
            game_steps_per_episode=500,
            difficulty=8
)
global_indv = None
agent = SimpleAgent()
def test(indv):
    global global_indv
    global_indv = indv
    run([agent], env)
    return agent.total_reward


army_vector = []
for i in range(0, 16):
    army_vector.append((0, 5))
indv_template = binary_individual.BinaryIndividual(ranges=army_vector, eps=1.0)
population = population.Population(indv_template=indv_template, size=60).init()
# Use built-in operators here.
selection = RouletteWheelSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[FitnessStore])


@engine.fitness_register
def fitness(indv):
    building_queue = protoss_units.get_building_queue(indv.solution)
    fit = float(test(indv))
    for unit in building_queue[0:7]:
        fit -= unit.time / 8
    for unit in building_queue[7:12]:
        fit -= unit.time / 2
    for unit in building_queue[12:17]:
        fit -= unit.time
    print('fit :{fit}'.format(fit=fit))
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
        x = best_indv.solution
        y = engine.fitness(best_indv)
        msg = 'Optimal solution: ({}, {})'.format(x, y)
        self.logger.info(msg)


if __name__ == '__main__':
    engine.run(ng=20)
