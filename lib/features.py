import numpy as np
from pysc2.lib import features
from .protoss_units import protoss_units_array
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
class Feature:

    def extract_protoss_units(self, unit_img):
        unit_count_array = []
        for unit in protoss_units_array:
            pass

    def extract_units(self, obs):
        unit_img = np.array([obs.observation['screen'][_UNIT_TYPE]])


    def extract_img(self, obs):
        return np.array([obs.observation['screen']])

    def __init__(self, obs):
        super(Feature, self).__init__()
        self.units = self.extract_units(obs)
        self.img = self.extract_img(obs)

    def get_img(self):
        return self.img
