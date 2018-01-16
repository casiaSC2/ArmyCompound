from pysc2.lib import actions
from .units import Terran
from .protoss_units import Unit
Marine = Unit('marine', build_id=Terran.Barracks, train_id=actions.FUNCTIONS.Train_Marine_quick.id, unit_id=Terran.Marine, minerals=50, gas=0, time=25)