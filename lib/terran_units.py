from pysc2.lib import actions
from .units import Terran
from .protoss_units import Unit
Marine = Unit('marine', build_id=Terran.Barracks, train_id=actions.FUNCTIONS.Train_Marine_quick.id, unit_id=Terran.Marine, minerals=50, gas=0, time=25)
Marauder = Unit('marauder', build_id=Terran.Barracks, train_id=actions.FUNCTIONS.Train_Marauder_quick.id, unit_id=Terran.Marauder, minerals=100, gas=25, time=25)
Reaper = Unit('reaper', build_id=Terran.Barracks, train_id=actions.FUNCTIONS.Train_Reaper_quick.id, unit_id=Terran.Reaper, minerals=50, gas=25, time=25)
Ghosts = Unit('ghost', build_id=Terran.Barracks, train_id=actions.FUNCTIONS.Train_Ghost_quick.id, unit_id=Terran.Ghost, minerals=200, gas=100, time=25)
# todo
# = Unit('marine', build_id=Terran.Barracks, train_id=actions.FUNCTIONS.Train_Marauder_quick.id, unit_id=Terran.Marauder, minerals=100, gas=25, time=25)

