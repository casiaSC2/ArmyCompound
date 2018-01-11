from pysc2.lib import actions
import random

class Unit:
    def __init__(self, name, build_id, train_id, unit_id, minerals, gas, time):
        self.name = name
        self.build_id = build_id
        self.train_id = train_id
        self.unit_id = unit_id
        self.minerals = minerals
        self.gas = gas
        self.time = time


# Unit IDs
PROTOSS_ADEPT = 311
PROTOSS_ADEPTPHASESHIFT = 801
PROTOSS_ARCHON = 141
PROTOSS_ASSIMILATOR = 61
PROTOSS_CARRIER = 79
PROTOSS_COLOSSUS = 4
PROTOSS_CYBERNETICSCORE = 72
PROTOSS_DARKSHRINE = 69
PROTOSS_DARKTEMPLAR = 76
PROTOSS_DISRUPTOR = 694
PROTOSS_DISRUPTORPHASED = 733
PROTOSS_FLEETBEACON = 64
PROTOSS_FORGE = 63
PROTOSS_GATEWAY = 62
PROTOSS_HIGHTEMPLAR = 75
PROTOSS_IMMORTAL = 83
PROTOSS_INTERCEPTOR = 85
PROTOSS_MOTHERSHIP = 10  # ,    // SMART, MOVE, PATROL, HOLDPOSITION, EFFECT_PHOTONOVERCHARGE, EFFECT_TIMEWARP, STOP, ATTACK, EFFECT_MASSRECALL
PROTOSS_MOTHERSHIPCORE = 488  # ,   // SMART, MOVE, PATROL, HOLDPOSITION, MORPH_MOTHERSHIP, EFFECT_PHOTONOVERCHARGE, EFFECT_TIMEWARP, CANCEL, STOP, ATTACK, EFFECT_MASSRECALL
PROTOSS_NEXUS = 59  # ,    // SMART, EFFECT_CHRONOBOOST, TRAIN_PROBE, TRAIN_MOTHERSHIPCORE, CANCEL, CANCEL_LAST, RALLY_WORKERS
PROTOSS_OBSERVER = 82  # ,    // SMART, MOVE, PATROL, HOLDPOSITION, STOP, ATTACK
PROTOSS_ORACLE = 495  # ,   // SMART, MOVE, PATROL, HOLDPOSITION, EFFECT_ORACLEREVELATION, BEHAVIOR_PULSARBEAMON, BEHAVIOR_PULSARBEAMOFF, BUILD_STASISTRAP, CANCEL, STOP, ATTACK
PROTOSS_ORACLESTASISTRAP = 732  # ,   // CANCEL
PROTOSS_PHOENIX = 78  # ,    // SMART, MOVE, PATROL, HOLDPOSITION, EFFECT_GRAVITONBEAM, CANCEL, STOP, ATTACK
PROTOSS_PHOTONCANNON = 66  # ,    // SMART, CANCEL, STOP, ATTACK
PROTOSS_PROBE = 84  # ,    // SMART, MOVE, PATROL, HOLDPOSITION, BUILD_NEXUS, BUILD_PYLON, BUILD_ASSIMILATOR, BUILD_GATEWAY, BUILD_FORGE, BUILD_FLEETBEACON, BUILD_TWILIGHTCOUNCIL, BUILD_PHOTONCANNON, BUILD_STARGATE, BUILD_TEMPLARARCHIVE, BUILD_DARKSHRINE, BUILD_ROBOTICSBAY, BUILD_ROBOTICSFACILITY, BUILD_CYBERNETICSCORE, STOP, HARVEST_GATHER, HARVEST_RETURN, ATTACK, EFFECT_SPRAY
PROTOSS_PYLON = 60  # ,    // CANCEL
PROTOSS_PYLONOVERCHARGED = 894  # ,   // SMART, STOP, ATTACK
PROTOSS_ROBOTICSBAY = 70  # ,    // RESEARCH_GRAVITICBOOSTER, RESEARCH_GRAVITICDRIVE, RESEARCH_EXTENDEDTHERMALLANCE, CANCEL, CANCEL_LAST
PROTOSS_ROBOTICSFACILITY = 71  # ,    // SMART, TRAIN_WARPPRISM, TRAIN_OBSERVER, TRAIN_COLOSSUS, TRAIN_IMMORTAL, TRAIN_DISRUPTOR, CANCEL, CANCEL_LAST, RALLY_UNITS
PROTOSS_SENTRY = 77  # ,    // SMART, MOVE, PATROL, HOLDPOSITION, EFFECT_GUARDIANSHIELD, HALLUCINATION_ARCHON, HALLUCINATION_COLOSSUS, HALLUCINATION_HIGHTEMPLAR, HALLUCINATION_IMMORTAL, HALLUCINATION_PHOENIX, HALLUCINATION_PROBE, HALLUCINATION_STALKER, HALLUCINATION_VOIDRAY, HALLUCINATION_WARPPRISM, HALLUCINATION_ZEALOT, EFFECT_FORCEFIELD, HALLUCINATION_ORACLE, HALLUCINATION_DISRUPTOR, HALLUCINATION_ADEPT, STOP, RALLY_UNITS, ATTACK
PROTOSS_STALKER = 74  # ,    // SMART, MOVE, PATROL, HOLDPOSITION, STOP, RALLY_UNITS, ATTACK, EFFECT_BLINK
PROTOSS_STARGATE = 67  # ,    // SMART, TRAIN_PHOENIX, TRAIN_CARRIER, TRAIN_VOIDRAY, TRAIN_ORACLE, TRAIN_TEMPEST, CANCEL, CANCEL_LAST, RALLY_UNITS
PROTOSS_TEMPEST = 496  # ,   // SMART, MOVE, PATROL, HOLDPOSITION, EFFECT_TEMPESTDISRUPTIONBLAST, CANCEL, STOP, ATTACK
PROTOSS_TEMPLARARCHIVE = 68  # ,    // RESEARCH_PSISTORM, CANCEL, CANCEL_LAST
PROTOSS_TWILIGHTCOUNCIL = 65  # ,    // RESEARCH_CHARGE, RESEARCH_BLINK, RESEARCH_ADEPTRESONATINGGLAIVES, CANCEL, CANCEL_LAST
PROTOSS_VOIDRAY = 80  # ,    // SMART, MOVE, PATROL, HOLDPOSITION, EFFECT_VOIDRAYPRISMATICALIGNMENT, STOP, ATTACK
PROTOSS_WARPGATE = 133  # // SMART, TRAINWARP_ZEALOT, TRAINWARP_STALKER, TRAINWARP_HIGHTEMPLAR, TRAINWARP_DARKTEMPLAR, TRAINWARP_SENTRY, TRAINWARP_ADEPT, MORPH_GATEWAY
PROTOSS_WARPPRISM = 81  # ,    // SMART, MOVE, PATROL, HOLDPOSITION, MORPH_WARPPRISMPHASINGMODE, STOP, LOAD, UNLOADALLAT, ATTACK
PROTOSS_WARPPRISMPHASING = 136  # ,   // SMART, MORPH_WARPPRISMTRANSPORTMODE, STOP, LOAD, UNLOADALLAT
PROTOSS_ZEALOT = 73  # ,    // SMART, MOVE, PATROL, HOLDPOSITION, EFFECT_CHARGE, STOP, RALLY_UNITS, ATTACK  # Train IDs

# train actions
_TRAIN_ZEALOT = actions.FUNCTIONS.Train_Zealot_quick.id
_TRAIN_STALKER = actions.FUNCTIONS.Train_Stalker_quick.id
_TRAIN_SENTRY = actions.FUNCTIONS.Train_Sentry_quick.id
_TRAIN_HIGHTEMPLAR = actions.FUNCTIONS.Train_HighTemplar_quick.id
_TRAIN_DARKTEMPLAR = actions.FUNCTIONS.Train_DarkTemplar_quick.id
_TRAIN_ADEPT = actions.FUNCTIONS.Train_Adept_quick.id

_TRAIN_OBSERVER = actions.FUNCTIONS.Train_Observer_quick.id
_TRAIN_IMMORTAL = actions.FUNCTIONS.Train_Immortal_quick.id
_TRAIN_WARPPRISM = actions.FUNCTIONS.Train_WarpPrism_quick.id
_TRAIN_COLOSSUS = actions.FUNCTIONS.Train_Colossus_quick.id
_TRAIN_DISRUPTOR = actions.FUNCTIONS.Train_Disruptor_quick.id

_TRAIN_PHOENIX = actions.FUNCTIONS.Train_Phoenix_quick.id
_TRAIN_VOIDRAY = actions.FUNCTIONS.Train_VoidRay_quick.id
_TRAIN_CARRIER = actions.FUNCTIONS.Train_Carrier_quick.id
_TRAIN_ORACLE = actions.FUNCTIONS.Train_Oracle_quick.id
_TRAIN_TEMPEST = actions.FUNCTIONS.Train_Tempest_quick.id

Zealot = Unit('zealot', build_id=PROTOSS_GATEWAY, train_id=_TRAIN_ZEALOT, unit_id=PROTOSS_ZEALOT, minerals=100, gas=0, time=38)
Stalker = Unit('stalker', build_id=PROTOSS_GATEWAY, train_id=_TRAIN_STALKER, unit_id=PROTOSS_STALKER, minerals=125,
               gas=50, time=42)
Sentry = Unit('sentry', build_id=PROTOSS_GATEWAY, train_id=_TRAIN_SENTRY, unit_id=PROTOSS_SENTRY, minerals=50, gas=100, time=37)
HighTemplar = Unit('highTemplar', build_id=PROTOSS_GATEWAY, train_id=_TRAIN_HIGHTEMPLAR, unit_id=PROTOSS_HIGHTEMPLAR,
                   minerals=50, gas=150, time=55)
DarkTemplar = Unit('darkTemplar', build_id=PROTOSS_GATEWAY, train_id=_TRAIN_DARKTEMPLAR, unit_id=PROTOSS_DARKTEMPLAR,
                   minerals=125, gas=125, time=55)
Adept = Unit('adept', build_id=PROTOSS_GATEWAY, train_id=_TRAIN_ADEPT, unit_id=PROTOSS_ADEPT, minerals=100, gas=25, time=38)

Observer = Unit('observer', build_id=PROTOSS_ROBOTICSFACILITY, train_id=_TRAIN_OBSERVER, unit_id=PROTOSS_OBSERVER,
                minerals=25, gas=75, time=30)
Immortal = Unit('immortal', build_id=PROTOSS_ROBOTICSFACILITY, train_id=_TRAIN_IMMORTAL, unit_id=PROTOSS_IMMORTAL,
                minerals=250, gas=100, time=55)
WarpPrism = Unit('warpPrism', build_id=PROTOSS_ROBOTICSFACILITY, train_id=_TRAIN_WARPPRISM, unit_id=PROTOSS_WARPPRISM,
                 minerals=200, gas=0, time=50)
Colossus = Unit('colossus', build_id=PROTOSS_ROBOTICSFACILITY, train_id=_TRAIN_COLOSSUS, unit_id=PROTOSS_COLOSSUS,
                minerals=300, gas=200, time=75)
Disruptor = Unit('disruptor', build_id=PROTOSS_ROBOTICSFACILITY, train_id=_TRAIN_DISRUPTOR, unit_id=PROTOSS_DISRUPTOR,
                 minerals=150, gas=150, time=50)

Phoenix = Unit('phoenix', build_id=PROTOSS_STARGATE, train_id=_TRAIN_PHOENIX, unit_id=PROTOSS_PHOENIX, minerals=150,
               gas=100, time=35)
VoidRay = Unit('voidRay', build_id=PROTOSS_STARGATE, train_id=_TRAIN_VOIDRAY, unit_id=PROTOSS_VOIDRAY, minerals=250,
               gas=150, time=60)
Carrier = Unit('carrier', build_id=PROTOSS_STARGATE, train_id=_TRAIN_CARRIER, unit_id=PROTOSS_CARRIER, minerals=350,
               gas=250, time=120)
Oracle = Unit('carrier', build_id=PROTOSS_STARGATE, train_id=_TRAIN_ORACLE, unit_id=PROTOSS_ORACLE, minerals=150,
              gas=150, time=50)
Tempest = Unit('tempest', build_id=PROTOSS_STARGATE, train_id=_TRAIN_TEMPEST, unit_id=PROTOSS_TEMPEST, minerals=300,
               gas=200, time=60)

protoss_units_array = [Zealot, Stalker, Sentry, HighTemplar, DarkTemplar, Adept, Observer, WarpPrism, Immortal, Colossus,
                       Disruptor, Phoenix, VoidRay, Oracle, Carrier, Tempest]

def get_raw_building_queue(unit_tuple):
    gateway_building_queue = []
    factory_building_queue = []
    stargate_building_queue = []
    zealot_num, stalker_num, sentry_num, highTemplar_num, darkTemplar_num, adept_num, \
    observer_num, warpPrism_num, immortal_num, colossus_num, disruptor_num, \
    phoenix_num, voidRay_num, oracle_num, carrier_num, tempest_num = unit_tuple
    print(unit_tuple)
    for i in range(0, int(zealot_num)):
        gateway_building_queue.append(Zealot)
    for i in range(0, int(stalker_num)):
        gateway_building_queue.append(Stalker)
    for i in range(0, int(sentry_num)):
        gateway_building_queue.append(Sentry)
    for i in range(0, int(highTemplar_num)):
        gateway_building_queue.append(HighTemplar)
    for i in range(0, int(darkTemplar_num)):
        gateway_building_queue.append(DarkTemplar)
    for i in range(0, int(adept_num)):
        gateway_building_queue.append(Adept)


    for i in range(0, int(observer_num)):
        factory_building_queue.append(Observer)
    for i in range(0, int(warpPrism_num)):
        factory_building_queue.append(WarpPrism)
    for i in range(0, int(immortal_num)):
        factory_building_queue.append(Immortal)
    for i in range(0, int(colossus_num)):
        factory_building_queue.append(Colossus)
    for i in range(0, int(disruptor_num)):
        factory_building_queue.append(Disruptor)

    for i in range(0, int(phoenix_num)):
        stargate_building_queue.append(Phoenix)
    for i in range(0, int(voidRay_num)):
        stargate_building_queue.append(VoidRay)
    for i in range(0, int(oracle_num)):
        stargate_building_queue.append(Oracle)
    for i in range(0, int(carrier_num)):
        stargate_building_queue.append(Carrier)
    for i in range(0, int(tempest_num)):
        stargate_building_queue.append(Tempest)

    return gateway_building_queue + factory_building_queue + stargate_building_queue


def get_building_queue(unit_tuple, total_time = 900, gateway_count = 8, factory_count = 2, stargate_count = 1):
    gateway_building_queue = []
    factory_building_queue = []
    stargate_building_queue = []
    zealot_num, stalker_num, sentry_num, highTemplar_num, darkTemplar_num, adept_num, \
    observer_num, warpPrism_num, immortal_num, colossus_num, disruptor_num, \
    phoenix_num, voidRay_num, oracle_num, carrier_num, tempest_num = unit_tuple
    print(unit_tuple)
    for i in range(0, int(zealot_num)):
        gateway_building_queue.append(Zealot)
    for i in range(0, int(stalker_num)):
        gateway_building_queue.append(Stalker)
    for i in range(0, int(sentry_num)):
        gateway_building_queue.append(Sentry)
    for i in range(0, int(highTemplar_num)):
        gateway_building_queue.append(HighTemplar)
    for i in range(0, int(darkTemplar_num)):
        gateway_building_queue.append(DarkTemplar)
    for i in range(0, int(adept_num)):
        gateway_building_queue.append(Adept)
    random.shuffle(gateway_building_queue)
    gateway_time = 0
    gateway_queue = []
    for i in gateway_building_queue:
        building_time = i.time / gateway_count
        if gateway_time + building_time > total_time:
            break
        else:
            gateway_time += building_time
            gateway_queue.append(i)

    for i in range(0, int(observer_num)):
        factory_building_queue.append(Observer)
    for i in range(0, int(warpPrism_num)):
        factory_building_queue.append(WarpPrism)
    for i in range(0, int(immortal_num)):
        factory_building_queue.append(Immortal)
    for i in range(0, int(colossus_num)):
        factory_building_queue.append(Colossus)
    for i in range(0, int(disruptor_num)):
        factory_building_queue.append(Disruptor)
    random.shuffle(factory_building_queue)
    factory_time = 0
    factory_queue = []
    for i in factory_building_queue:
        building_time = i.time / factory_count
        if factory_time + building_time > total_time:
            break
        else:
            factory_time += building_time
            factory_queue.append(i)


    for i in range(0, int(phoenix_num)):
        stargate_building_queue.append(Phoenix)
    for i in range(0, int(voidRay_num)):
        stargate_building_queue.append(VoidRay)
    for i in range(0, int(oracle_num)):
        stargate_building_queue.append(Oracle)
    for i in range(0, int(carrier_num)):
        stargate_building_queue.append(Carrier)
    for i in range(0, int(tempest_num)):
        stargate_building_queue.append(Tempest)

    random.shuffle(stargate_building_queue)
    stargate_time = 0
    stargate_queue = []
    for i in stargate_building_queue:
        building_time = i.time / stargate_count
        if stargate_time + building_time > total_time:
            break
        else:
            stargate_time += building_time
            stargate_queue.append(i)
    return gateway_queue + factory_queue + stargate_queue
