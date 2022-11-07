import time
from abc import ABC, abstractmethod
from typing import Union, Dict
from transitions import State
from transitions.extensions.nesting import NestedState
# from transitions.core import CallbacksArg
from skillest.analysis.distance import Distance, EuclideanDistance


class Rule(State, ABC):
    def __init__(self,
                 name: str,
                 key_data: Dict,
                 distance: Distance, 
                 key_deviation: Dict=None,
                 duration=None,
                 duration_deviation=None,
                 on_enter = None,
                 on_exit = None,
                 ignore_invalid_triggers: bool = None) -> None:
        super().__init__(name, on_enter, on_exit, ignore_invalid_triggers)
        self.key_data = key_data
        self.distance = distance
        self.key_deviation = key_deviation
        self.duration = duration
        self.duration_deviation = duration_deviation
        
        self.parent = None
        self.child = None
    
    def set_parent(self, rule):
        self.parent = rule
    
    def set_child(self, rule):
        self.child = rule


class DynamicRule(Rule):

    pass


class StaticRule(Rule):

    def __init__(self,
                 name: str,
                 key_data: Dict,
                 distance: Distance,
                 key_deviation: Dict=None,
                 duration=None,
                 duration_deviation=None,
                 on_enter = None,
                 on_exit = None,
                 ignore_invalid_triggers: bool = None) -> None:
        super().__init__(name, key_data, distance, key_deviation, duration, duration_deviation, on_enter, on_exit, ignore_invalid_triggers)

    def compare(self, actual: Dict) -> Dict:
        return self.distance(actual, self.key_data)
    
    def is_valid(self, actual):
        distance = self.compare(actual)
        valid = {}
        for key in self.key_deviation.keys():
            if distance[key] == None:
                valid[key] = False
            elif key != "all":
                # print(f"{self.key_deviation[key][0]} < {distance[key]} < {self.key_deviation[key][1]}")
                valid[key] = self.key_deviation[key][0] < distance[key] < self.key_deviation[key][1]
    
        if self.duration is not None and self.duration_deviation is not None:
            curr_time = time.time()
            if self.start_time is not None:
                self.start_time = curr_time 
            valid["duration"] =  curr_time - self.start_time < self.duration_deviation
            if not valid["duration"]:
                self.end_time = curr_time
        # print(valid)
        return all(valid.values()), valid


class Uncertain(StaticRule):

    def __init__(self,
                 distance: Distance, 
                 name="uncertain",
                 key_deviation: Dict=None,
                 duration=None,
                 duration_deviation=None,
                 on_enter = None,
                 on_exit = None,
                 ignore_invalid_triggers: bool = None):
        super().__init__(name,
                         None,
                         distance,
                         key_deviation,
                         duration,
                         duration_deviation,
                         on_enter,
                         on_exit,
                         ignore_invalid_triggers)
    
    def set_key_pose(self, pose):
        self.key_data = pose


if __name__ == "__main__":
    from skillest.fsm.model import Model
    from transitions import Machine, State
    from transitions.extensions import HierarchicalMachine
    initial_pose = {"left_shoulder": 0, "right_shoulder": 0}
    initial_pose_deviation = {"left_shoulder": [-30, 30], "right_shoulder": [-30, 30]}
    initial = StaticRule("init", key_data=initial_pose,
                       distance=EuclideanDistance(),
                       key_deviation=initial_pose_deviation)
    
    uncertain_pose_deviation = {"left_shoulder": [-20, 10], "right_shoulder": [-20, 20]}
    uncertain = Uncertain(EuclideanDistance(),
                          key_deviation=uncertain_pose_deviation,)

    approaching_pose_deviation = {"left_shoulder": [-20, 20], "right_shoulder": [-20, 20]}
    approaching = Uncertain(EuclideanDistance(),
                            name="approaching",
                            key_deviation=approaching_pose_deviation)
    
    states = [initial]
    initial.add_substates([uncertain, approaching])
    transitions = [{"trigger": "un", "source": "init", "dest": "init_uncertain"},
                   {"trigger": "ap", "source": "init_uncertain", "dest": "init"}]

    activity = Model({name: state for name, state in zip([state.name for state in states], states)})
    m = HierarchicalMachine(model=activity, states=states, transitions=transitions, initial="init", send_event=True)

    print(m.states)
    activity.un()
    print(activity.state)
    activity.ap()
    print(activity.state)
