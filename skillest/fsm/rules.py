import time
from abc import ABC, abstractmethod
from typing import Union, Dict
from transitions import State
# from transitions.core import CallbacksArg
from skillest.analysis.distance import Distance


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
