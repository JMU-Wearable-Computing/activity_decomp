from tokenize import Double
from typing import Dict
from transitions import Machine, State
from skillest.fsm.rules import Rule, StaticRule, Uncertain
from skillest.analysis.distance import EuclideanDistance


class Model():

    def __init__(self, states: Dict[str, StaticRule], init_frames=30, approaching_frames=2) -> None:
        self.states = states
        self.rules = {name: state for name, state in self.states.items()
                     if (isinstance(state, Rule)
                         and state.name != "approaching")}
        self.num_init_frames = init_frames
        self.approaching_frames = approaching_frames
        self.approaching = ""

        self.data = []
        self.seg_points = []
        self.state = None
        self.init_frames = 0
        self.approaching_history = []
    
    def _get_parameter(self, event, key="pose", arg_idx=0):
        if "pose" in event.kwargs:
            return event.kwargs["pose"]
        if len(event.args) != 0:
            return event.args[arg_idx]
        raise TypeError("Pose not passed as parameter")

    def rule_transition(self, seg_point, event):
        self.seg_points.append(seg_point)
    
    def add_data(self, data, event):
        self.data.extend(list(data))
    
    def left_curr_state(self, event):
        pose = self._get_parameter(event, "pose")
        return not self.rules[self.state].is_valid(pose)[0]
    
    def in_valid_state(self, event):
        pose = self._get_parameter(event, "pose")
        for name, rule in self.rules.items():
            if name == self.state or name == "uncertain":
                continue # Skip the current state
            if rule.is_valid(pose)[0]:
                return True
        return False
    
    def set_uncertain_pose(self, event):
        pose = self._get_parameter(event, "pose")
        uncertain: Uncertain = self.states["uncertain"]
        uncertain.set_key_pose(pose)
    
    def set_approaching_pose(self, event):
        self.states["approaching"].key_data = self.states[event.transition.source].key_data
    
    def find_approaching(self, event):
        pose = self._get_parameter(event, "pose")
        self.approaching_history.append(pose)
        print(self.approaching_history)
        if len(self.approaching_history) >= self.approaching_frames:
            best = -1e10
            best_rule = None
            for name, rule in self.rules.items():
                if name == "uncertain" or name == "init":
                    continue # Skip the uncertain state
                compare_first = abs(rule.compare(self.approaching_history[0])["all"])
                compare_last = abs(rule.compare(self.approaching_history[-1])["all"])
                diff = compare_first - compare_last
                if diff > best:
                    best = diff
                    best_rule = rule
                print(diff)
            self.approaching = best_rule.name
            self.approaching_history.clear()
    
    def clear_approaching(self, event):
        self.approaching = ""
    
    def in_dest_pose(self, event):
        pose = self._get_parameter(event, "pose")
        return self.rules[event.transition.dest].is_valid(pose)[0]
    
    def is_approaching_dest_parent_pose(self, event):
        print(f"approaching {self.approaching}")
        rules = list(self.rules.values())
        idx = None
        for i, rule in enumerate(rules):
            if rule.name == event.transition.dest:
                idx = i
        if event.transition.dest == "start":
            return rules[2].name == self.approaching
        return rules[idx - 1].name == self.approaching
    
    def is_valid_init_time(self, event):
        return self.init_frames >= self.num_init_frames

    def inc_init_frames(self, event):
        self.init_frames += 1

    def reset_init_frames(self, event):
        self.init_frames += 1

def get_jumping_jack():
    initial_pose = {"left_shoulder": 0, "right_shoulder": 0}
    initial_pose_deviation = {"left_shoulder": [-30, 30], "right_shoulder": [-30, 30]}
    initial = StaticRule("init", key_data=initial_pose,
                       distance=EuclideanDistance(),
                       key_deviation=initial_pose_deviation)

    start_pose = {"left_shoulder": 0, "right_shoulder": 0}
    start_pose_deviation = {"left_shoulder": [-45, 45], "right_shoulder": [-45, 45]}
    start = StaticRule("start", key_data=start_pose,
                       distance=EuclideanDistance(),
                       key_deviation=start_pose_deviation)

    middle_pose = {"left_shoulder": 120, "right_shoulder": 120}
    middle_pose_deviation = {"left_shoulder": [-20, 100], "right_shoulder": [-20, 100]}
    middle = StaticRule("middle", key_data=middle_pose,
                       distance=EuclideanDistance(),
                       key_deviation=middle_pose_deviation)

    uncertain_pose_deviation = {"left_shoulder": [-20, 10], "right_shoulder": [-20, 20]}
    uncertain = Uncertain(EuclideanDistance(),
                          key_deviation=uncertain_pose_deviation,
                          on_enter="set_uncertain_pose")

    approaching_pose_deviation = {"left_shoulder": [-20, 20], "right_shoulder": [-20, 20]}
    approaching = Uncertain(EuclideanDistance(),
                            name="approaching",
                            key_deviation=approaching_pose_deviation)
    states = [initial, start, middle, uncertain, approaching]
    transitions = [
                   {"trigger": "add_frame", "source": ["init", "start", "middle"], "dest": "="},
                   {"trigger": "add_frame", "source": "approaching", "dest": "=", "prepare": "find_approaching"},
                   {"trigger": "add_frame", "source": "uncertain", "dest": "approaching", "conditions": "left_curr_state", "before": ["set_approaching_pose", "find_approaching"]},

                   {"trigger": "in_init_pos", "source": "init", "dest": "start", "prepare": "inc_init_frames", "conditions": "is_valid_init_time"},

                   {"trigger": "segment", "source": "start", "dest": "middle", "conditions": ["left_curr_state", "in_valid_state"]},
                   {"trigger": "segment", "source": "start", "dest": "uncertain", "conditions": "left_curr_state", "unless": "in_valid_state", "prepare": "set_uncertain_pose" },

                   {"trigger": "segment", "source": "middle", "dest": "start", "conditions": ["left_curr_state", "in_valid_state"]},
                   {"trigger": "segment", "source": "middle", "dest": "uncertain", "conditions": "left_curr_state", "unless": "in_valid_state", "prepare": "set_uncertain_pose"},

                   {"trigger": "segment", "source": "uncertain", "dest": "uncertain", "unless": ["left_curr_state", "in_valid_state"]},
                #    {"trigger": "segment", "source": "uncertain", "dest": "approaching", "conditions": "left_curr_state", "unless": "in_valid_state"},

                   {"trigger": "approach", "source": "approaching", "dest": "start", "conditions": ["is_approaching_dest_parent_pose"], "before": "clear_approaching"},
                   {"trigger": "approach", "source": "approaching", "dest": "middle", "conditions": ["is_approaching_dest_parent_pose"], "before": "clear_approaching"},

                #    {"trigger": "segment", "source": "*", "dest": "start", "conditions": ["left_curr_state", "in_dest_pose"]},
                #    {"trigger": "segment", "source": "*", "dest": "middle", "conditions": ["left_curr_state", "in_dest_pose"]},
                ]
    
    activity = Model({name: state for name, state in zip([state.name for state in states], states)})
    m = Machine(model=activity, states=states, transitions=transitions, initial="init", send_event=True)
    return activity, m

def get_squat():
    initial_pose = {"left_knee": 180, "right_knee": 180}
    initial_pose_deviation = {"left_knee": [-30, 30], "right_knee": [-30, 30]}
    initial = StaticRule("init", key_data=initial_pose,
                       distance=EuclideanDistance(),
                       key_deviation=initial_pose_deviation)

    start_pose = {"left_knee": 180, "right_knee": 180}
    start_pose_deviation = {"left_knee": [-30, 30], "right_knee": [-30, 30]}
    start = StaticRule("start", key_data=start_pose,
                       distance=EuclideanDistance(),
                       key_deviation=start_pose_deviation)

    middle_pose = {"left_knee": 90, "right_knee": 90}
    middle_pose_deviation = {"left_knee": [-30, 30], "right_knee": [-30, 30]}
    middle = StaticRule("middle", key_data=middle_pose,
                       distance=EuclideanDistance(),
                       key_deviation=middle_pose_deviation)

    states = [initial, start, middle]
    transitions = [{"trigger": "in_init_pos", "source": "init", "dest": "start", "prepare": "inc_init_frames", "conditions": "is_valid_init_time"},
                   {"trigger": "segment", "source": "start", "dest": "middle"},
                   {"trigger": "segment", "source": "middle", "dest": "start"}]
    
    activity = Model()
    m = Machine(model=activity, states=states, transitions=transitions, initial="init")
    return activity, m

if __name__ == "__main__":
    # Jumping jack
    activity, m = get_jumping_jack()

    print(activity.state)
    for i in range(30):
        activity.in_init_pos()
    print(activity.state)
    activity.segment({"left_shoulder": 0, "right_shoulder": 0})
    print(activity.state)
    activity.segment({"left_shoulder": 70, "right_shoulder": 70})
    print(activity.state)
    activity.add_frame({"left_shoulder": 49, "right_shoulder": 49})
    activity.approach()
    print(activity.state)
    activity.add_frame({"left_shoulder": 45, "right_shoulder": 45})
    activity.approach()
    print(activity.state)
    activity.segment({"left_shoulder": 0, "right_shoulder": 0})
    print(activity.state)
    # activity.segment({"left_shoulder": 20, "right_shoulder": 20})
    # print(activity.state)
    