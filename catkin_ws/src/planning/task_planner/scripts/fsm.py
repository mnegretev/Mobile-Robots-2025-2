import time

import ros

class State:
    def __init__(self, name, say=None, execute=None, transition=None, next=None):
        self.name = name
        self.say = say
        self.execute = execute
        self.transition = transition
        self.next = next

    def __say__(self, say_funct=print):
        if self.say:
            print(f"State: {self.name}")
            # 'say' function must be available in the main script
            if callable(self.say):
                say_funct(self.say())
            else:
                say_funct(self.say)

    def __execute__(self):
        if self.execute:
            self.execute()

    def __transition__(self):
        if self.transition:
            if isinstance(self.next, tuple):
                return self.next[0] if self.transition() else self.next[1]
            else:
                while not self.transition():
                    time.sleep(0.5)
                return self.next
        return self.next

    def run(self, say_funct=print):
        self.__say__(say_funct)
        self.__execute__()
        return self.__transition__()

class FSM:
    def __init__(self, initial_state: State, states: dict = None, say = None):
        self.current_state = initial_state
        self.initial_state = initial_state
        self.states = states if states else {}
        self.say = say

    def run(self, ros_shutdown: callable = None):
        try:
            while not ros_shutdown():
                next_state = self.current_state.run(self.say)
                print(f"Transitioning to: {next_state}")
                if next_state is None:
                    break
                self.current_state = self.states.get(next_state)
                if self.current_state is None:
                    print(f"State '{next_state}' not found.")
                    break
        except Exception as e:
            print(f"FSM encountered an error: {e}")
            self.say(f"Error occurred on {self.current_state.name} state. Returning to initial state.")
            self.current_state = self.initial_state
            self.run(ros_shutdown)