# src/environment/env.py

class Environment:
    def __init__(self, config):
        self.alice_secret = config['alice_secret']
        self.bob_secret = config['bob_secret']
        self.max_turns = config['max_turns']
        self.current_turn = 0

    def reset(self):
        self.current_turn = 0
        return self._get_state()

    def step(self, action):
        self.current_turn += 1
        done = self.current_turn >= self.max_turns
        return self._get_state(), done

    def _get_state(self):
        return {
            'turn': self.current_turn,
            'alice_secret': self.alice_secret,
            'bob_secret': self.bob_secret
        }