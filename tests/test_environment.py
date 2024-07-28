import unittest
from src.environments.env import Environment
from src.utils.helpers import load_config

class TestEnvironment(unittest.TestCase):
    def setUp(self):
        config = load_config("configs/config.yaml")
        self.env = Environment(config)

    def test_reset(self):
        state = self.env.reset()
        self.assertEqual(state['turn'], 0)
        self.assertIsNone(state['alice_shared'])
        self.assertIsNone(state['bob_shared'])

    def test_step(self):
        self.env.reset()
        action = "Test action"
        next_state, reward, done, _ = self.env.step(action)
        
        self.assertEqual(next_state['turn'], 1)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

if __name__ == '__main__':
    unittest.main()