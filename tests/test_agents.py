import unittest
from src.agents.alice import Alice
from src.agents.bob import Bob

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.alice = Alice("gpt2", "Alice's secret")
        self.bob = Bob("gpt2", "Bob's secret")

    def test_generate_message(self):
        context = "Test context"
        alice_message = self.alice.generate_message(context)
        bob_message = self.bob.generate_message(context)
        
        self.assertIsInstance(alice_message, str)
        self.assertIsInstance(bob_message, str)

    def test_process_message(self):
        message = "Test message"
        alice_response = self.alice.process_message(message)
        bob_response = self.bob.process_message(message)
        
        self.assertIsInstance(alice_response, str)
        self.assertIsInstance(bob_response, str)

if __name__ == '__main__':
    unittest.main()