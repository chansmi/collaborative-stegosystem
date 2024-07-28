import unittest
from src.overseer.eve import Eve

class TestOverseer(unittest.TestCase):
    def setUp(self):
        self.eve = Eve("gpt2")

    def test_evaluate(self):
        message = "This is a test message without any secret information."
        score = self.eve.evaluate(message)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

if __name__ == '__main__':
    unittest.main()