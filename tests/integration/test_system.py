import unittest
from azumi.core.personality_engine import Identity
from azumi.core.emotional import EmotionDetector, ResponseGenerator
from azumi.core.memory import ShortTermMemory, LongTermMemory

class TestSystemIntegration(unittest.TestCase):
    def setUp(self):
        self.identity = Identity(
            name="TestSystem",
            base_traits=["analytical", "friendly"]
        )
        self.emotion_detector = EmotionDetector()
        self.response_generator = ResponseGenerator(self.emotion_detector)
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        
    def test_full_interaction(self):
        user_input = "I'm excited to test this system!"
        
        emotions = self.emotion_detector.detect_emotions(user_input)
        response = self.response_generator.generate_response(user_input, {})
        
        self.assertIsNotNone(emotions)
        self.assertIsNotNone(response)
        
if __name__ == "__main__":
    unittest.main()
