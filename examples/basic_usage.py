from azumi.core.personality_engine import Identity
from azumi.core.emotional import EmotionDetector, ResponseGenerator
from azumi.core.memory import ShortTermMemory, LongTermMemory

def basic_personality_example():
    # Create a basic personality
    identity = Identity(
        name="Aria",
        base_traits=["friendly", "creative", "analytical"]
    )
    
    # Initialize emotional components
    emotion_detector = EmotionDetector()
    response_generator = ResponseGenerator(emotion_detector)
    
    # Initialize memory
    short_term = ShortTermMemory()
    long_term = LongTermMemory()
    
    # Example interaction
    user_input = "I'm feeling excited about learning AI!"
    emotions = emotion_detector.detect_emotions(user_input)
    response = response_generator.generate_response(user_input, {})
    
    print(f"Detected emotions: {emotions}")
    print(f"Response: {response}")
