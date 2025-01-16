from azumi.core.personality_engine import Identity, DynamicTraits, CognitiveSystem
from azumi.narrative import NarrativeGenerator, ConflictResolver
from azumi.core.memory import MemoryIntegration

def advanced_interaction_example():
    # Create complex personality
    identity = Identity(
        name="Sage",
        base_traits=["analytical", "empathetic", "curious", "strategic"]
    )
    
    traits = DynamicTraits(identity)
    cognitive = CognitiveSystem(identity)
    
    # Initialize narrative components
    narrative_gen = NarrativeGenerator(identity, LongTermMemory())
    conflict_resolver = ConflictResolver(identity)
    
    # Example complex interaction
    context = {
        "scenario": "problem_solving",
        "difficulty": 0.8,
        "emotional_state": "challenged"
    }
    
    narrative = narrative_gen.generate_narrative(context)
    
    print(f"Generated narrative: {narrative}")
    
if __name__ == "__main__":
    advanced_interaction_example()
