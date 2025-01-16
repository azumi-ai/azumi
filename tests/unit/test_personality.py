import unittest
from azumi.core.personality_engine import Identity, DynamicTraits

class TestPersonality(unittest.TestCase):
    def setUp(self):
        self.identity = Identity(
            name="Test",
            base_traits=["friendly", "analytical"]
        )
        
    def test_identity_creation(self):
        self.assertEqual(self.identity.core.name, "Test")
        self.assertEqual(len(self.identity.core.traits), 2)
        
    def test_trait_evolution(self):
        traits = DynamicTraits(self.identity)
        initial_values = self.identity.core.values.copy()
        
        traits.evolve_traits({"friendly": 0.8})
        
        self.assertNotEqual(
            self.identity.core.values["friendly"],
            initial_values["friendly"]
        )
