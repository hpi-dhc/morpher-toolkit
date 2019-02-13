from unittest import TestCase

import morpher

class TestJoke(TestCase):
    def test_is_string(self):
        s = morpher.joke()
        self.assertTrue(isinstance(s, str))
