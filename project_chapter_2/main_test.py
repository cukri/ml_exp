import unittest
from main import display_scores

class TestMain(unittest.TestCase):
    def test_display_scores(self):
        self.assertEqual(display_scores(), 69104.07998247063)

if __name__ == '__main__':
    unittest.main()