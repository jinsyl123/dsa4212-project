"""Fisher test in seed_counts.py, against the values quoted in claims.md. Run: python -m unittest tests.test_seed_counts"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from seed_counts import fisher_p  # noqa: E402


class TestFisher(unittest.TestCase):
    def test_claims_values(self):
        # claims.md, shared rules: against 3/6, 0/10 and 10/10 give 0.036 and 1/10 gives 0.12;
        # against 5/10, 0/10 gives 0.033 and 1/10 gives 0.14; against 4/10, 0/10 gives 0.087
        for (k1, n1, k2, n2), p in [((0, 10, 3, 6), 0.036), ((10, 10, 3, 6), 0.036), ((1, 10, 3, 6), 0.12),
                                    ((0, 10, 5, 10), 0.033), ((1, 10, 5, 10), 0.14), ((0, 10, 4, 10), 0.087)]:
            with self.subTest(k1=k1, k2=k2, n2=n2):
                self.assertAlmostEqual(fisher_p(k1, n1, k2, n2), p, delta=0.005)

    def test_same_counts_give_one(self):
        self.assertEqual(fisher_p(3, 10, 3, 10), 1.0)
        self.assertEqual(fisher_p(0, 10, 0, 6), 1.0)


if __name__ == "__main__":
    unittest.main()
