import copy, importlib.util, math, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load():
    spec = importlib.util.spec_from_file_location("sub_model", ROOT / "submission" / "model.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.predict

S1 = {"normalized_name": "OpenAI GPT-4o", "provider": "OpenAI", "harness": "h", "reasoning_effort": "", "release_date": "2024-05-13"}
S2 = {"normalized_name": "Qwen3 8B", "provider": "Alibaba", "harness": "h", "reasoning_effort": "", "release_date": "2025-04-29"}
def it(b, k): return {"item_content": f"question number {k} about topic {b}", "item_features": "tier=medium", "interactors": "", "benchmark_id": b}
def lab(s, b, k, y): return [[s, it(b, k)], y]

class T(unittest.TestCase):
    def setUp(self): self.predict = load()
    def test_budget0_is_prior_probability(self):
        v = self.predict([S1, it("b1", 0)], []); self.assertTrue(0.01 <= v <= 0.99)
        self.assertEqual(v, self.predict([S1, it("b1", 0)], None)); self.assertEqual(v, self.predict([S1, it("b1", 0)]))
    def test_labels_move_prediction_in_the_right_direction(self):
        base = self.predict([S1, it("b1", 99)], [])
        up = self.predict([S1, it("b1", 99)], [lab(S1, "b1", k, 1) for k in range(3)])
        down = self.predict([S1, it("b1", 99)], [lab(S1, "b1", k, 0) for k in range(3)])
        self.assertGreater(up, base); self.assertLess(down, base)
    def test_other_benchmarks_do_not_leak(self):
        L = [lab(S1, "b2", k, 1) for k in range(15)]
        self.assertEqual(self.predict([S1, it("b1", 99)], L), self.predict([S1, it("b1", 99)], [lab(S1, "b3", 0, 1)][:0] + [lab(S1, "b9", k, 0) for k in range(15)]))
    def test_cross_subject_pooling_and_item_effect(self):
        L = [lab(S2, "b1", k, 1) for k in range(10)]
        pooled = self.predict([S1, it("b1", 99)], L); self.assertGreater(pooled, self.predict([S1, it("b1", 99)], []))
        same_item = self.predict([S1, it("b1", 0)], [lab(S2, "b1", 0, 0)] + [lab(S2, "b1", k, 1) for k in range(1, 10)])
        other_item = self.predict([S1, it("b1", 50)], [lab(S2, "b1", 0, 0)] + [lab(S2, "b1", k, 1) for k in range(1, 10)])
        self.assertLess(same_item, other_item)
    def test_always_valid_probability(self):
        weird = [{}, {"item_content": "x" * 100000}, {"item_content": "ünïcödé 日本語 🚀", "benchmark_id": "b1"}]
        for item in weird:
            for L in ([], [lab(S1, "b1", 0, 1)]):
                v = self.predict([S1 if item else {}, item], L); self.assertTrue(math.isfinite(v) and 0 <= v <= 1)
    def test_inputs_not_mutated_and_budget_order_independent(self):
        L = [lab(S1 if k % 2 else S2, "b1", k, k % 2) for k in range(31)]; before = copy.deepcopy(L)
        a = [self.predict([S1, it("b1", 99)], L[:n]) for n in (31, 0, 7, 1, 15, 3)]
        b = [self.predict([S1, it("b1", 99)], L[:n]) for n in (31, 0, 7, 1, 15, 3)]
        self.assertEqual(a, b); self.assertEqual(L, before)
    def test_bad_labels_ignored(self):
        self.assertEqual(self.predict([S1, it("b1", 0)], [lab(S1, "b1", 0, float("nan"))]), self.predict([S1, it("b1", 0)], []))

if __name__ == "__main__": unittest.main()
