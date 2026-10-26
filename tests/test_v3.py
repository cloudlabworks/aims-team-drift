"""Unit tests for the v3 submission: labeling.py (kit-exact, deterministic), the stdlib encoder, and the model contract.
Run from the aims/ folder:  .venv/bin/python -I -m unittest discover -s tests -p 'test_v3.py' -v
"""
import hashlib, importlib.util, json, math, os, random, sys, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, 'submission3')
sys.path.insert(0, SUB)
import labeling as LB  # noqa: E402

KIT = os.path.join(ROOT, 'paiec_baseline', 'tools', 'streaming_ingestion.py')


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def kit_random_decision(context, current_input):
    """The kit's random-mode rule, copied from tools/streaming_ingestion.py (asserted present below)."""
    key = json.dumps([context, current_input], sort_keys=True, separators=(",", ":"))
    uniform = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64
    return uniform < min(1, context["labels_remaining"] / context["items_remaining"])


def _input(i):
    return [{"normalized_name": "M", "provider": "P"}, {"item_content": f"question {i}?", "item_features": "", "interactors": "", "benchmark_id": "benchmark_x"}]


def stream(n_items=50, budget=31, tag=""):
    got, remaining = [], budget
    for pos in range(n_items):
        ctx = dict(subject_id="subject_000", benchmark_id="benchmark_x", labels_remaining=remaining,
                   labels_acquired=len(got), max_labels=budget, items_remaining=n_items - pos)
        if LB.acquisition_function(_input(f"{tag}{pos}"), context=ctx):
            got.append(pos); remaining -= 1
    return got


class TestAcquisition(unittest.TestCase):
    def test_kit_source_still_has_the_rule(self):
        src = open(KIT).read()
        for frag in ('json.dumps([context, current_input], sort_keys=True, separators=(",", ":"))',
                     'hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64',
                     'query = uniform < min(1, context["labels_remaining"] / context["items_remaining"])'):
            self.assertIn(frag, src)

    def test_decisions_equal_kit_rule(self):
        rnd = random.Random(7)
        for _ in range(5000):
            n = rnd.randint(1, 120); rem = rnd.randint(1, 31)
            ctx = dict(subject_id=f"subject_{rnd.randint(0, 9):03d}", benchmark_id="b", labels_remaining=rem,
                       labels_acquired=31 - rem, max_labels=31, items_remaining=n)
            inp = _input(rnd.random())
            self.assertIs(LB.acquisition_function(inp, context=ctx), bool(kit_random_decision(ctx, inp)))

    def test_deterministic(self):
        self.assertEqual(stream(tag="a"), stream(tag="a"))
        self.assertNotEqual(stream(tag="a"), stream(tag="b"))

    def test_budget_exactly_reached_and_never_exceeded(self):
        for t in range(60):
            for n_items in (31, 32, 40, 50, 200):
                got = stream(n_items, 31, tag=f"{t}-")
                self.assertEqual(len(got), 31)  # forced reveal once items_remaining <= labels_remaining
        self.assertEqual(len(stream(10, 31)), 10)  # fewer items than labels: all of them
        self.assertEqual(stream(50, 0), [])

    def test_nested_budgets_are_prefixes(self):
        got = stream(80, 31)
        for n in (0, 1, 3, 7, 15, 31):
            self.assertEqual(got[:n], got[:31][:n])

    def test_edge_contexts(self):
        ctx = dict(labels_remaining=0, items_remaining=5)
        self.assertIs(LB.acquisition_function(_input(0), context=ctx), False)
        self.assertIs(LB.acquisition_function(_input(0), context=dict(labels_remaining=3, items_remaining=0)), False)
        with self.assertRaises(ValueError):
            LB.acquisition_function(_input(0))

    def test_platform_resolves_stream_mode_without_prediction(self):
        hook_mod = _load(KIT, 'kit_stream')
        hook = hook_mod.AcquisitionHook(LB.acquisition_function)
        self.assertEqual(hook.mode, "stream")
        self.assertFalse(hook.needs_prediction)  # no extra predict() calls for acquisition
        ctx = dict(subject_id="s", benchmark_id="b", labels_remaining=31, labels_acquired=0, max_labels=31, items_remaining=40)
        self.assertIs(type(hook(_input(1), None, [], ctx)), bool)


class TestLabels(unittest.TestCase):
    def test_binarize(self):
        b = LB.binarize_response
        self.assertEqual([b(0), b(1), b(0.0), b(1.0), b("1"), b(True)], [0, 1, 0, 1, 1, 1])
        for bad in (None, float("nan"), 0.5, 2, -1, "x", ""):
            self.assertIsNone(b(bad))

    def test_aggregate(self):
        self.assertEqual(LB.aggregate_responses([1, 0, 1, float("nan"), 0.5, None]), (3, 2))
        self.assertEqual(LB.aggregate_responses([]), (0, 0))

    def test_construct_label_deterministic_and_unbiased(self):
        self.assertEqual(LB.construct_label(1, 1, "k"), 1)
        self.assertEqual(LB.construct_label(0, 3, "k"), 0)
        self.assertEqual(LB.construct_label(2, 3, "z"), LB.construct_label(2, 3, "z"))
        m = sum(LB.construct_label(1, 4, f"key{i}") for i in range(20000)) / 20000
        self.assertAlmostEqual(m, 0.25, delta=0.02)
        for bad in ((0, 0), (3, 2), (-1, 2)):
            with self.assertRaises(ValueError):
                LB.construct_label(*bad, "k")


class TestEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import embed as E
        cls.E = E
        cls.enc = E.Encoder(os.path.join(SUB, 'artifacts'))

    def test_unit_norm_and_empty(self):
        v = self.enc.encode("What is the capital of France?")
        self.assertEqual(len(v), self.E.DIM)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in v)), 1.0, places=5)
        self.assertEqual(self.enc.encode(""), [0.0] * self.E.DIM)
        self.assertEqual(self.enc.encode("核酸"), [0.0] * self.E.DIM)  # all-[UNK] text -> zero vector like model2vec

    def test_similarity_is_semantic(self):
        cos = lambda a, b: sum(x * y for x, y in zip(a, b))
        a = self.enc.encode("What is the capital city of France?")
        b = self.enc.encode("Which city is the capital of Germany?")
        c = self.enc.encode("def quicksort(arr): return sorted(arr)")
        self.assertGreater(cos(a, b), cos(a, c))

    def test_tokenizer_special_tokens_and_accents(self):
        ids = self.enc.token_ids("Café [SEP] naïve")
        self.assertEqual(ids, self.enc.token_ids("cafe [SEP] naive"))
        self.assertIn(self.enc.vocab["[SEP]"], ids)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.M = _load(os.path.join(SUB, 'model.py'), 'v3model')
        cls.path = os.path.join(SUB, 'artifacts')

    def test_contract_and_budget_zero(self):
        subj = {"normalized_name": "M", "provider": "P", "release_date": "2025-01-01", "access_date": "", "harness": "h",
                "harness_version": "", "reasoning_effort": "", "subject_features_extra": ""}
        item = {"item_content": "What is 2+2?", "item_features": "", "interactors": "", "benchmark_id": "benchmark_x"}
        p0 = self.M.predict([subj, item], [])
        self.assertTrue(0.0 <= p0 <= 1.0 and math.isfinite(p0))
        lab = [[[subj, dict(item, item_content=f"q{i}")], i % 2] for i in range(7)]
        p1 = self.M.predict([subj, item], lab)
        self.assertTrue(0.0 <= p1 <= 1.0 and math.isfinite(p1))
        for _ in range(3):
            self.assertEqual(self.M.predict([subj, item], lab), p1)  # deterministic

    def test_embedding_term_generalises_to_unseen_items(self):
        subj = {"normalized_name": "M", "provider": "P", "release_date": "2025-01-01", "access_date": "", "harness": "h",
                "harness_version": "", "reasoning_effort": "", "subject_features_extra": ""}
        mk = lambda t: {"item_content": t, "item_features": "", "interactors": "", "benchmark_id": "benchmark_x"}
        geo = [f"What is the capital city of {c}?" for c in ("France", "Spain", "Italy", "Peru", "Japan", "Kenya", "Chile", "Egypt")]
        code = [f"def f{i}(x): return sorted(x)[{i}] # python function" for i in range(8)]
        lab = [[[subj, mk(t)], 1] for t in geo] + [[[subj, mk(t)], 0] for t in code]
        p_geo = self.M.predict([subj, mk("What is the capital city of Norway?")], lab)
        p_code = self.M.predict([subj, mk("def g(y): return max(y) # python function")], lab)
        self.assertGreater(p_geo, p_code)

    def test_bucket_of(self):
        b = self.M.bucket_of
        self.assertEqual([b(n) for n in (0, 1, 2, 3, 4, 7, 8, 15, 16, 31)], [0, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    def test_calibration_maps(self):
        meta, arr, _ = self.M._load_art(self.path)
        plat = {"kind": "platt", "buckets": [{"a": 0.5, "c": 0.0}] * 6}
        iso = {"kind": "iso", "buckets": [{"x": [0.0, 0.5, 1.0], "y": [0.2, 0.5, 0.8]}] * 6}
        for cal, want in ((plat, 0.5), (iso, 0.5)):
            P = self.M.Predictor(meta, arr, calib=cal)
            self.assertAlmostEqual(P.calibrate(0.0, 3), want, places=6)
        # isotonic interpolation: raw p = 0.9 lies 80% of the way from (0.5 -> 0.5) to (1.0 -> 0.8)
        self.assertAlmostEqual(self.M.Predictor(meta, arr, calib=iso).calibrate(math.log(0.9 / 0.1), 0), 0.74, places=6)


if __name__ == '__main__':
    unittest.main()
