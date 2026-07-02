"""Claim-level tests (step 6): best-checkpoint selection + cross-LM comparisons."""
import numpy as np
import pandas as pd

from src.analysis import claim_tests as ct


def _fake_results():
    rows = []
    for lm, dlls in [("gpt", [2.0, 5.0, 1.0]), ("llama", [-1.0, 0.5, -2.0])]:
        for idx, dll in enumerate(dlls, start=1):
            rows.append(
                {
                    "model_lm": lm,
                    "measure": "TFT",
                    "model": "aligned",
                    "index": idx,
                    "epoch": 4**idx,
                    "delta_ll": dll,
                    "p_lrt": 0.01 if dll > 2 else 0.5,
                    "b_surprisal": dll / 100,
                    "se_surprisal": 0.01,
                }
            )
    return pd.DataFrame(rows)


def _fake_reader_ll(best):
    rng = np.random.default_rng(0)
    readers = [f"r{i}" for i in range(20)]
    rows = []
    for lm, gain in [("gpt", 0.5), ("llama", 0.0)]:
        idx = best.set_index("model_lm").loc[lm, "index"]
        for r in readers:
            base = rng.normal(-100, 5)
            rows.append({"model_lm": lm, "measure": "TFT", "model": "base_ref",
                         "index": pd.NA, "reader_id": r, "ll_reader": base})
            rows.append({"model_lm": lm, "measure": "TFT", "model": "aligned",
                         "index": idx, "reader_id": r,
                         "ll_reader": base + gain + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


def test_best_checkpoints_picks_max_delta_ll():
    best = ct.best_checkpoints(_fake_results())
    b = best.set_index("model_lm")
    assert b.loc["gpt", "index"] == 2
    assert b.loc["llama", "index"] == 2
    # Wald CI around the surprisal slope.
    assert b.loc["gpt", "ci_low"] < b.loc["gpt", "b_surprisal"] < b.loc["gpt", "ci_high"]


def test_run_claim_tests_cross_lm():
    results = _fake_results()
    best = ct.best_checkpoints(results)
    reader_ll = _fake_reader_ll(best)
    best2, cross = ct.run_claim_tests(results, reader_ll)
    pd.testing.assert_frame_equal(best, best2)
    row = cross.iloc[0]
    assert {row["lm1"], row["lm2"]} == {"gpt", "llama"}
    # slope difference: (5.0 - 0.5)/100 with se 0.01 each -> strongly significant.
    assert row["p"] < 0.01
    # per-reader ΔLL: gpt gains ~0.5/reader, llama ~0 -> paired tests detect it.
    assert row["n_readers"] == 20
    assert row["p_t"] < 0.01
    assert row["p_wilcoxon"] < 0.01
    assert abs(row["mean_diff"] - 0.5) < 0.2
