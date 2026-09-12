"""The three protocol invariants that cost us a day each (EXPERIMENTS.md, 2026-09-08/09):

1. pollinator negative pool -- on pollinator-side splits, held-out pollinators never appear as sampled negatives;
2. filtered ranking on warm -- an evaluated plant's known partners are removed from pooled metrics and ranked last;
3. the opportunity switch -- with pair_stat_at_inference=False the co-presence term is exactly zero at inference.

  python tests/test_protocol.py
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia import negpool
from antheia.bundle import evaluate_scores
from antheia.models.rgcn import RGCNRanker


def test_negative_pool():
    rng = np.random.default_rng(0)
    negpool.set_pool([3, 5, 7])
    try:
        s = negpool.sample(rng, 10_000, n_total=100)
        assert set(np.unique(s)) <= {3, 5, 7}, "sampled a pollinator outside the training pool"
        assert len(negpool.pool(100)) == 3
    finally:
        negpool.set_pool(None)
    assert len(negpool.pool(100)) == 100, "unset pool must be every pollinator"


def test_filtered_ranking():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(40, 300)); Y = (rng.random((40, 300)) < 0.02).astype(np.int8)
    known = (rng.random((40, 300)) < 0.05) & (Y == 0)
    S[known] += 10.0                                     # a warm model scores its own training partners highest
    q = [str(i) for i in range(40)]; strata = {"a": np.zeros(40, bool)}
    m_raw, _ = evaluate_scores(S, Y, q, strata)
    m_fil, pq = evaluate_scores(S, Y, q, strata, exclude=known)
    assert m_fil["n_excluded"] == int(known.sum())
    assert m_fil["auroc"] > m_raw["auroc"] and m_fil["aupr"] > m_raw["aupr"], "excluding known pairs must remove their penalty"
    # per query: excluded entries sit below every candidate, so they can never occupy a top-k slot
    S_f = np.where(known, S.min() - 1.0, S)
    for i in range(40):
        top = np.argsort(-S_f[i])[:10]
        assert not known[i, top].any()


def test_opportunity_switch():
    r = RGCNRanker(pair_stat="joint", pair_stat_at_inference=False, use_taxon_nodes=False)
    r.dev = "cpu"
    r.ps_p = torch.randn(5, 8); r.ps_q = torch.randn(7, 8); r.ps_scale = 1.0; r.ps_mu = 0.3; r.ps_sd = 0.5
    with torch.no_grad():
        e = r._extra(torch.arange(5), torch.arange(7))
    assert e is not None and e.shape == (5, 7, 1) and float(e.abs().max()) == 0.0, "opportunity term must be exactly zero at inference"
    with torch.enable_grad():
        e_train = r._extra(torch.arange(5), torch.arange(7))
    assert float(e_train.abs().max()) > 0.0, "the term must still be live during training"
    r2 = RGCNRanker(pair_stat="joint", use_taxon_nodes=False); r2.dev = "cpu"
    r2.ps_p, r2.ps_q, r2.ps_scale, r2.ps_mu, r2.ps_sd = r.ps_p, r.ps_q, 1.0, 0.3, 0.5
    with torch.no_grad():
        assert float(r2._extra(torch.arange(5), torch.arange(7)).abs().max()) > 0.0, "default keeps the term on"


if __name__ == "__main__":
    for t in (test_negative_pool, test_filtered_ranking, test_opportunity_switch):
        t(); print(f"ok  {t.__name__}")
