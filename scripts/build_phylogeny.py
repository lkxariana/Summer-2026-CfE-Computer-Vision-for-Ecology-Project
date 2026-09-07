"""Phylogenetic similarity between plant taxa, from the GBOTB.extended megatree.

The error analysis localised the model's failure to plants whose genus never appears in training:
79 of 663 held-out plants at 0.086 nrecall@10 against 0.351 for the rest. The affinity table returns
zeros for such a plant because genus is a flat, two-level key -- two plants in different genera of
one family are treated as equidistant whether they diverged five or eighty million years ago. A dated
phylogeny replaces that with continuous patristic distance, which is what "congeneric transfer, but
graded" requires.

GBOTB.extended (Smith & Brown 2018 for seed plants, Zanne et al. 2014 for pteridophytes, as
distributed with V.PhyloMaker2) has 71,510 tips with branch lengths. Of the universe, 59.3% of
species-rank plants match a tip exactly and 94.7% have their genus present, so unmatched species are
bound to their genus's most recent common ancestor -- the standard V.PhyloMaker placement.

Patristic distance is d(a,b) = depth(a) + depth(b) - 2*depth(MRCA(a,b)), and the MRCA depth is the
total branch length of the shared root-path. Writing B for the sparse tip-by-node incidence matrix of
root-paths and L for branch lengths, that shared length is exactly (B diag(L) B^T), so the whole
matrix follows from one sparse product rather than 90 million tree walks.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]


def parse_newick(text):
    """Iterative Newick parse -> (parent, length, name) arrays. Root has parent -1."""
    parent, length, name = [-1], [0.0], [""]
    stack, cur = [], 0
    i, n = 0, len(text)
    token = re.compile(r"[^(),:;]+")
    while i < n:
        c = text[i]
        if c == "(":
            parent.append(cur); length.append(0.0); name.append("")
            stack.append(cur); cur = len(parent) - 1
            i += 1
        elif c == ",":
            parent.append(stack[-1]); length.append(0.0); name.append("")
            cur = len(parent) - 1
            i += 1
        elif c == ")":
            cur = stack.pop()
            i += 1
        elif c == ":":
            m = re.match(r":(-?[\d.eE+-]+)", text[i:])
            length[cur] = float(m.group(1))
            i += m.end()
        elif c == ";":
            break
        else:
            m = token.match(text, i)
            name[cur] = m.group(0).strip()
            i = m.end()
    return np.array(parent), np.array(length, np.float64), name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--universe", default=ROOT / "data/network/modelled_universe.json")
    ap.add_argument("--out", default=ROOT / "data/features")
    ap.add_argument("--tau", type=float, default=100.0, help="kernel bandwidth in millions of years")
    args = ap.parse_args()

    parent, length, name = parse_newick(open(args.tree).read())
    print(f"[tree] {len(parent):,} nodes", flush=True)
    tip_idx = {nm: i for i, nm in enumerate(name) if nm}
    depth = np.zeros(len(parent))
    order = np.argsort(np.where(parent < 0, -1, parent))          # parents precede children by index
    for v in range(1, len(parent)):
        pass
    # depths by walking each node up once, memoised
    memo = {}

    def dep(v):
        stack = []
        while v not in memo and v >= 0:
            stack.append(v); v = parent[v]
        acc = 0.0 if v < 0 else memo[v]
        for u in reversed(stack):
            acc += length[u]; memo[u] = acc
        return memo[stack[0]] if stack else acc

    for v in range(len(parent)):
        depth[v] = dep(v)

    u = json.load(open(args.universe))
    plants = [p["label"] for p in u["plants"]]
    genus_tips = {}
    for nm, i in tip_idx.items():
        g = nm.split("_")[0]
        genus_tips.setdefault(g, []).append(i)

    node_of, how = np.full(len(plants), -1), []
    for k, s in enumerate(plants):
        key = s.replace(" ", "_")
        if key in tip_idx:
            node_of[k] = tip_idx[key]; how.append("tip")
            continue
        g = s.split()[0]
        if g in genus_tips:                                        # bind to the genus MRCA
            v = genus_tips[g][0]
            anc = set()
            while v >= 0:
                anc.add(v); v = parent[v]
            for t in genus_tips[g][1:]:
                w, path = t, set()
                while w >= 0:
                    path.add(w); w = parent[w]
                anc &= path
            node_of[k] = max(anc, key=lambda x: depth[x])
            how.append("genus")
        else:
            how.append("none")
    how = np.array(how)
    print(f"[place] tip {int((how=='tip').sum()):,}  genus-MRCA {int((how=='genus').sum()):,}  "
          f"unplaced {int((how=='none').sum()):,}", flush=True)

    placed = np.flatnonzero(node_of >= 0)
    rows, cols, vals = [], [], []
    for r in placed:
        v = node_of[r]
        while v >= 0:
            rows.append(r); cols.append(v); vals.append(length[v]); v = parent[v]
    B = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(plants), len(parent)))
    Lm = csr_matrix((np.array(vals), (rows, cols)), shape=(len(plants), len(parent)))
    shared = (Lm @ B.T).toarray()                                  # [P, P] shared root-path length
    dp = depth[np.where(node_of >= 0, node_of, 0)]
    D = dp[:, None] + dp[None, :] - 2 * shared
    D[node_of < 0, :] = np.nan
    D[:, node_of < 0] = np.nan
    np.fill_diagonal(D, 0.0)
    print(f"[dist] {D.shape}  median finite distance {np.nanmedian(D):.1f}", flush=True)

    out = Path(args.out)
    np.save(out / "phylo_dist.npy", D.astype(np.float32))
    np.save(out / "phylo_placed.npy", node_of >= 0)
    print(f"[wrote] {out}/phylo_dist.npy")


if __name__ == "__main__":
    main()
