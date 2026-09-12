"""Token caches for the fusion re-ranker from a field directory: joint, and the two marginal controls.

  joint   top-k (cell, week) by presence p_s(c, w) with a per-month floor       -- train_field_v2.py writes this
  space   top-k cells by the week-summed presence  sum_w p_s(c, w)               -- what range overlap sees
  time    all 52 weeks by the cell-summed presence sum_c p_s(c, w)               -- what phenology overlap sees

The marginal caches exist so that model 2 can be run with the same architecture on tokens that carry
only a marginal (plan M2.7): if joint tokens beat both, the field helps for the stated reason.
Presence is sigmoid(u . h(c, w)) from the field's species vectors and grid embedding.
"""
import argparse
from pathlib import Path
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field-dir", required=True)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--month-floor", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    fd = Path(args.field_dir); dev = args.device if torch.cuda.is_available() else "cpu"
    H = torch.from_numpy(np.load(fd / "grid_h.npy")).to(dev)            # [C, 52, d]
    C, W, d = H.shape
    Hf = H.reshape(-1, d)
    month_of = torch.clamp((torch.arange(W, device=dev) * 7 + 3) // 30, 0, 11)
    for side in ("plant", "poll"):
        U = torch.from_numpy(np.load(fd / f"{side}_field.npy")).to(dev)  # [n, d]
        n = len(U)
        out = {v: {"cell": np.zeros((n, args.topk), np.int16), "week": np.zeros((n, args.topk), np.int8),
                   "logp": np.zeros((n, args.topk), np.float16)} for v in ("joint", "space")}
        tw = np.zeros((n, W), np.float16)
        for s in range(0, n, 256):
            P = torch.sigmoid(U[s:s + 256] @ Hf.T).view(-1, C, W)          # [b, C, W]
            b = P.shape[0]
            # joint with month floor
            flat = P.reshape(b, -1); sel = torch.zeros_like(flat, dtype=torch.bool)
            for m in range(12):
                cols = torch.nonzero(month_of == m).squeeze(1)              # weeks in month m
                idx = (torch.arange(C, device=dev)[:, None] * W + cols[None, :]).reshape(-1)
                top = flat[:, idx].topk(args.month_floor, dim=1).indices
                sel.scatter_(1, idx[top], True)
            top = flat.masked_fill(sel, 2.0).topk(args.topk, dim=1).indices
            out["joint"]["cell"][s:s + b] = (top // W).cpu().numpy(); out["joint"]["week"][s:s + b] = (top % W).cpu().numpy()
            out["joint"]["logp"][s:s + b] = torch.log(torch.gather(flat, 1, top) + 1e-9).cpu().numpy()
            # space marginal: cells by summed presence over weeks (stored as mean, so log p stays comparable)
            Ps = P.mean(2); tops = Ps.topk(args.topk, dim=1).indices
            out["space"]["cell"][s:s + b] = tops.cpu().numpy(); out["space"]["week"][s:s + b] = 0
            out["space"]["logp"][s:s + b] = torch.log(torch.gather(Ps, 1, tops) + 1e-9).cpu().numpy()
            # time marginal: all weeks by presence averaged over cells
            tw[s:s + b] = torch.log(P.mean(1) + 1e-9).cpu().numpy()
        np.savez(fd / f"{side}_tokens.npz", **out["joint"])
        np.savez(fd / f"{side}_tokens_space.npz", **out["space"])
        np.savez(fd / f"{side}_tokens_time.npz", cell=np.zeros((n, W), np.int16), week=np.tile(np.arange(W, dtype=np.int8), (n, 1)), logp=tw)
        print(f"[{side}] {n} taxa -> joint/space top-{args.topk}, time 52 weeks", flush=True)


if __name__ == "__main__":
    main()
