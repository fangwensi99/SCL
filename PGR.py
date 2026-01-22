#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PGR module implementation (Prototype-centered Pull & Push)
Paper mapping:
- Soft assignment a_k(x): Eq.(18)
- Pull loss: Eq.(19)
- Push loss: Eq.(20)-(21)
- Gamma margin as r-quantile of empirical gap distribution: described near Eq.(21)
- L_PGR = L_pull + L_push: Eq.(22)

This script implements PGR as a standalone loss module.
You provide:
- projected embeddings z_tilde [N, Dz]
- domain-specific prototypes c_proj [K, Dz]
Then it returns L_PGR and statistics.

Important:
- This matches the paper design: push uses the most confident prototype k+(x) and
  the top-m competing prototypes K-(x) (nearest competitors).
"""

import argparse
import json
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return a @ b.t()


def compute_pgr_loss(
    z_tilde: torch.Tensor,   # [N, Dz]
    c_proj: torch.Tensor,    # [K, Dz]
    tau_in: float = 0.07,    # Eq.(18)
    m_comp: int = 5,         # number of competing prototypes
    r_quantile: float = 0.3, # push activation ratio r
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Returns:
      L_pgr, stats
    """
    z = l2_normalize(z_tilde)
    c = l2_normalize(c_proj)
    N, K = z.size(0), c.size(0)

    # Cosine similarities to prototypes: [N, K]
    sim = cosine_sim(z, c)

    # Soft assignments a_k(x): Eq.(18)
    a = torch.softmax(sim / tau_in, dim=1)  # [N,K]

    # Pull: sum_k a_k(x) * (1 - cos(z, c_k))  Eq.(19)
    L_pull = (a * (1.0 - sim)).sum(dim=1).mean()

    # Identify k+(x): Eq.(20)
    k_plus = torch.argmax(a, dim=1)  # [N]

    # For each sample, choose m most similar competing prototypes excluding k_plus
    m_comp = min(m_comp, max(1, K - 1))

    # Build competitor indices using sim (largest are closest)
    sim_excl = sim.clone()
    sim_excl[torch.arange(N, device=z.device), k_plus] = -1e9
    comp_idx = torch.topk(sim_excl, k=m_comp, dim=1, largest=True, sorted=True).indices  # [N,m]

    # Compute empirical gap distribution:
    # gap = cos(z, c_{k+}) - cos(z, c_k) for k in competitors
    sim_pos = sim[torch.arange(N, device=z.device), k_plus].unsqueeze(1)  # [N,1]
    sim_neg = sim[torch.arange(N, device=z.device).unsqueeze(1), comp_idx]  # [N,m]
    gap = (sim_pos - sim_neg).reshape(-1)  # [N*m]

    # Gamma as r-quantile of empirical gap distribution (paper: set so approx r fraction activates)
    # If r_quantile is small, gamma small -> more activation; if large, gamma large -> less.
    r_quantile = float(min(max(r_quantile, 0.0), 1.0))
    gamma = torch.quantile(gap.detach().clamp_min(0.0), q=r_quantile).item()

    # Push: mean over competitors of hinge( gamma - cos(z,c_{k+}) + cos(z,c_k) ) Eq.(21)
    hinge = torch.relu(torch.tensor(gamma, device=z.device) - sim_pos + sim_neg)  # [N,m]
    L_push = hinge.mean(dim=1).mean()

    L_pgr = L_pull + L_push  # Eq.(22)

    stats = {
        "L_pull": float(L_pull.detach().cpu().item()),
        "L_push": float(L_push.detach().cpu().item()),
        "L_pgr": float(L_pgr.detach().cpu().item()),
        "gamma": float(gamma),
    }
    return L_pgr, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tau_in", type=float, default=0.07)
    parser.add_argument("--m_comp", type=int, default=5)
    parser.add_argument("--r_quantile", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Dummy example (replace with your real tensors)
    torch.manual_seed(0)
    N, K, Dz = 512, 65, 256
    z_tilde = l2_normalize(torch.randn(N, Dz, device=device))
    c_proj = l2_normalize(torch.randn(K, Dz, device=device))

    L_pgr, stats = compute_pgr_loss(
        z_tilde=z_tilde,
        c_proj=c_proj,
        tau_in=args.tau_in,
        m_comp=args.m_comp,
        r_quantile=args.r_quantile,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
