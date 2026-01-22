#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCL module implementation (Constraint Bank + Generic Constraint Loss)
Paper mapping:
- Prototype construction: Eq.(5)
- CLIP prototype (image/text) averages: Eq.(6)
- External similarity s(x,c): Eq.(7)
- External distribution p(x,c): Eq.(8)
- Relation filtering I+/I-: Eq.(9)
- Target distributions t+/t-: Eq.(12)
- Confidence weight w(x): Eq.(13)
- Sample loss: Eq.(14)
- L_inter / L_intra / L_SCL: Eq.(15)-(17)

This script focuses on the SCL module itself.
You can integrate it into your training loop by:
1) Provide projected embeddings z_tilde for each sample (per-domain)
2) Provide CLIP image/text features for each sample (frozen)
3) Provide cluster assignments per domain (or run kmeans externally)
4) Call build_prototypes(...) then build_constraint_bank(...) then compute_lscl_loss(...)
"""

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn.functional as F


# -------------------------
# Utils
# -------------------------
def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [N, D], b: [M, D] -> [N, M]
    a = l2_normalize(a)
    b = l2_normalize(b)
    return a @ b.t()


def safe_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(logits, dim=dim)


def topk_indices(sim: torch.Tensor, k: int) -> torch.Tensor:
    # sim: [N, M] -> idx: [N, k]
    k = min(k, sim.size(1))
    return torch.topk(sim, k=k, dim=1, largest=True, sorted=True).indices


# -------------------------
# Data containers
# -------------------------
@dataclass
class DomainData:
    """
    Per-domain cached tensors (all already on the same device).
    """
    # projected embeddings used for optimization/retrieval: z~(x), Eq.(4)
    z_tilde: torch.Tensor  # [N, Dz]
    # frozen CLIP image features: Phi_img(x)
    clip_img: torch.Tensor  # [N, Di]
    # frozen CLIP text features (your noun-retrieved text embedding t(x), Eq.(2))
    clip_txt: torch.Tensor  # [N, Dt]
    # cluster id per sample (0..K-1)
    cluster_id: torch.Tensor  # [N]


@dataclass
class Prototypes:
    """
    Prototypes for one domain.
    """
    # prototype in projected space: c_j (Eq.(5)) -> used by q(x)
    c_proj: torch.Tensor  # [K, Dz]
    # CLIP image-level prototype vbar_j (Eq.(6))
    c_img: torch.Tensor  # [K, Di]
    # CLIP text-level prototype tbar_j (Eq.(6))
    c_txt: torch.Tensor  # [K, Dt]


@dataclass
class ConstraintBank:
    """
    Stores filtered constraints for a domain's samples against candidate prototypes.
    For each sample i, we store:
      - candidate prototype ids (global ids in union set)
      - external distribution p over candidates
      - positive mask I+ and negative mask I-
      - scope flags (optional): same-domain vs cross-domain
    """
    cand_idx: torch.Tensor       # [N, C] long
    p_ext: torch.Tensor          # [N, C] float
    pos_mask: torch.Tensor       # [N, C] bool
    neg_mask: torch.Tensor       # [N, C] bool
    cand_is_cross: torch.Tensor  # [N, C] bool  True if prototype from other domain


# -------------------------
# Core: build prototypes
# -------------------------
@torch.no_grad()
def build_prototypes(domain: DomainData, num_clusters: int) -> Prototypes:
    """
    Build prototypes by averaging members in each cluster.
    Assumes domain.cluster_id in [0, num_clusters-1].

    Eq.(5): c_j = mean z~(x) over cluster
    Eq.(6): vbar_j = mean Phi_img(x) ; tbar_j = mean t(x)
    """
    device = domain.z_tilde.device
    N = domain.z_tilde.size(0)
    K = num_clusters

    c_proj = torch.zeros((K, domain.z_tilde.size(1)), device=device, dtype=domain.z_tilde.dtype)
    c_img  = torch.zeros((K, domain.clip_img.size(1)), device=device, dtype=domain.clip_img.dtype)
    c_txt  = torch.zeros((K, domain.clip_txt.size(1)), device=device, dtype=domain.clip_txt.dtype)
    counts = torch.zeros((K,), device=device, dtype=torch.long)

    cid = domain.cluster_id.long().clamp_(0, K - 1)
    counts.scatter_add_(0, cid, torch.ones_like(cid))

    # sum then divide
    c_proj.index_add_(0, cid, domain.z_tilde)
    c_img.index_add_(0, cid, domain.clip_img)
    c_txt.index_add_(0, cid, domain.clip_txt)

    counts_f = counts.clamp_min(1).to(c_proj.dtype).unsqueeze(1)
    c_proj = c_proj / counts_f
    c_img  = c_img  / counts_f
    c_txt  = c_txt  / counts_f

    # normalize (recommended for cosine)
    c_proj = l2_normalize(c_proj)
    c_img  = l2_normalize(c_img)
    c_txt  = l2_normalize(c_txt)

    return Prototypes(c_proj=c_proj, c_img=c_img, c_txt=c_txt)


# -------------------------
# Core: build constraint bank
# -------------------------
@torch.no_grad()
def build_constraint_bank(
    domA: DomainData,
    domB: DomainData,
    protA: Prototypes,
    protB: Prototypes,
    k_in: int = 10,
    k_cross: int = 10,
    tau_p: float = 0.07,
    delta_pos: float = 0.85,
    delta_neg: float = 0.10,
) -> Tuple[ConstraintBank, ConstraintBank]:
    """
    For each sample x in a domain, retrieve candidate prototypes:
      - in-domain top-k (by projected cosine or CLIP cosine; paper says cosine retrieval then CLIP scores)
      - cross-domain top-k
    Then compute external similarity score s(x,c) (Eq.(7)),
    normalize to p(x,c) (Eq.(8)), and apply filtering (Eq.(9)).

    Return bank for A and bank for B.
    """
    device = domA.z_tilde.device
    assert domB.z_tilde.device == device, "A and B must be on same device"

    # Global prototype union indexing:
    # [0..KA-1] for A, [KA..KA+KB-1] for B
    KA = protA.c_proj.size(0)
    KB = protB.c_proj.size(0)

    # Candidate retrieval in projected space (fast & stable)
    simA_in = cosine_sim(domA.z_tilde, protA.c_proj)  # [NA, KA]
    simA_cr = cosine_sim(domA.z_tilde, protB.c_proj)  # [NA, KB]
    idxA_in = topk_indices(simA_in, k_in)             # [NA, k_in] in [0,KA)
    idxA_cr = topk_indices(simA_cr, k_cross)          # [NA, k_cross] in [0,KB)

    simB_in = cosine_sim(domB.z_tilde, protB.c_proj)  # [NB, KB]
    simB_cr = cosine_sim(domB.z_tilde, protA.c_proj)  # [NB, KA]
    idxB_in = topk_indices(simB_in, k_in)             # [NB, k_in] in [0,KB)
    idxB_cr = topk_indices(simB_cr, k_cross)          # [NB, k_cross] in [0,KA)

    # Build candidate index arrays in global union id space
    candA = torch.cat([idxA_in, idxA_cr + KA], dim=1)  # [NA, C]
    candB = torch.cat([idxB_in + KA, idxB_cr], dim=1)  # [NB, C]
    CA = candA.size(1)
    CB = candB.size(1)
    assert CA == CB, "Use same k_in/k_cross for symmetry"
    C = CA

    # Helper: gather prototype CLIP representations for candidates
    def gather_union_proto_imgtxt(cand: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # cand: [N, C] global id
        # returns (vbar, tbar): each [N, C, D]
        # Build union tensors
        union_img = torch.cat([protA.c_img, protB.c_img], dim=0)  # [KA+KB, Di]
        union_txt = torch.cat([protA.c_txt, protB.c_txt], dim=0)  # [KA+KB, Dt]
        v = union_img[cand]  # [N, C, Di]
        t = union_txt[cand]  # [N, C, Dt]
        return v, t

    # External similarity s(x,c) = 0.5*(cos(img,img_proto)+cos(txt,txt_proto))  Eq.(7)
    def external_p(dom: DomainData, cand: torch.Tensor) -> torch.Tensor:
        vbar, tbar = gather_union_proto_imgtxt(cand)  # [N,C,Di], [N,C,Dt]
        x_img = l2_normalize(dom.clip_img).unsqueeze(1)  # [N,1,Di]
        x_txt = l2_normalize(dom.clip_txt).unsqueeze(1)  # [N,1,Dt]
        s_img = (x_img * vbar).sum(dim=-1)               # [N,C]
        s_txt = (x_txt * tbar).sum(dim=-1)               # [N,C]
        s = 0.5 * (s_img + s_txt)                        # [N,C]
        p = safe_softmax(s / tau_p, dim=1)               # Eq.(8)
        return p

    pA = external_p(domA, candA)  # [NA,C]
    pB = external_p(domB, candB)  # [NB,C]

    # Filtering Eq.(9)
    posA = pA >= delta_pos
    negA = pA <= delta_neg
    posB = pB >= delta_pos
    negB = pB <= delta_neg

    # Mask for whether candidate is cross-domain
    # For A: first k_in are in-domain(A), last k_cross are cross(B)
    is_crossA = torch.zeros_like(posA, dtype=torch.bool)
    is_crossA[:, k_in:] = True
    # For B: first k_in are in-domain(B) (global shifted), last k_cross are cross(A)
    is_crossB = torch.zeros_like(posB, dtype=torch.bool)
    is_crossB[:, k_in:] = True

    bankA = ConstraintBank(cand_idx=candA, p_ext=pA, pos_mask=posA, neg_mask=negA, cand_is_cross=is_crossA)
    bankB = ConstraintBank(cand_idx=candB, p_ext=pB, pos_mask=posB, neg_mask=negB, cand_is_cross=is_crossB)
    return bankA, bankB


# -------------------------
# Core: compute L_SCL
# -------------------------
def compute_lscl_loss(
    domA: DomainData,
    domB: DomainData,
    protA: Prototypes,
    protB: Prototypes,
    bankA: ConstraintBank,
    bankB: ConstraintBank,
    tau_q: float = 0.07,
    alpha_intra: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Implements Eq.(11)-(17).

    q(x): softmax over cosine(z~, c_proj) on candidate set, Eq.(11)
    t+(x), t-(x): masked+renormalized using p_ext, Eq.(12)
    w(x): max_c p(x,c), Eq.(13)
    L(x): w*KL(t+||q) - sum t-_j log(1-q_j), Eq.(14)
    L_inter: candidate set restricted to cross-domain prototypes
    L_intra: candidate set restricted to in-domain prototypes
    L_SCL = L_inter + alpha*L_intra
    """

    # Union projected prototypes for q(x)
    union_cproj = torch.cat([protA.c_proj, protB.c_proj], dim=0)  # [KA+KB, Dz]

    def per_domain_loss(dom: DomainData, bank: ConstraintBank) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns (L_inter, L_intra) for this domain
        """
        z = l2_normalize(dom.z_tilde)                      # [N,Dz]
        cand = bank.cand_idx                               # [N,C]
        cproj_cand = union_cproj[cand]                     # [N,C,Dz]
        # q logits: cos(z, c_j) / tau_q
        q_logits = (z.unsqueeze(1) * cproj_cand).sum(dim=-1) / tau_q  # [N,C]
        q = safe_softmax(q_logits, dim=1).clamp(eps, 1.0 - eps)       # [N,C]

        p = bank.p_ext.clamp(eps, 1.0)                     # [N,C]
        w = p.max(dim=1).values.detach()                   # Eq.(13) stop-grad

        # masks
        pos = bank.pos_mask
        neg = bank.neg_mask

        # t+ and t- Eq.(12)
        tpos_unnorm = p * pos.float()
        tneg_unnorm = p * neg.float()

        tpos_sum = tpos_unnorm.sum(dim=1, keepdim=True).clamp_min(eps)
        tneg_sum = tneg_unnorm.sum(dim=1, keepdim=True).clamp_min(eps)

        tpos = tpos_unnorm / tpos_sum
        tneg = tneg_unnorm / tneg_sum

        # KL(t+ || q) = sum t+ (log t+ - log q)
        # For samples with no positive constraints, set KL term to 0 by masking
        has_pos = (pos.any(dim=1)).float()
        kl = (tpos * (torch.log(tpos.clamp_min(eps)) - torch.log(q))).sum(dim=1) * has_pos

        # negative term: - sum t- log(1-q)
        has_neg = (neg.any(dim=1)).float()
        neg_term = -(tneg * torch.log((1.0 - q).clamp(eps, 1.0))).sum(dim=1) * has_neg

        L_all = w * kl + neg_term  # Eq.(14)

        # split inter vs intra using cand_is_cross
        is_cross = bank.cand_is_cross
        # For inter: keep cross candidates only
        # For intra: keep in-domain candidates only
        # We recompute L with masked candidate sets by zeroing out irrelevant entries.
        def masked_loss(keep_mask: torch.Tensor) -> torch.Tensor:
            # keep_mask: [N,C] bool
            # mask q, tpos, tneg then renormalize
            keep_f = keep_mask.float()

            q_m = q * keep_f
            q_m = q_m / q_m.sum(dim=1, keepdim=True).clamp_min(eps)

            tpos_m = tpos * keep_f
            tneg_m = tneg * keep_f

            # renormalize (if empty -> zeros, and kl/neg masked out)
            tpos_sum_m = tpos_m.sum(dim=1, keepdim=True).clamp_min(eps)
            tneg_sum_m = tneg_m.sum(dim=1, keepdim=True).clamp_min(eps)
            tpos_m = tpos_m / tpos_sum_m
            tneg_m = tneg_m / tneg_sum_m

            has_pos_m = (pos & keep_mask).any(dim=1).float()
            has_neg_m = (neg & keep_mask).any(dim=1).float()

            kl_m = (tpos_m * (torch.log(tpos_m.clamp_min(eps)) - torch.log(q_m.clamp(eps, 1.0)))).sum(dim=1) * has_pos_m
            neg_m = -(tneg_m * torch.log((1.0 - q_m).clamp(eps, 1.0))).sum(dim=1) * has_neg_m

            return (w * kl_m + neg_m).mean()

        L_inter = masked_loss(is_cross)          # cross-domain candidates
        L_intra = masked_loss(~is_cross)         # in-domain candidates
        return L_inter, L_intra

    L_inter_A, L_intra_A = per_domain_loss(domA, bankA)
    L_inter_B, L_intra_B = per_domain_loss(domB, bankB)

    L_inter = 0.5 * (L_inter_A + L_inter_B)
    L_intra = 0.5 * (L_intra_A + L_intra_B)
    L_scl = L_inter + alpha_intra * L_intra  # Eq.(17)

    stats = {
        "L_inter": float(L_inter.detach().cpu().item()),
        "L_intra": float(L_intra.detach().cpu().item()),
        "L_scl": float(L_scl.detach().cpu().item()),
    }
    return L_scl, stats


# -------------------------
# Demo main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k_in", type=int, default=10)
    parser.add_argument("--k_cross", type=int, default=10)
    parser.add_argument("--tau_p", type=float, default=0.07)
    parser.add_argument("--tau_q", type=float, default=0.07)
    parser.add_argument("--delta_pos", type=float, default=0.85)
    parser.add_argument("--delta_neg", type=float, default=0.10)
    parser.add_argument("--alpha_intra", type=float, default=1.0)
    parser.add_argument("--save_bank", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Dummy data (replace with your real tensors)
    NA, NB = 512, 512
    Dz, Di, Dt = 256, 512, 512
    KA, KB = 65, 65

    torch.manual_seed(0)
    domA = DomainData(
        z_tilde=l2_normalize(torch.randn(NA, Dz, device=device)),
        clip_img=l2_normalize(torch.randn(NA, Di, device=device)),
        clip_txt=l2_normalize(torch.randn(NA, Dt, device=device)),
        cluster_id=torch.randint(0, KA, (NA,), device=device),
    )
    domB = DomainData(
        z_tilde=l2_normalize(torch.randn(NB, Dz, device=device)),
        clip_img=l2_normalize(torch.randn(NB, Di, device=device)),
        clip_txt=l2_normalize(torch.randn(NB, Dt, device=device)),
        cluster_id=torch.randint(0, KB, (NB,), device=device),
    )

    protA = build_prototypes(domA, KA)
    protB = build_prototypes(domB, KB)

    bankA, bankB = build_constraint_bank(
        domA, domB, protA, protB,
        k_in=args.k_in, k_cross=args.k_cross,
        tau_p=args.tau_p,
        delta_pos=args.delta_pos,
        delta_neg=args.delta_neg,
    )

    L_scl, stats = compute_lscl_loss(
        domA, domB, protA, protB, bankA, bankB,
        tau_q=args.tau_q,
        alpha_intra=args.alpha_intra,
    )
    print(json.dumps(stats, indent=2))

    if args.save_bank:
        # Save bank tensors to a pt file (recommended; json for huge tensors is not practical)
        payload = {
            "bankA": {
                "cand_idx": bankA.cand_idx.cpu(),
                "p_ext": bankA.p_ext.cpu(),
                "pos_mask": bankA.pos_mask.cpu(),
                "neg_mask": bankA.neg_mask.cpu(),
                "cand_is_cross": bankA.cand_is_cross.cpu(),
            },
            "bankB": {
                "cand_idx": bankB.cand_idx.cpu(),
                "p_ext": bankB.p_ext.cpu(),
                "pos_mask": bankB.pos_mask.cpu(),
                "neg_mask": bankB.neg_mask.cpu(),
                "cand_is_cross": bankB.cand_is_cross.cpu(),
            },
        }
        torch.save(payload, args.save_bank)
        print(f"[OK] saved constraint banks to: {args.save_bank}")


if __name__ == "__main__":
    main()
