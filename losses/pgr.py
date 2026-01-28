import torch
import torch.nn.functional as F

def pgr_loss(z, proto, tau_in=0.07, m=5, r=0.3, eps=1e-6):
    """
    z: (B, D) normalized
    proto: (K, D) normalized
    Implements Eq.(18)-(22).
    gamma is set as r-quantile of gap distribution within the batch (Eq.(21) comment).
    """
    # a_k(x) Eq.(18)
    logits = (z @ proto.t()) / tau_in
    a = F.softmax(logits, dim=-1)

    # pull Eq.(19)
    cos_all = (z @ proto.t()).clamp(-1, 1)
    pull = (a * (1.0 - cos_all)).sum(dim=-1).mean()

    # k+(x) Eq.(20)
    k_pos = a.argmax(dim=-1)  # (B,)

    # competing prototypes: top-m excluding k_pos
    topv, topi = torch.topk(cos_all, k=min(m+1, cos_all.size(1)), dim=-1)
    # remove self
    comp = []
    for b in range(z.size(0)):
        idxs = topi[b].tolist()
        kp = int(k_pos[b].item())
        idxs = [j for j in idxs if j != kp][:m]
        if len(idxs) < m:
            # pad with random / repeat
            idxs = (idxs + [idxs[-1]] * (m - len(idxs))) if idxs else [kp]*m
        comp.append(idxs)
    comp = torch.tensor(comp, device=z.device, dtype=torch.long)  # (B,m)

    cos_pos = cos_all[torch.arange(z.size(0), device=z.device), k_pos]  # (B,)
    cos_neg = cos_all.gather(1, comp)  # (B,m)

    # gap distribution for gamma
    gaps = (cos_pos.unsqueeze(1) - cos_neg).reshape(-1).detach()
    gamma = torch.quantile(gaps, r).item()

    # push Eq.(21)
    hinge = F.relu(gamma - cos_pos.unsqueeze(1) + cos_neg)
    push = hinge.mean()

    return pull + push
