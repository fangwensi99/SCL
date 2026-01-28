import torch
import torch.nn.functional as F

def cosine(a, b):
    # a: (B,D), b: (K,D) -> (B,K)
    return a @ b.t()

@torch.no_grad()
def build_external_prior(
    img_feat, txt_feat,
    proto_img, proto_txt,
    tau_p: float,
):
    """
    img_feat:  (B, Dclip)
    txt_feat:  (B, Dclip)
    proto_img: (K, Dclip)
    proto_txt: (K, Dclip)
    returns p: (B, K)
    """
    s_img = cosine(img_feat, proto_img)
    s_txt = cosine(txt_feat, proto_txt)
    s = 0.5 * (s_img + s_txt)
    p = F.softmax(s / tau_p, dim=-1)
    return p

def selective_constraint_loss(
    z_tilde,                 # (B, D)
    proto_z,                 # (K, D) projected-space prototypes
    p_ext,                   # (B, K) external prior prob over candidate prototypes
    tau_q: float,
    delta_pos: float,
    delta_neg: float,
    eps: float = 1e-6,
):
    """
    Implements Eq.(11)-(14) over a given candidate set.
    Here we assume proto_z and p_ext already correspond to that candidate set.
    """
    # q(x): model-induced similarity distribution Eq.(11)
    q_logits = cosine(z_tilde, proto_z) / tau_q
    q = F.softmax(q_logits, dim=-1).clamp(min=eps, max=1 - eps)

    # Indicators Eq.(9)
    I_pos = (p_ext >= delta_pos).float()
    I_neg = (p_ext <= delta_neg).float()

    # If a sample has no pos/neg after filtering, avoid NaN by fallback to zeros.
    pos_mass = (I_pos * p_ext).sum(dim=-1, keepdim=True).clamp_min(eps)
    neg_mass = (I_neg * p_ext).sum(dim=-1, keepdim=True).clamp_min(eps)

    t_pos = (I_pos * p_ext) / pos_mass  # Eq.(12)
    t_neg = (I_neg * p_ext) / neg_mass

    # w(x) Eq.(13)
    w = p_ext.max(dim=-1).values.detach()

    # Eq.(14)
    kl = (t_pos * (t_pos.clamp_min(eps).log() - q.log())).sum(dim=-1)
    neg_term = -(t_neg * torch.log(1.0 - q)).sum(dim=-1)
    loss = w * kl + neg_term
    return loss.mean()
