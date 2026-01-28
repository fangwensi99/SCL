import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TxtImageDataset
from models.clip_backbone import CLIPBackbone
from models.projector import MLPProjector
from losses.scl import build_external_prior, selective_constraint_loss
from losses.pgr import pgr_loss
from utils.seed import set_seed
from utils.logger import Logger
from utils.meters import AvgMeter
from utils.kmeans import run_kmeans
from utils.retrieval import precision_at_k

# Office-Home 65 class names (常用版本；如你label映射不同，可改成你的版本)
OFFICEHOME_CLASSES = [
 "Alarm_Clock","Backpack","Batteries","Bed","Bike","Bottle","Bucket","Calculator","Calendar","Candles",
 "Chair","Clipboards","Computer","Couch","Curtains","Desk_Lamp","Drill","Eraser","Exit_Sign","Fan",
 "File_Cabinet","Flipflops","Flowers","Folder","Fork","Glasses","Hammer","Helmet","Kettle","Keyboard",
 "Knives","Lamp_Shade","Laptop","Marker","Mop","Mouse","Mug","Notebook","Oven","Pan","Paper_Clip",
 "Pen","Pencil","Postit_Notes","Printer","Push_Pin","Radio","Refrigerator","Ruler","Scissors","Screwdriver",
 "Shelf","Sink","Sneakers","Soda","Speaker","Spoon","Table","Telephone","ToothBrush","Toys","Trash_Can",
 "TV","Webcam"
]

@torch.no_grad()
def build_semantic_enhanced(clip, images, noun_text_emb, top_m=5):
    """
    v(x)=CLIP_img(x)
    t(x)=weighted aggregation of top-m noun embeddings retrieved by image-text similarity
    z(x)=concat(v,t)
    """
    v = clip.encode_image(images)  # (B,Dc)
    # retrieve nouns by similarity v @ noun_text_emb^T
    sim = v @ noun_text_emb.t()  # (B, Nnoun)
    topv, topi = torch.topk(sim, k=min(top_m, noun_text_emb.size(0)), dim=-1)
    w = F.softmax(topv, dim=-1)  # (B,top_m)
    t = (w.unsqueeze(-1) * noun_text_emb[topi]).sum(dim=1)
    t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    z = torch.cat([v, t], dim=-1)
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return v, t, z

@torch.no_grad()
def extract_all_features(loader, clip, projector, noun_text_emb, top_m, device):
    all_ztilde = []
    all_v = []
    all_t = []
    all_y = []
    for images, y, _ in tqdm(loader, desc="Extract", leave=False):
        images = images.to(device, non_blocking=True)
        v, t, z = build_semantic_enhanced(clip, images, noun_text_emb, top_m=top_m)
        zt = projector(z)
        all_ztilde.append(zt.cpu())
        all_v.append(v.cpu())
        all_t.append(t.cpu())
        all_y.append(y)
    return (torch.cat(all_ztilde, dim=0),
            torch.cat(all_v, dim=0),
            torch.cat(all_t, dim=0),
            torch.cat(all_y, dim=0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art_txt", type=str, required=True)
    ap.add_argument("--clipart_txt", type=str, required=True)

    ap.add_argument("--clip_model", type=str, default="ViT-B-32")
    ap.add_argument("--clip_pretrained", type=str, default="openai")
    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--proj_hidden", type=int, default=1024)
    ap.add_argument("--proj_layers", type=int, default=2)

    ap.add_argument("--K", type=int, default=65, help="num prototypes per domain")
    ap.add_argument("--k_cand", type=int, default=10, help="top-k candidate prototypes")
    ap.add_argument("--update_proto_every", type=int, default=1, help="epochs")

    ap.add_argument("--top_m_noun", type=int, default=5)

    ap.add_argument("--tau_p", type=float, default=0.02)
    ap.add_argument("--tau_q", type=float, default=0.07)
    ap.add_argument("--tau_in", type=float, default=0.07)
    ap.add_argument("--delta_pos", type=float, default=0.85)
    ap.add_argument("--delta_neg", type=float, default=0.10)

    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.5)

    ap.add_argument("--m_comp", type=int, default=5)
    ap.add_argument("--r_push", type=float, default=0.3)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=5e-4)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logdir", type=str, default="runs/scl_art_clipart")

    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.logdir, exist_ok=True)
    logger = Logger(args.logdir)
    logger.log(str(vars(args)))

    # CLIP (frozen)
    clip = CLIPBackbone(args.clip_model, args.clip_pretrained, device=device)

    # transforms
    preprocess = clip.preprocess

    # datasets
    dsA = TxtImageDataset(args.art_txt, transform=preprocess)
    dsB = TxtImageDataset(args.clipart_txt, transform=preprocess)
    dlA = DataLoader(dsA, batch_size=args.batch_size, shuffle=True,
                     num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dlB = DataLoader(dsB, batch_size=args.batch_size, shuffle=True,
                     num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # eval loaders (no shuffle)
    dlA_eval = DataLoader(dsA, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    dlB_eval = DataLoader(dsB, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # noun set: Office-Home class names as default unified nouns
    noun_texts = [c.replace("_", " ") for c in OFFICEHOME_CLASSES]
    with torch.no_grad():
        noun_text_emb = clip.encode_text(noun_texts).to(device)

    # projectors g_A, g_B (trainable)
    in_dim = noun_text_emb.size(1) * 2  # concat(v,t) => 2*Dclip
    gA = MLPProjector(in_dim, out_dim=args.proj_dim, hidden_dim=args.proj_hidden, num_layers=args.proj_layers).to(device)
    gB = MLPProjector(in_dim, out_dim=args.proj_dim, hidden_dim=args.proj_hidden, num_layers=args.proj_layers).to(device)

    opt = torch.optim.AdamW(list(gA.parameters()) + list(gB.parameters()),
                            lr=args.lr, weight_decay=args.wd)

    # prototypes (initialized later)
    protoA_z = None  # (K,Dproj)
    protoB_z = None
    protoA_img = None  # (K,Dclip) vbar
    protoB_img = None
    protoA_txt = None  # (K,Dclip) tbar
    protoB_txt = None

    def topk_candidates(z, proto_z, k):
        # z:(B,D), proto:(K,D) => indices (B,k)
        sim = z @ proto_z.t()
        return sim.topk(k=min(k, proto_z.size(0)), dim=-1).indices

    # training loop
    for ep in range(1, args.epochs + 1):
        # ---- update prototypes periodically ----
        if protoA_z is None or (ep - 1) % args.update_proto_every == 0:
            logger.log(f"Epoch {ep}: updating prototypes via KMeans ...")
            zA_all, vA_all, tA_all, yA_all = extract_all_features(dlA_eval, clip, gA, noun_text_emb, args.top_m_noun, device)
            zB_all, vB_all, tB_all, yB_all = extract_all_features(dlB_eval, clip, gB, noun_text_emb, args.top_m_noun, device)

            protoA_z, assignA = run_kmeans(zA_all, K=args.K, seed=args.seed + ep)
            protoB_z, assignB = run_kmeans(zB_all, K=args.K, seed=args.seed + ep)

            # compute prototype clip means (Eq.(6) spirit)
            protoA_img = torch.zeros(args.K, vA_all.size(1))
            protoA_txt = torch.zeros(args.K, tA_all.size(1))
            protoB_img = torch.zeros(args.K, vB_all.size(1))
            protoB_txt = torch.zeros(args.K, tB_all.size(1))

            for k in range(args.K):
                idx = (assignA == k).nonzero()[0] if (assignA == k).any() else None
                if idx is not None:
                    vv = vA_all[idx].mean(dim=0)
                    tt = tA_all[idx].mean(dim=0)
                    protoA_img[k] = vv / vv.norm().clamp_min(1e-6)
                    protoA_txt[k] = tt / tt.norm().clamp_min(1e-6)

                idx = (assignB == k).nonzero()[0] if (assignB == k).any() else None
                if idx is not None:
                    vv = vB_all[idx].mean(dim=0)
                    tt = tB_all[idx].mean(dim=0)
                    protoB_img[k] = vv / vv.norm().clamp_min(1e-6)
                    protoB_txt[k] = tt / tt.norm().clamp_min(1e-6)

            protoA_z = protoA_z.to(device)
            protoB_z = protoB_z.to(device)
            protoA_img = protoA_img.to(device)
            protoB_img = protoB_img.to(device)
            protoA_txt = protoA_txt.to(device)
            protoB_txt = protoB_txt.to(device)

        # ---- train for one epoch ----
        gA.train(); gB.train()
        loss_meter = AvgMeter()
        itA = iter(dlA)
        itB = iter(dlB)
        steps = min(len(dlA), len(dlB))

        pbar = tqdm(range(steps), desc=f"Train Ep{ep}", leave=False)
        for _ in pbar:
            imgA, _, _ = next(itA)
            imgB, _, _ = next(itB)
            imgA = imgA.to(device, non_blocking=True)
            imgB = imgB.to(device, non_blocking=True)

            # semantic-enhanced
            vA, tA, zA = build_semantic_enhanced(clip, imgA, noun_text_emb, top_m=args.top_m_noun)
            vB, tB, zB = build_semantic_enhanced(clip, imgB, noun_text_emb, top_m=args.top_m_noun)

            zA_t = gA(zA)
            zB_t = gB(zB)

            # candidate prototypes: top-k by projected similarity in each domain
            idxA_in = topk_candidates(zA_t, protoA_z, args.k_cand)
            idxA_cross = topk_candidates(zA_t, protoB_z, args.k_cand)
            idxB_in = topk_candidates(zB_t, protoB_z, args.k_cand)
            idxB_cross = topk_candidates(zB_t, protoA_z, args.k_cand)

            # build external priors p(x,c) over candidate sets (Eq.(7)-(8))
            # A intra candidates
            pA_in = build_external_prior(
                vA, tA,
                protoA_img[idxA_in], protoA_txt[idxA_in],
                tau_p=args.tau_p
            )
            # A cross candidates (A->B)
            pA_cr = build_external_prior(
                vA, tA,
                protoB_img[idxA_cross], protoB_txt[idxA_cross],
                tau_p=args.tau_p
            )
            # B intra
            pB_in = build_external_prior(
                vB, tB,
                protoB_img[idxB_in], protoB_txt[idxB_in],
                tau_p=args.tau_p
            )
            # B cross (B->A)
            pB_cr = build_external_prior(
                vB, tB,
                protoA_img[idxB_cross], protoA_txt[idxB_cross],
                tau_p=args.tau_p
            )

            # select constraint losses (Eq.(14)-(17))
            # we need proto_z candidate tensors aligned with p_ext shape
            protoA_in_z = protoA_z[idxA_in]
            protoB_cr_z = protoB_z[idxA_cross]
            protoB_in_z = protoB_z[idxB_in]
            protoA_cr_z = protoA_z[idxB_cross]

            # reshape candidates to (B,k,D)
            # convert to batched: we compute loss per sample by flattening
            # simplest: loop-batch with gather per sample (still fast for k<=10, bs<=64)
            def batch_scl(z_t, proto_z_batched, p_ext):
                B, k, D = proto_z_batched.shape
                loss = 0.0
                for i in range(B):
                    loss = loss + selective_constraint_loss(
                        z_t[i:i+1],
                        proto_z_batched[i],
                        p_ext[i:i+1],
                        tau_q=args.tau_q,
                        delta_pos=args.delta_pos,
                        delta_neg=args.delta_neg
                    )
                return loss / B

            L_A_in = batch_scl(zA_t, protoA_in_z, pA_in)
            L_A_cr = batch_scl(zA_t, protoB_cr_z, pA_cr)
            L_B_in = batch_scl(zB_t, protoB_in_z, pB_in)
            L_B_cr = batch_scl(zB_t, protoA_cr_z, pB_cr)

            L_inter = 0.5 * (L_A_cr + L_B_cr)
            L_intra = 0.5 * (L_A_in + L_B_in)
            L_scl = L_inter + args.alpha * L_intra

            # PGR per domain (Eq.(18)-(22))
            L_pgr = 0.5 * (
                pgr_loss(zA_t, protoA_z, tau_in=args.tau_in, m=args.m_comp, r=args.r_push) +
                pgr_loss(zB_t, protoB_z, tau_in=args.tau_in, m=args.m_comp, r=args.r_push)
            )

            loss = L_scl + args.lam * L_pgr

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_meter.update(loss.item(), 1)
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        logger.log(f"Epoch {ep}: train loss={loss_meter.avg:.4f}")

        # ---- eval P@K (Art->Clipart and Clipart->Art) ----
        gA.eval(); gB.eval()
        zA_all, _, _, yA_all = extract_all_features(dlA_eval, clip, gA, noun_text_emb, args.top_m_noun, device)
        zB_all, _, _, yB_all = extract_all_features(dlB_eval, clip, gB, noun_text_emb, args.top_m_noun, device)

        zA_all = zA_all.to(device); zB_all = zB_all.to(device)
        yA_all = yA_all.to(device); yB_all = yB_all.to(device)

        res_A2B = precision_at_k(zA_all, yA_all, zB_all, yB_all, ks=(1,5,15))
        res_B2A = precision_at_k(zB_all, yB_all, zA_all, yA_all, ks=(1,5,15))

        logger.log(f"Eval Art->Clipart: " + ", ".join([f"P@{k}={res_A2B[k]:.2f}" for k in res_A2B]))
        logger.log(f"Eval Clipart->Art: " + ", ".join([f"P@{k}={res_B2A[k]:.2f}" for k in res_B2A]))

        # save checkpoints
        ckpt = {
            "epoch": ep,
            "gA": gA.state_dict(),
            "gB": gB.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.logdir, f"ckpt_ep{ep}.pt"))

    logger.log("Done.")

if __name__ == "__main__":
    main()
