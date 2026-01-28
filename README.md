Input Format

This implementation expects two TXT files, one per domain.

Each line in the TXT file should contain:
/absolute/path/to/image.jpg label

Basic Command (Art ↔ Clipart):
python train.py \
  --art_txt ./dataset/Art.txt \
  --clipart_txt ./dataset/Clipart.txt \
  --logdir runs/art_clipart \
  --epochs 200 \
  --batch_size 32 \
  --K 65 \
  --k_cand 10 \
  --delta_pos 0.85 \
  --delta_neg 0.10 \
  --alpha 1.0 \
  --lam 0.5
