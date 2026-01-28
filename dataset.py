import os
from PIL import Image
from torch.utils.data import Dataset

class TxtImageDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.txt_path = txt_path
        self.transform = transform
        self.samples = []  # (path, label)
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                img_path = parts[0]
                label = int(parts[1]) if len(parts) > 1 else -1
                self.samples.append((img_path, label))

        # 简单检查
        missing = sum(0 if os.path.exists(p) else 1 for p, _ in self.samples[:200])
        if missing > 0:
            print(f"[Warn] {txt_path}: first 200 samples missing files = {missing}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y, path
