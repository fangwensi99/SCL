import torch

class CLIPBackbone(torch.nn.Module):
    """
    Frozen CLIP backbone wrapper:
      - encode_image(images) -> (B, D)
      - encode_text(tokenized) -> (N, D)
    Supports open_clip first, falls back to clip if available.
    """
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device="cuda"):
        super().__init__()
        self.device = device

        self.backend = "open_clip"
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            self.model = model.to(device).eval()
            self.preprocess = preprocess
            self.tokenizer = tokenizer
        except Exception as e:
            self.backend = "clip"
            import clip
            model, preprocess = clip.load(model_name, device=device, jit=False)
            self.model = model.eval()
            self.preprocess = preprocess
            self.tokenizer = clip.tokenize

        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        # images: already preprocessed tensor (B,3,H,W)
        feat = self.model.encode_image(images)
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return feat

    @torch.no_grad()
    def encode_text(self, texts) -> torch.Tensor:
        # texts: list[str]
        tokens = self.tokenizer(texts).to(self.device)
        feat = self.model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return feat
