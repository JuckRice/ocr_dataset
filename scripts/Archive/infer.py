from __future__ import annotations
import os, glob, argparse
import torch
from PIL import Image
from ocr.models.transformer_ocr import OCRTransformer
from ocr.utils.tokenizer import Tokenizer
from ocr.data.transforms import default_val_transform

@torch.no_grad()
def infer_one(model, img_path, tokenizer, device, img_height=224, max_width=1024, patch_size=14, beam_size=5, max_len=128):
    img = Image.open(img_path).convert("RGB")
    x = default_val_transform(img, img_height=img_height, max_width=max_width, patch_size=patch_size)
    x = x.unsqueeze(0).to(device)
    from ocr.utils.decoding import beam_search
    seq = beam_search(model, x, tokenizer, beam_size=beam_size, max_len=max_len, device=device)[0]
    text = tokenizer.decode(seq, remove_special=True)
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, default=None)
    ap.add_argument("--folder", type=str, default=None)
    ap.add_argument("--pattern", type=str, default="*.jpg")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--vocab", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--dec_layers", type=int, default=6)
    ap.add_argument("--ffn_dim", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--freeze_stages", type=int, default=0)
    ap.add_argument("--img_height", type=int, default=224)
    ap.add_argument("--max_width", type=int, default=1024)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer(args.vocab)
    model = OCRTransformer(
        vocab_size=len(tokenizer),
        model_name=args.model_name,
        d_model=args.d_model, nhead=args.nhead, dec_layers=args.dec_layers,
        dim_feedforward=args.ffn_dim, dropout=args.dropout, freeze_stages=args.freeze_stages
    )
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device).eval()

    if args.img:
        text = infer_one(model, args.img, tokenizer, device, img_height=args.img_height, max_width=args.max_width, patch_size=args.patch_size, beam_size=args.beam_size, max_len=args.max_len)
        print(f"{args.img}\t{text}")
    elif args.folder:
        paths = sorted(glob.glob(os.path.join(args.folder, args.pattern)))
        for p in paths:
            text = infer_one(model, p, tokenizer, device, img_height=args.img_height, max_width=args.max_width, patch_size=args.patch_size, beam_size=args.beam_size, max_len=args.max_len)
            print(f"{p}\t{text}")
    else:
        raise ValueError("Provide --img or --folder")

if __name__ == "__main__":
    main()
