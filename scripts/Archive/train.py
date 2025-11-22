from __future__ import annotations
import os, argparse, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from ocr.data.datasets import OCRDataset, ocr_collate_fn
from ocr.models.transformer_ocr import OCRTransformer
from ocr.utils.tokenizer import Tokenizer
from ocr.utils.schedulers import CosineWithWarmup
from ocr.data.datasets import OCRDataset, ocr_collate_fn # 确保 ocr_collate_fn 被导入

class Collator:
    """
    一个可 pickle 的 callable 类，用于替换 lambda collate_fn。
    它在初始化时存储 pad_id。
    """
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        return ocr_collate_fn(batch, self.pad_id)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ann", type=str, required=True)
    parser.add_argument("--val_ann", type=str, default=None)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="vit_base_patch14_dinov2.lvd142m")

    parser.add_argument("--img_height", type=int, default=224)
    parser.add_argument("--max_width", type=int, default=1024)
    parser.add_argument("--patch_size", type=int, default=14)

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dec_layers", type=int, default=6)
    parser.add_argument("--ffn_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze_stages", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=1)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=80000)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.exp_dir, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer(args.vocab)
    pad_id = tokenizer.pad_id
    
    # 1. 创建 collator 实例
    collator_instance = Collator(pad_id)

    train_ds = OCRDataset(args.train_ann, tokenizer, img_height=args.img_height, max_width=args.max_width,
                          patch_size=args.patch_size, is_train=True) #
    # 2. 在 DataLoader 中使用该实例
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          pin_memory=True, collate_fn=collator_instance) #

    if args.val_ann: #
        val_ds = OCRDataset(args.val_ann, tokenizer, img_height=args.img_height, max_width=args.max_width,
                            patch_size=args.patch_size, is_train=False) #
        # 3. 在验证集的 DataLoader 中也使用该实例
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=collator_instance) #
    else:
        val_dl = None

    model = OCRTransformer(
        vocab_size=len(tokenizer),
        model_name=args.model_name,
        d_model=args.d_model,
        nhead=args.nhead,
        dec_layers=args.dec_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        freeze_stages=args.freeze_stages
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params/1e6:.2f}M")

    # Separate LR for backbone (smaller) vs decoder/head (larger)
    backbone_params = list(model.encoder.parameters())
    decoder_params = [p for n,p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": decoder_params, "lr": args.lr},
    ], weight_decay=args.wd)

    scheduler = CosineWithWarmup(opt, warmup_steps=args.warmup_steps, max_steps=args.max_steps, min_lr_ratio=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    best_cer = 1e9
    global_step = 0
    epoch = 0

    while global_step < args.max_steps:
        epoch += 1
        print(f"\n=== Epoch {epoch} ===")
        for i, batch in enumerate(train_dl):
            model.train()
            images = batch["images"].to(device, non_blocking=True)
            tgt = batch["tgt"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.split(':')[0], dtype=torch.float16):
                out = model(images, tgt[:, :-1])  # teacher forcing
                logits = out["logits"]
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt[:, 1:].reshape(-1),
                    ignore_index=pad_id,
                    label_smoothing=0.1,
                )

            scaler.scale(loss).backward()
            if (global_step + 1) % args.accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                scheduler.step()

            global_step += 1
            if global_step % 100 == 0:
                print(f"step {global_step}/{args.max_steps} | loss {loss.item():.4f}")

            if val_dl and (global_step % args.eval_every == 0):
                print("Running validation...")
                from ocr.train_utils import evaluate
                metrics = evaluate(model, val_dl, tokenizer, device=device, max_eval_batches=50)
                print(f"Val CER {metrics['cer']:.4f} | ACC {metrics['acc']:.4f}")
                if metrics['cer'] < best_cer:
                    best_cer = metrics['cer']
                    ckpt_path = os.path.join(args.exp_dir, "best.pt")
                    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
                    print(f"  Saved best to {ckpt_path}")

            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.exp_dir, f"step_{global_step}.pt")
                torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
                print(f"  Saved checkpoint to {ckpt_path}")

            if global_step >= args.max_steps:
                break

    ckpt_path = os.path.join(args.exp_dir, f"final.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Saved final to {ckpt_path}")

if __name__ == "__main__":
    main()
