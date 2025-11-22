from __future__ import annotations
import torch
from .utils.metrics import cer, acc

def evaluate(model, dl, tokenizer, device="cuda", max_eval_batches=None):
    """Simple evaluation with beam search (beam=5)."""
    model.eval()
    tot_cer, tot_acc, n = 0.0, 0.0, 0
    from .utils.decoding import beam_search
    with torch.no_grad():
        for i, batch in enumerate(dl):
            images = batch["images"].to(device)
            tgt = batch["tgt"].to(device)
            seqs = beam_search(model, images, tokenizer, beam_size=5, max_len=tgt.size(1), device=device)
            for b in range(images.size(0)):
                pred = tokenizer.decode(seqs[b], remove_special=True)
                gt = batch["texts"][b]
                tot_cer += cer(pred, gt)
                tot_acc += acc(pred, gt)
                n += 1
            if max_eval_batches and i + 1 >= max_eval_batches:
                break
    model.train()
    return {"cer": tot_cer / max(1, n), "acc": tot_acc / max(1, n)}
