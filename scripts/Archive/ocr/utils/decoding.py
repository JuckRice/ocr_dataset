from __future__ import annotations
import torch
from typing import List

@torch.no_grad()
def beam_search(model, images, tokenizer, beam_size=5, max_len=128, device="cuda"):
    """
    Plain autoregressive beam search (no external LM, no fallbacks).
    The model must implement forward(images, tgt_tokens) -> {"logits": [B, T, V]}.
    """
    model.eval()
    images = images.to(device, non_blocking=True)
    B = images.size(0)

    bos = tokenizer.bos_id
    eos = tokenizer.eos_id

    # Per-batch beams: list of (score, seq_tensor)
    beam_seqs = [[(0.0, torch.tensor([bos], device=device, dtype=torch.long))] for _ in range(B)]
    finished = [[] for _ in range(B)]

    for _ in range(max_len):
        new_beam_seqs = []
        for b in range(B):
            candidates = []
            for score, seq in beam_seqs[b]:
                if seq[-1].item() == eos:
                    candidates.append((score, seq))
                    continue
                # One decoding step
                logits = model(images[b:b+1], seq.unsqueeze(0))["logits"]  # [1, L, V]
                log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)    # [V]
                topk = torch.topk(log_probs, k=beam_size)
                for i in range(beam_size):
                    tok = topk.indices[i].unsqueeze(0)
                    sc = score + topk.values[i].item()
                    candidates.append((sc, torch.cat([seq, tok], dim=0)))
            # Keep top-k
            candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]
            # Separate finished
            still_open = []
            for sc, sq in candidates:
                if sq[-1].item() == eos:
                    finished[b].append((sc, sq))
                else:
                    still_open.append((sc, sq))
            if len(still_open) == 0:
                still_open = candidates[:beam_size]
            new_beam_seqs.append(still_open)
        beam_seqs = new_beam_seqs
        if all(len(f) >= beam_size for f in finished):
            break

    # Finalize output
    results: List[List[int]] = []
    for b in range(B):
        if len(finished[b]) == 0:
            best = max(beam_seqs[b], key=lambda x: x[0])[1]
        else:
            best = max(finished[b], key=lambda x: x[0])[1]
        results.append(best.tolist())
    return results
