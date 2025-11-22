import editdistance

def cer(pred: str, tgt: str) -> float:
    """Character Error Rate."""
    if len(tgt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return editdistance.eval(pred, tgt) / len(tgt)

def acc(pred: str, tgt: str) -> float:
    """Exact match accuracy (whole string)."""
    return 1.0 if pred == tgt else 0.0
