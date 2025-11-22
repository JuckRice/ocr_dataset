from __future__ import annotations
import io, argparse, collections

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=str, required=True, help="annotations.tsv")
    ap.add_argument("--out", type=str, required=True, help="vocab.txt")
    ap.add_argument("--min_freq", type=int, default=1)
    args = ap.parse_args()

    cnt = collections.Counter()
    with io.open(args.ann, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" in line:
                _, t = line.split("\t", 1)
            elif "\t" not in line:
                # Prefer real TAB if possible; otherwise try comma
                if "\t" not in line and "," in line:
                    _, t = line.split(",", 1)
                else:
                    parts = line.split()
                    _, t = parts[0], "".join(parts[1:]) if len(parts) > 1 else ""
            for ch in t:
                cnt[ch] += 1

    tokens = [ch for ch, n in cnt.most_common() if n >= args.min_freq]
    with io.open(args.out, "w", encoding="utf-8") as f:
        for t in tokens:
            f.write(t + "\n")
    print(f"saved {len(tokens)} tokens to {args.out}")

if __name__ == "__main__":
    main()
