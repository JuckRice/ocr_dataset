import json
import argparse
from pathlib import Path
from tqdm import tqdm

def convert_jsonl_to_tsv(jsonl_path, tsv_out_path, img_root_prefix):
    """
    Converts a .jsonl annotation file (from synth_text_dataset.py)
    to the .tsv format required by train.py (image_path<TAB>text).
    """
    count = 0
    # Determine image prefix
    img_prefix = img_root_prefix if img_root_prefix else "images"
    
    with open(jsonl_path, 'r', encoding='utf-8') as f_in, \
         open(tsv_out_path, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc=f"Converting {jsonl_path.name}"):
            try:
                ann = json.loads(line)
                image_id = ann['image_id']
                
                # e.g., "images/train_0000001.jpg"
                img_path = f"{img_prefix}/{image_id}.jpg"

                for instance in ann.get('instances', []):
                    text = instance.get('text', '')
                    if text: # text is not empty
                        # TSV: image_path<TAB>text
                        f_out.write(f"{img_path}\t{text}\n")
                        count += 1
                        
            except json.JSONDecodeError:
                print(f"[Warn] Skipping invalid JSON line: {line[:50]}...")
                
    print(f"Conversion complete. Wrote {count} lines to {tsv_out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL to TSV format.")
    parser.add_argument('--jsonl_path', type=Path, required=True, help='Path to the input .jsonl file.')
    parser.add_argument('--out_path', type=Path, required=True, help='Path for the output .tsv file.')
    parser.add_argument('--img_prefix', type=str, default='images', help='Path prefix for images (relative to the tsv file location).')
    args = parser.parse_args()

    convert_jsonl_to_tsv(args.jsonl_path, args.out_path, args.img_prefix)