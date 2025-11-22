# OCR Dataset Generator
This repository is built upon the original ImageNet dataset, augmented with synthetically overlaid text to generate images for OCR tasks.

## Install
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python synth_text_dataset_en.py \
  --backgrounds /path/to/imagenet_subset \
  --fonts /path/to/fonts_dir \
  --out_dir ./out \
  --num_images 1000 \
  --res 1024 1024 \
  --jsonl
```
- If `--backgrounds` or `--fonts` is empty, the script falls back to gradient backgrounds and PIL's default font so you can quickly validate the pipeline.
- Output:
```
out/
  images/<split>_0000000.jpg
  labels/<split>_0000000.json   # or annotations_<split>.jsonl
```
- JSON annotation fields cover L1/L2/L3 as described in the script.
