# OCR Transformer (DINOv2 + Encoder-Decoder)

This repo implements OCR using a **timm DINOv2 ViT** visual encoder and a **Transformer Decoder** (autoregressive).
- ✅ Pure PyTorch + timm
- ✅ Variable-width inputs (fixed height, aspect-preserving resize; pad to ViT patch multiple)
- ✅ 2D visual tokens → text via cross-attention + beam search
- ✅ **No curriculum learning** and **no fallback mechanisms** (no dictionary constraints, no second-pass decoders)

> Annotation format: `annotations.tsv` where each line is `image_path<TAB>text` (UTF-8).

## Setup (Conda)

### Option A) GPU (NVIDIA CUDA 12.x)
```bash
conda env create -f environment.gpu.yml
conda activate ocr-transformer
```

### Option B) CPU-only (or macOS / no NVIDIA GPU)
```bash
conda env create -f environment.cpu.yml
conda activate ocr-transformer
```

> Manual install (alternative):
> ```bash
> conda create -n ocr-transformer python=3.10 -y
> conda activate ocr-transformer
> # GPU (Linux + NVIDIA):
> conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
> # CPU only:
> # conda install pytorch torchvision cpuonly -c pytorch
> pip install -r requirements.txt
> ```

## Quickstart

1) **Build or provide a vocabulary**
- If you already have `vocab.txt` (one token per line; include a space if needed), place it in the project root.
- Otherwise build from annotations:
```bash
python scripts/build_vocab.py --ann annotations.tsv --out vocab.txt --min_freq 1
```
Special tokens are added by the code: `[PAD], [BOS], [EOS], [UNK]`.

2) **Train**
```bash
python train.py   --train_ann annotations.tsv   --val_ann val.tsv   --vocab vocab.txt   --exp_dir runs/exp1   --img_height 224   --max_width 1024   --model_name vit_base_patch14_dinov2.lvd142m   --d_model 512 --nhead 8 --dec_layers 6   --batch_size 16 --accum_steps 1   --lr 2e-4 --wd 0.01 --warmup_steps 2000 --max_steps 80000   --freeze_stages 0   --num_workers 8
```
- Two parameter groups are used: backbone with LR×0.1, decoder with LR.
- Scheduler: warmup + cosine.
- Mixed precision (AMP) on CUDA is enabled by default.
- Validation uses beam search (beam=5) without any external LM or fallback.

3) **Inference**
- Single image:
```bash
python infer.py   --img path/to/img.jpg   --vocab vocab.txt   --ckpt runs/exp1/best.pt   --model_name vit_base_patch14_dinov2.lvd142m
```
- A folder:
```bash
python infer.py   --folder path/to/images   --pattern "*.png"   --vocab vocab.txt   --ckpt runs/exp1/best.pt   --model_name vit_base_patch14_dinov2.lvd142m
```

## Design notes
- **Visual encoder**: timm DINOv2 ViT (e.g., `vit_base_patch14_dinov2.lvd142m`). We reuse its `patch_embed/pos_embed/blocks/norm` and interpolate positional embeddings to match arbitrary H×W (multiples of the patch size).
- **Tokens**: last-layer patch tokens **without CLS** → `[B, T, C]` as memory for the decoder’s cross-attention.
- **Text decoder**: standard `nn.TransformerDecoder` (teacher forcing training; beam search at inference).
- **Image size**: resize input to a fixed height (e.g., 224), keep aspect ratio, then **pad** to multiples of patch size (14 for ViT-B/14). The width is clipped at `--max_width`.

## Metrics
- Character Error Rate (CER) and whole-string Accuracy (ACC).

## License
MIT
