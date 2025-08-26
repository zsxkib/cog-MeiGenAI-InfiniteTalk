
# InfiniteTalk setup on a fresh CUDA box (conda)

These are the exact steps we used to get InfiniteTalk running via the command line, plus the small code tweaks needed for current library versions.

## 1) Create and activate a conda env (Python 3.10)

```bash
conda create -n infinitetalk python=3.10 -y
conda activate infinitetalk
```

## 2) Install PyTorch + torchvision + torchaudio and xformers (CUDA 12.1 wheels)

```bash
pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
```

Notes:
- We did NOT require flash-attn; the code will fall back to scaled_dot_product_attention.
- If you try to install flash-attn and hit CUDA/nvcc issues, skip it. It’s optional.

## 3) Optional: CUDA dev tools (only if you attempt building flash-attn)

```bash
sudo apt update
sudo apt install -y nvidia-cuda-toolkit
# If needed: export CUDA_HOME accordingly, e.g.:
# export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
```

## 4) Project Python dependencies

```bash
# Core repo deps
pip install -r requirements.txt

# Additional audio/video deps we used
conda install -c conda-forge -y librosa ffmpeg
pip install soxr
```

## 5) Hugging Face model weights

We used huggingface-cli to fetch required weights:

```bash
# Make a local weights dir
mkdir -p weights

# Wan base model
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P

# Audio encoder
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
# Also fetch the safetensors variant used by the code
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base

# InfiniteTalk weights
huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk
```

## 6) Small code fixes we applied

- Set Wav2Vec2 to use eager attention (newer Transformers default to sdpa which forbids `output_attentions=True`). We applied this two places:
  - In `generate_infinitetalk.py` inside `custom_init(...)`:
    ```python
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True, attn_implementation="eager").to(device)
    ```
  - In `app.py` where Wav2Vec2Model is loaded for the Gradio app:
    ```python
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True, attn_implementation="eager").to(device)
    ```

- Provide a safe fallback when FlashAttention is unavailable. In `wan/modules/attention.py`, the internal function now falls back to `torch.nn.functional.scaled_dot_product_attention(...)` instead of asserting FA is present.

- Install `soxr` to satisfy `transformers.audio_utils` import.

## 7) Running examples

Image-to-video example (lower VRAM):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python generate_infinitetalk.py \
  --ckpt_dir weights/Wan2.1-I2V-14B-480P \
  --wav2vec_dir weights/chinese-wav2vec2-base \
  --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
  --input_json examples/single_example_image.json \
  --size infinitetalk-480 \
  --sample_steps 40 \
  --num_persistent_param_in_dit 0 \
  --mode streaming \
  --motion_frame 9 \
  --save_file infinitetalk_res
```

Gradio app:

```bash
# app.py already uses eager attention and will launch with share=True if configured.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python app.py \
  --ckpt_dir weights/Wan2.1-I2V-14B-480P \
  --wav2vec_dir weights/chinese-wav2vec2-base \
  --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
  --num_persistent_param_in_dit 0 \
  --motion_frame 9
```

## 8) VRAM tips / OOM mitigation

- Use `--size infinitetalk-480`.
- Use `--num_persistent_param_in_dit 0`.
- Reduce `--sample_steps`.
- Keep `offload_model=True` (defaulted to True in single-GPU mode).
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.
- Consider quantized weights (`--quant fp8` with the provided quant models).

## 9) H100 note

On an H100 (CUDA 12.2/12.4), the same environment works well with the cu121 wheels above. If you prefer, bump torch/torchvision/torchaudio to matching newer versions with the corresponding CUDA wheels.
