
"""Cog predictor for InfiniteTalk (audio-driven video generation).

This wraps the existing generate() entrypoint in generate_infinitetalk.py.
It will download the required HF weights at runtime, prepare a small
input JSON, and return a generated MP4.
"""

import os
import json
from argparse import Namespace
from typing import Optional
from cog import BasePredictor, Input, Path
from huggingface_hub import snapshot_download, hf_hub_download


# Reduce CUDA fragmentation a bit
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

WEIGHTS_DIR = "weights"
WAN_DIR = os.path.join(WEIGHTS_DIR, "Wan2.1-I2V-14B-480P")
WAV2VEC_DIR = os.path.join(WEIGHTS_DIR, "chinese-wav2vec2-base")
INFT_DIR = os.path.join(WEIGHTS_DIR, "InfiniteTalk")


class Predictor(BasePredictor):
    def setup(self) -> None:
        os.makedirs(WEIGHTS_DIR, exist_ok=True)

        # Download Wan base model
        if not os.path.exists(WAN_DIR):
            snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_dir=WAN_DIR)

        # wav2vec base
        if not os.path.exists(WAV2VEC_DIR):
            snapshot_download("TencentGameMate/chinese-wav2vec2-base", local_dir=WAV2VEC_DIR)
        # also fetch model.safetensors from PR revision used by the repo, if not present
        safepath = os.path.join(WAV2VEC_DIR, "model.safetensors")
        if not os.path.exists(safepath):
            try:
                hf_hub_download(
                    repo_id="TencentGameMate/chinese-wav2vec2-base",
                    filename="model.safetensors",
                    revision="refs/pr/1",
                    local_dir=WAV2VEC_DIR,
                    local_dir_use_symlinks=False,
                )
            except Exception as e:
                # Not fatal – base bin will still work; generate script will pick what's available
                print(f"[warn] could not fetch wav2vec model.safetensors PR: {e}")

        # InfiniteTalk
        if not os.path.exists(INFT_DIR):
            snapshot_download("MeiGen-AI/InfiniteTalk", local_dir=INFT_DIR)

    def predict(
        self,
        prompt: str = Input(description="Text prompt"),
        cond_video: Path = Input(description="Image or video to dub (image-to-video or video-to-video)"),
        cond_audio: Path = Input(description="Audio file (wav/mp3) or a video (audio will be extracted)"),
        size: str = Input(description="Resolution preset", choices=["infinitetalk-480", "infinitetalk-720"], default="infinitetalk-480"),
        sample_steps: int = Input(description="Sampling steps", ge=4, le=60, default=40),
        mode: str = Input(description="Generation mode", choices=["streaming", "clip"], default="streaming"),
        motion_frame: int = Input(description="Driven frame length for long video", ge=1, default=9),
        low_vram: bool = Input(description="Enable low-VRAM mode (keeps fewer params on GPU)", default=True),
        offload_model: bool = Input(description="Offload models to CPU between steps", default=True),
        seed: Optional[int] = Input(description="Random seed (-1 = random)", default=42),
    ) -> Path:
        # Prepare input json for the generate() wrapper
        input_json = {
            "prompt": prompt,
            "cond_video": str(cond_video),
            "cond_audio": {"person1": str(cond_audio)},
        }
        input_json_path = "/tmp/infinitetalk_input.json"
        with open(input_json_path, "w") as f:
            json.dump(input_json, f)

        # Build args namespace for generate()
        save_stem = "/tmp/infinitetalk_out"
        args = Namespace(
            task="infinitetalk-14B",
            size=size,
            frame_num=81,
            max_frame_num=1000,
            ckpt_dir=WAN_DIR,
            infinitetalk_dir=os.path.join(INFT_DIR, "single", "infinitetalk.safetensors"),
            quant_dir=None,
            wav2vec_dir=WAV2VEC_DIR,
            dit_path=None,
            lora_dir=None,
            lora_scale=[1.2],
            offload_model=offload_model,
            ulysses_size=1,
            ring_size=1,
            t5_fsdp=False,
            t5_cpu=False,
            dit_fsdp=False,
            save_file=save_stem,
            audio_save_dir="save_audio",
            base_seed=seed if seed is not None else 42,
            input_json=input_json_path,
            motion_frame=motion_frame,
            mode=mode,
            sample_steps=sample_steps,
            sample_shift=None,
            sample_text_guide_scale=5.0,
            sample_audio_guide_scale=4.0,
            num_persistent_param_in_dit=0 if low_vram else None,
            audio_mode="localfile",
            use_teacache=False,
            teacache_thresh=0.2,
            use_apg=False,
            apg_momentum=-0.75,
            apg_norm_threshold=55,
            color_correction_strength=1.0,
            scene_seg=False,
            quant=None,
        )

        # Import here to keep Cog import-time light
        from generate_infinitetalk import generate

        generate(args)

        out_path = Path(save_stem + ".mp4")
        if not out_path.exists():
            raise RuntimeError("Generation finished but output video was not created.")
        return out_path
