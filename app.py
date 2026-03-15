import os
import random
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from datasets import load_dataset

MODEL_NAME = os.environ.get("SD_MODEL", "runwayml/stable-diffusion-v1-5")
LORA_DIR = os.environ.get("LORA_DIR", "./lora")
LORA_WEIGHT_NAME = os.environ.get("LORA_WEIGHT_NAME", "pytorch_lora_weights.safetensors")
DATASET_NAME = os.environ.get("DATASET_NAME", "Skiittoo/cartoon-faces")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pipe = None
_caption_dataset = None
_caption_column = None


def _get_caption_dataset():
    global _caption_dataset, _caption_column
    if _caption_dataset is not None:
        return _caption_dataset, _caption_column
    ds = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)
    cols = ds.column_names
    for name in ("text", "caption", "prompt", "label", "captions"):
        if name in cols:
            _caption_column = name
            break
    if _caption_column is None:
        _caption_column = next((c for c in cols if c != "image"), cols[0] if cols else None)
    _caption_dataset = ds
    return _caption_dataset, _caption_column


def get_random_caption():
    try:
        ds, col = _get_caption_dataset()
        row = random.choice(ds)
        text = row.get(col)
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        caption = str(text).strip() or "cartoon face"
        for img_key in ("image", "images", "img"):
            img = row.get(img_key)
            if img is not None:
                if hasattr(img, "copy"):
                    return caption, img
                if isinstance(img, (list, tuple)) and img and hasattr(img[0], "copy"):
                    return caption, img[0]
        return caption, None
    except Exception as e:
        return f"(ошибка датасета: {e})", None


def load_pipeline():
    global pipe
    if pipe is not None:
        return
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _get_lora_weight_name():
    p = os.path.join(LORA_DIR, LORA_WEIGHT_NAME)
    return LORA_WEIGHT_NAME if os.path.isfile(p) else None


def run_both(prompt: str, num_inference_steps: int, guidance_scale: float, base_seed: int):
    if not prompt or not prompt.strip():
        return None, None
    load_pipeline()
    try:
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
        gen_base = torch.Generator(device=DEVICE).manual_seed(base_seed + 0)
        img_base = pipe(
            prompt=prompt.strip(),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=gen_base,
        ).images[0]

        if not os.path.isdir(LORA_DIR):
            return img_base, None
        pipe.load_lora_weights(LORA_DIR, weight_name=_get_lora_weight_name())
        gen_lora = torch.Generator(device=DEVICE).manual_seed(base_seed + 1)
        img_lora = pipe(
            prompt=prompt.strip(),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=gen_lora,
        ).images[0]

        return img_base, img_lora
    except Exception:
        return None, None


def build_ui():
    with gr.Blocks(title="T2I Base + LoRA") as demo:
        gr.Markdown("## Text-to-Image: базовая модель и LoRA")
        prompt = gr.Textbox(
            label="Промпт",
            placeholder="описание изображения или нажмите «Взять из датасета»...",
            lines=2,
        )
        with gr.Row():
            num_inference_steps = gr.Number(label="num_inference_steps", value=20, precision=0, minimum=1, maximum=100)
            guidance_scale = gr.Number(label="guidance_scale", value=4.0, minimum=1.0, maximum=20.0)
            base_seed = gr.Number(label="Seed (base_seed)", value=42, precision=0, minimum=0)
        with gr.Row():
            random_btn = gr.Button("Взять из датасета")
            run_btn = gr.Button("Отправить", variant="primary")
        with gr.Row():
            out_original = gr.Image(label="Оригинал из датасета")
            out_base = gr.Image(label="Базовая модель")
            out_lora = gr.Image(label="С LoRA")

        random_btn.click(fn=get_random_caption, inputs=[], outputs=[prompt, out_original])
        run_btn.click(
            fn=run_both,
            inputs=[prompt, num_inference_steps, guidance_scale, base_seed],
            outputs=[out_base, out_lora],
        )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
