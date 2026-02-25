import base64
import inspect
import io
import os
import urllib.request

import runpod
import torch
from pruna import PrunaModel
from PIL import Image
from runpod.serverless.utils import rp_cleanup, rp_upload
from runpod.serverless.utils.rp_validator import validate

from schemas import INPUT_SCHEMA

torch.cuda.empty_cache()


class ModelHandler:
    def __init__(self):
        self.pipe = None
    
    def load_models(self):
        # Load FLUX.1-dev pipeline from local cache first, then fallback to hub.
        model_id = os.environ.get("HF_MODEL", "PrunaAI/FLUX.1-dev-smashed-no-compile")
        try:
            self.pipe = PrunaModel.from_hub(
                model_id,
                local_files_only=True,
            )
            print(f"[ModelHandler] Loaded model from local cache: {model_id}", flush=True)
        except Exception as local_err:
            print(
                f"[ModelHandler] Local cache load failed for {model_id}: {local_err}. "
                "Falling back to Hugging Face download...",
                flush=True,
            )
            self.pipe = PrunaModel.from_hub(
                model_id,
                local_files_only=False,
            )
            print(f"[ModelHandler] Loaded model from Hugging Face: {model_id}", flush=True)
        self.pipe.move_to_device("cuda")
        return self.pipe


MODELS = None
MODEL_INIT_ERROR = None


def _get_models():
    global MODELS, MODEL_INIT_ERROR
    if MODELS is not None:
        return MODELS
    try:
        MODELS = ModelHandler()
        MODELS.load_models()
        MODEL_INIT_ERROR = None
        return MODELS
    except Exception as e:
        MODEL_INIT_ERROR = str(e)
        print(f"[ModelHandler] Initialization failed: {MODEL_INIT_ERROR}", flush=True)
        return None


def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get("BUCKET_ENDPOINT_URL", False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def _safe_float(value, default=0.7):
    try:
        return float(value)
    except Exception:
        return float(default)


def _decode_data_or_base64_to_pil(value: str):
    if not isinstance(value, str) or not value.strip():
        return None
    payload = value.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        raw = base64.b64decode(payload)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def _load_reference_image_from_candidate(candidate):
    if candidate is None:
        return None
    if not isinstance(candidate, str):
        return None
    value = candidate.strip()
    if not value:
        return None

    if value.startswith("http://") or value.startswith("https://"):
        try:
            with urllib.request.urlopen(value, timeout=30) as response:
                return Image.open(io.BytesIO(response.read())).convert("RGB")
        except Exception:
            return None

    if os.path.exists(value):
        try:
            return Image.open(value).convert("RGB")
        except Exception:
            return None

    return _decode_data_or_base64_to_pil(value)


def _extract_reference_candidates(job_input):
    candidates = []
    for key in (
        "reference_image_url",
        "reference_image_path",
        "reference_image_base64",
        "ref_image_url",
        "ref_image_path",
        "ref_image_base64",
    ):
        value = job_input.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value)

    for key in ("reference_images", "ref_images"):
        values = job_input.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, str) and item.strip():
                candidates.append(item)
            elif isinstance(item, dict):
                for alt in (
                    "url",
                    "image_url",
                    "path",
                    "image_path",
                    "base64",
                    "image_base64",
                    "image",
                ):
                    value = item.get(alt)
                    if isinstance(value, str) and value.strip():
                        candidates.append(value)
                        break
    return candidates


def _resolve_reference_image(job_input):
    for candidate in _extract_reference_candidates(job_input):
        image = _load_reference_image_from_candidate(candidate)
        if image is not None:
            return image, candidate
    return None, None


@torch.inference_mode()
def generate_image(job):
    """
    Generate an image from text using FLUX.1-dev Model
    """
    # -------------------------------------------------------------------------
    # 🐞 DEBUG LOGGING
    # -------------------------------------------------------------------------
    import json
    import pprint

    # Log the exact structure RunPod delivers so we can see every nesting level.
    print("[generate_image] RAW job dict:")
    try:
        print(json.dumps(job, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job, depth=4, compact=False)

    # -------------------------------------------------------------------------
    # Original (strict) behaviour – assume the expected single wrapper exists.
    # -------------------------------------------------------------------------
    models = _get_models()
    if models is None:
        return {
            "error": f"Model initialization failed: {MODEL_INIT_ERROR}",
            "refresh_worker": True,
        }

    job_input = job["input"]

    print("[generate_image] job['input'] payload:")
    try:
        print(json.dumps(job_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job_input, depth=4, compact=False)

    # Input validation
    try:
        validated_input = validate(job_input, INPUT_SCHEMA)
    except Exception as err:
        import traceback

        print("[generate_image] validate(...) raised an exception:", err, flush=True)
        traceback.print_exc()
        # Re-raise so RunPod registers the failure (but logs are now visible).
        raise

    print("[generate_image] validate(...) returned:")
    try:
        print(json.dumps(validated_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(validated_input, depth=4, compact=False)

    if "errors" in validated_input:
        return {"error": validated_input["errors"]}
    job_input = validated_input["validated_input"]

    if job_input["seed"] is None:
        job_input["seed"] = int.from_bytes(os.urandom(2), "big")

    # Create generator with proper device handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(job_input["seed"])
    reference_image, reference_source = _resolve_reference_image(job_input)
    reference_strength = _safe_float(job_input.get("reference_strength", 0.7), 0.7)
    reference_applied = False

    try:
        # Generate image using FLUX.1-dev pipeline
        with torch.inference_mode():
            call_kwargs = {
                "prompt": job_input["prompt"],
                "negative_prompt": job_input["negative_prompt"],
                "height": job_input["height"],
                "width": job_input["width"],
                "num_inference_steps": job_input["num_inference_steps"],
                "guidance_scale": job_input["guidance_scale"],
                "num_images_per_prompt": job_input["num_images"],
                "generator": generator,
            }

            if reference_image is not None:
                sig = inspect.signature(MODELS.pipe.__call__)
                params = sig.parameters
                if "image" in params:
                    call_kwargs["image"] = reference_image
                    if "strength" in params:
                        call_kwargs["strength"] = reference_strength
                    reference_applied = True
                elif "ip_adapter_image" in params:
                    call_kwargs["ip_adapter_image"] = reference_image
                    reference_applied = True

                if reference_applied:
                    print(
                        f"[generate_image] reference image applied via "
                        f"{'image' if 'image' in call_kwargs else 'ip_adapter_image'} "
                        f"(source={reference_source})",
                        flush=True,
                    )
                else:
                    print(
                        "[generate_image] reference image provided but current "
                        "pipeline signature has no supported image-conditioning args; "
                        "running text-only.",
                        flush=True,
                    )

            result = models.pipe(
                prompt=job_input["prompt"],
                negative_prompt=job_input["negative_prompt"],
                height=job_input["height"],
                width=job_input["width"],
                num_inference_steps=job_input["num_inference_steps"],
                guidance_scale=job_input["guidance_scale"],
                num_images_per_prompt=job_input["num_images"],
                generator=generator,
                **{
                    k: v
                    for k, v in call_kwargs.items()
                    if k
                    not in (
                        "prompt",
                        "negative_prompt",
                        "height",
                        "width",
                        "num_inference_steps",
                        "guidance_scale",
                        "num_images_per_prompt",
                        "generator",
                    )
                },
            )
            output = result.images
    except RuntimeError as err:
        print(f"[ERROR] RuntimeError in generation pipeline: {err}", flush=True)
        return {
            "error": f"RuntimeError: {err}, Stack Trace: {err.__traceback__}",
            "refresh_worker": True,
        }
    except Exception as err:
        print(f"[ERROR] Unexpected error in generation pipeline: {err}", flush=True)
        return {
            "error": f"Unexpected error: {err}",
            "refresh_worker": True,
        }

    image_urls = _save_and_upload_images(output, job["id"])

    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input["seed"],
        "reference_used": bool(reference_image is not None),
        "reference_applied": reference_applied,
        "reference_source": reference_source,
    }

    return results


runpod.serverless.start({"handler": generate_image})
