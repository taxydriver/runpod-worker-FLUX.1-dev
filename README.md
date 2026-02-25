[![Pruna AI Logo](https://github.com/PrunaAI/pruna/raw/main/docs/assets/images/logo.png)](https://pruna.ai)

**Simply make AI models faster, cheaper, smaller, greener!**

[![Documentation](https://img.shields.io/badge/Pruna_documentation-purple?style=for-the-badge)](https://docs.pruna.ai)

[![Website](https://img.shields.io/badge/Pruna.ai-purple?style=flat-square)](https://pruna.ai)
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FPrunaAI)](https://x.com/PrunaAI)
[![Devto](https://img.shields.io/badge/dev-to-black?style=flat-square)](https://dev.to/prunaai)
[![Reddit](https://img.shields.io/badge/Follow-r%2FPrunaAI-orange?style=social)](https://reddit.com/r/PrunaAI)
[![Discord](https://img.shields.io/badge/Discord-join_us-purple?style=flat-square)](https://discord.gg/prunaai)
[![Huggingface](https://img.shields.io/badge/Huggingface-models-yellow?style=flat-square)](https://huggingface.co/PrunaAI)
[![Replicate](https://img.shields.io/badge/replicate-black?style=flat-square)](https://replicate.com/prunaai)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat-square&logo=github&logoColor=white)](https://github.com/PrunaAI)

![Pruna AI Logo](https://github.com/PrunaAI/pruna/raw/main/docs/assets/images/triple_line.png)

![FLUX.1-dev Worker Banner](https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/flux_smashed_comparison.png)

---

We can optimize any diffusers models and optimized FLUX.1-dev using the following techniques:

![FLUX.1-dev-juiced Generated Optimisation techniques](https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/flux_combination.png)

---

Run an optimized [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) as a serverless endpoint to generate images.


> **⚠️ Important Notes:**
> - **Compilation Time**: The first request may take 2-3 minutes as the model compiles for optimal performance
> - **Warmup Time**: Subsequent requests will be faster but may still have a brief warmup period

---

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-flux1-dev)](https://www.runpod.io/console/hub/PrunaAI/runpod-worker-FLUX.1-dev)

---

## Usage

The worker accepts the following input parameters:

| Parameter                 | Type    | Default  | Required  | Description                                                                                                         |
| :------------------------ | :------ | :------- | :-------- | :------------------------------------------------------------------------------------------------------------------ |
| `prompt`                  | `str`   | `None`   | **Yes**   | The main text prompt describing the desired image.                                                                  |
| `negative_prompt`         | `str`   | `None`   | No        | Text prompt specifying concepts to exclude from the image                                                           |
| `height`                  | `int`   | `1024`   | No        | The height of the generated image in pixels                                                                         |
| `width`                   | `int`   | `1024`   | No        | The width of the generated image in pixels                                                                          |
| `seed`                    | `int`   | `None`   | No        | Random seed for reproducibility. If `None`, a random seed is generated                                              |
| `num_inference_steps`     | `int`   | `25`     | No        | Number of denoising steps for the base model                                                                        |
| `guidance_scale`          | `float` | `7.5`    | No        | Classifier-Free Guidance scale. Higher values lead to images closer to the prompt, lower values more creative       |
| `num_images`              | `int`   | `1`      | No        | Number of images to generate per prompt (Constraint: must be 1 or 2)                                                |
| `reference_images`        | `list`  | `None`   | No        | Optional list of reference image inputs (URL/path/base64 or dict entries). First valid image is used.              |
| `ref_images`              | `list`  | `None`   | No        | Alias of `reference_images`.                                                                                         |
| `reference_image_url`     | `str`   | `None`   | No        | Optional single reference image URL.                                                                                 |
| `reference_image_path`    | `str`   | `None`   | No        | Optional single local reference image path.                                                                          |
| `reference_image_base64`  | `str`   | `None`   | No        | Optional single reference image as base64/data URL.                                                                  |
| `reference_strength`      | `float` | `0.7`    | No        | Optional conditioning strength when pipeline supports `image` + `strength`.                                         |

### Example Request

```json
{
  "input": {
    "prompt": "a knitted purple prune",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
    "seed": 42,
    "num_images": 1,
    "reference_images": [
      "https://example.com/character_front.png"
    ],
    "reference_strength": 0.7
  }
}
```

If a reference image is provided but the loaded pipeline does not expose image-conditioning arguments, the worker logs that condition and falls back to text-only generation.

which is producing an output like this:

```json
{
  "delayTime": 11449,
  "executionTime": 6120,
  "id": "447f10b8-c745-4c3b-8fad-b1d4ebb7a65b-e1",
  "output": {
    "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAIAAADwf7zU...",
    "images": [
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAIAAADwf7zU..."
    ],
    "seed": 42
  },
  "status": "COMPLETED",
  "workerId": "462u6mrq9s28h6"
}
```

and when you convert the base64-encoded image into an actual image, it looks like this:

![FLUX.1-dev-juiced Generated Image: 'A knitted purple prune'](https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/flux_smashed_comparison.png)
