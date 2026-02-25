INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': False,
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'height': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'width': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 25
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 7.5
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda img_count: 3 > img_count > 0
    },
    # Optional reference-image inputs (first valid image is used)
    'reference_images': {
        'type': list,
        'required': False,
        'default': None,
    },
    'ref_images': {
        'type': list,
        'required': False,
        'default': None,
    },
    'reference_image_url': {
        'type': str,
        'required': False,
        'default': None,
    },
    'reference_image_path': {
        'type': str,
        'required': False,
        'default': None,
    },
    'reference_image_base64': {
        'type': str,
        'required': False,
        'default': None,
    },
    'ref_image_url': {
        'type': str,
        'required': False,
        'default': None,
    },
    'ref_image_path': {
        'type': str,
        'required': False,
        'default': None,
    },
    'ref_image_base64': {
        'type': str,
        'required': False,
        'default': None,
    },
    'reference_strength': {
        'type': float,
        'required': False,
        'default': 0.7,
    },
}
