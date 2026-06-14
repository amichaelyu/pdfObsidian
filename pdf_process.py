from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model, processor = load("mlx-community/dots.ocr-bf16")
config = load_config("mlx-community/dots.ocr-bf16")

image = ["img.png"]
prompt = ""

formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=1
)

output = generate(model, processor, formatted_prompt, image, max_tokens=10000)
print(output)