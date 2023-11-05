import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                         torch_dtype=torch.float16,
                                         use_safetensors=True, variant="fp16")
# If your GPU has > 8GB vRAM
# pipe.to("cuda")
# If your GPU has < 8GB vRAM
pipe.enable_model_cpu_offload()
# pipe.enable_xformers_memory_efficient_attention()

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = (input("describe the picture you want: "))

images = pipe(prompt=prompt).images[0]
images.save("result_SDXL.png")
