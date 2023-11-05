from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# pipe.to("cuda")
pipe.enable_model_cpu_offload()
# pipe.enable_xformers_memory_efficient_attention()

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()


prompt = (input("describe the picture you want: "))

images = pipe(prompt=prompt).images[0]
images.save("result_SDXL.png")
