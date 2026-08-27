import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

print("=" * 60)
print("MODEL LOAD TEST")
print("=" * 60)

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

print("\nCreating quantization config...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

print("\nMODEL LOADED SUCCESSFULLY")

print("\nDevice map:")
print(model.hf_device_map)

print("\nGPU memory:")
print(
    f"Allocated: "
    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
)

print(
    f"Reserved: "
    f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
)