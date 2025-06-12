from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define the model name
model_name = "deepseek-ai/deepseek-llm-7b-base"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
# Using torch.bfloat16 for memory efficiency if your GPU supports it, or torch.float16.
# If you only have CPU, remove `torch_dtype` and `device_map`.
# If you run out of memory, consider loading in 8-bit or 4-bit (requires `bitsandbytes` and `accelerate` libraries).
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, # or torch.float16
        device_map="auto" # Automatically determines where to put model parts (e.g., GPU)
    )
    print(f"Model '{model_name}' loaded successfully with GPU acceleration.")
except Exception as e:
    print(f"Could not load model with GPU. Falling back to CPU. Error: {e}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"Model '{model_name}' loaded successfully on CPU.")

print(f"Model '{model_name}' and tokenizer loaded.")

# Example inference (optional, just to test)
# prompt = "What is the capital of France?"
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # Move inputs to the same device as the model
#
# outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1)
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(f"\nExample response: {response}")