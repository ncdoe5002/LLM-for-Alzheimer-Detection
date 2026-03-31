from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Lambda LLM model and tokenizer from Hugging Face
model_name = "microsoft/Lambda"  # Replace with exact Lambda model repo if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define a simple prompt
prompt = "Hello world! Can you generate a short greeting?"

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate output (text)
outputs = model.generate(**inputs, max_new_tokens=50)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
