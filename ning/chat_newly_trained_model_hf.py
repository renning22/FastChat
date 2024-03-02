# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", device_map="cuda", torch_dtype=torch.float16)

# tokenizer = AutoTokenizer.from_pretrained("/data/ning/output_gemma_2")
# model = AutoModelForCausalLM.from_pretrained("/data/ning/output_gemma_2", device_map="cuda", torch_dtype=torch.float16)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(max_length = 512, **input_ids)
print(tokenizer.decode(outputs[0]))