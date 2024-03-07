# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", device_map="cuda", torch_dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained("/data/ning/gemma_2b_nectar_20k_1")
model = AutoModelForCausalLM.from_pretrained("/data/ning/gemma_2b_nectar_20k_1", device_map="cuda")

input_text = "List all heros in dota 2."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(max_length = 512, **input_ids)
print(tokenizer.decode(outputs[0]))