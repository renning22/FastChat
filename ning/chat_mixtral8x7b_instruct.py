import pprint

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

messages = [
    {"role": "user", "content": "List all heros in dota 2."},
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

pprint.pprint(inputs)

inputs = inputs.to("cuda")

outputs = model.generate(inputs, max_new_tokens=4096, do_sample=True, temperature=0.1, top_p=0.15, top_k=0, repetition_penalty=1.1)
print(tokenizer.decode(outputs[0]))