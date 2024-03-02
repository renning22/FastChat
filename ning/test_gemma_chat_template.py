from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_id = "google/gemma-7b-it"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=dtype,
)

chat = [
    { "role": "user", "content": "Who are you?" },
    { "role": "model", "content": "I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS)." },
    { "role": "user", "content": "What are you doing?" },
    # { "role": "model", "content": "You too!" },
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

print(prompt)

print('===========')

inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=350)
print(tokenizer.decode(outputs[0]))