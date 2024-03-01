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
    { "role": "user", "content": "Have a nice day!" },
    { "role": "model", "content": "You too!" },
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

print(prompt)