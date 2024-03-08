import json

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)  # attn_implementation="flash_attention_2",

dataset = load_dataset("berkeley-nest/Nectar")


def process(content):
    messages = [
        {"role": "user", "content": 'Summerize following texts into few emojis sequence.\n\n' + content},
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, max_new_tokens=4096, do_sample=True, temperature=0.1, top_p=0.30, top_k=0, repetition_penalty=1.1)
    texts = tokenizer.decode(outputs[0], skip_special_tokens=True)

    parts = texts.split('[/INST]')
    if len(parts) != 2:
        return content
    return parts[1]

results = []

total = 50000
for i in range(total):
    entry = dataset['train'][i]

    prompt = entry['prompt'].strip().removeprefix('Human:').strip().removesuffix('Assistant:').strip()
    answer = entry['answers'][0]
    assert answer['rank'] == 1.0

    processed = process(answer['answer'])
    assistant = processed

    if i < 10:
        print('>>>>>>>>>>>>>')
        print(prompt)
        print('--------------')
        print(assistant)
        print('<<<<<<<<<<<<<')
    else:
        print(f'{i}/{total}')

    result = {
        "id": str(i),
        "conversations": [
            {
                "from": "human",
                "value": prompt,
            },
            {
                "from": "gpt",
                "value": assistant,
            },
        ]
    }
    results.append(result)

    if i % 1000 == 1:
        filename = f"nectar_emoji_{i//1000}k.json"
        print(f'Dumping to {filename}')
        with open(filename, "w") as outfile:
            json.dump(results, outfile, indent = '  ')



