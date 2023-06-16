import llm_client
from transformers import LlamaForCausalLM, LlamaTokenizer
import numpy as np
from scipy.special import softmax
import torch

client = llm_client.Client(address="babel-0-23")
# text = "A B C D E F G" 
text = "Once upon a time there was a terrible dragon."
outputs = client.prompt([text], output_scores=True)
# tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
tokenizer = LlamaTokenizer.from_pretrained("/home/yifengw2/lti-llm/llama-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
input_ids = tokenizer(text)["input_ids"]
# input_ids = [id for id in input_ids if id != 1]

tokens = [tokenizer.decode([tok]) for tok in input_ids]
logits = outputs[0].scores
logits = np.where(np.isinf(logits), 0, logits)

logits = logits[:len(input_ids),:]
scores = np.log(softmax(logits, axis = 1))
scores = scores[range(len(input_ids)), input_ids]
print(outputs[0].scores)
print(input_ids)
print(tokens)
print(logits.shape)
# print([id for id in input_ids if id != 1])
print(scores)
for tok, s in zip(tokens, scores):
    print(f"[{tok}]: {s:.2f}")
print(tokenizer(text))
print(type(input_ids))
