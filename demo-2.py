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

output_text = outputs[0].text
input_ids = tokenizer(output_text)["input_ids"]
# input_ids = [id for id in input_ids if id != 1]

#print(input_ids)

tokens = [tokenizer.decode([tok]) for tok in input_ids]
#print(tokens)
logits = outputs[0].scores

logits_tensor = torch.tensor(logits)
input_ids_tensor = torch.tensor(input_ids)
scores = torch.log(logits_tensor.softmax(dim=-1)).detach()

print(scores)
#print(type(logits_tensor), logits_tensor)
#print(type(input_ids_tensor), input_ids_tensor)
