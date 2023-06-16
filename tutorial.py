import llm_client
from transformers import LlamaForCausalLM, LlamaTokenizer
import numpy as np
from scipy.special import softmax
import torch

client = llm_client.Client(address="babel-0-23") 
#text = "Once upon a time there was a terrible dragon."
#outputs = client.prompt([text], output_scores=True)

# Initialize the tokenizer
tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

text = "Once upon a time there was a terrible dragon."

inputs = tokenizer(text, padding=True, return_tensors="pt")
input_ids = inputs["input_ids"]

outputs = client.prompt([text], output_scores=True)
logits = outputs[0].scores

#scores = torch.log((softmax(logits, axis = 1))).detach()
scores = np.log(softmax(logits, axis = 1))

#print(type(scores), scores)
#print(type(input_ids), input_ids)
print(logits)
