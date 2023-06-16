import llm_client
import numpy as np
import torch

client = llm_client.Client(address="babel-0-19")

# input_ids, logits, tokens = client.score(["Once upon a time there was a terrible dragon."])

# input_ids = torch.tensor(input_ids)
# logits = torch.tensor(logits)

# scores = torch.log(logits.softmax(dim=-1)).detach()
# scores = scores.cuda()
# input_ids = input_ids.cuda()

# scores = torch.gather(scores, 2, input_ids[:, :, None].cuda()).squeeze(-1)
# scores = scores.cpu().numpy()[0, :].tolist()
# # tokens = outputs["tokens"]

# print(input_ids)
# print(tokens)
# print(scores)

# text = "The sun slowly set behind the mountains, casting a warm orange glow across the tranquil lake."
text = "Once upon a time there was a terrible dragon."
tokens, scores = client.score([text])

print(text)
for tok, s in zip(tokens, scores):
    print(f"[{tok}]: {s:.2f}")
#score = outputs[0].scores

#score_inf = np.isinf(score)
#print(score.shape[0] * score.shape[1] - score_inf.sum())
# print(outputs[0])
