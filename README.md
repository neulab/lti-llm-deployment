# LTI's Large Language Model Deployment

**TODO**: Add a description of the project.

This repo was originally a fork of the [huggingface](https://huggingface.co/)'s [BLOOM inference demos](https://github.com/huggingface/transformers-bloom-inference), ported to it's own repo to allow for more flexibility in the future.

## Installation

```bash
pip install -e .
```

## Example API Usage

Currently, the client must be run from a compute node on the tir cluster.
If you don't have access to the tir cluster, please contact your advisor and ask.

Run the following commands, where `tir-x-xx` is the current location of the `lti-llm` running process.
The first parameter, `text` corresponds the prompt that will be forced-decoded by the model. The function will return a list of `Output` objects, one for every prompt in the input list.

```python
import llm_client

client = llm_client.Client(address="tir-x-xx")
outputs = client.prompt("CMU's PhD students are")
print(outputs[0].text)
```

### Model State

It is also possible to obtain the raw logit scores / output distribution from the model.

```python
import llm_client

client = llm_client.Client(address="tir-x-xx")
outputs = client.prompt(["CMU's PhD students are"], output_scores=True)
print(outputs[0].scores.shape)
```

And equivalently, it is possible to obtain the raw hidden states from the model.

```python
import llm_client

client = llm_client.Client(address="tir-x-xx")
outputs = client.prompt(["CMU's PhD students are"], output_hidden_states=True)
for layer in outputs[0].hidden_states:
    print(f"Layer {layer}: {layer.shape}")
```


### Other Available Arguments


The rest available arguments are basically identical to [Huggingface transformers' `model.generate`](https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate) function.
However, not all arguments are available, and better documentation of the ones that are will provided in the future.
