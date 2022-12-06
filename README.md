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

```python
import llm_client

client = llm_client.Client(address="tir-x-xx")
client.prompt("CMU's PhD students are")
```

## Available arguments

The first parameter, `text` corresponds the prompt that will be forced-decoded by the model.

The rest available arguments are basically identical to [Huggingface transformers' `model.generate`](https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate) function.
However, not all arguments are available, and better documentation of the ones that are will provided in the future.