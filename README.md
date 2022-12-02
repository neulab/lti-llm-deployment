# LTI's Large Language Model Deployment

**TODO**: Add a description of the project.

This repo was originally a fork of the [huggingface](https://huggingface.co/)'s [BLOOM inference demos](https://github.com/huggingface/transformers-bloom-inference), ported to it's own repo to allow for more flexibility in the future.

## Installation

```bash
pip install -e .
```

## Example API Usage

```python
import lti_llm_client

client = lti_llm_client.Client()
client.prompt("CMU's PhD students are")
```

## Available arguments
The available arguments are basically identical to [Huggingface transformers' `model.generate`](https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate) function.

The first parameter (i.e., the prompt) is called `text`.

Additional arguments:
* `max_tokens`/`max_new_tokens` (`int`, default: `64`)
* `greedy` (`True`/`False`, default: `False`)
