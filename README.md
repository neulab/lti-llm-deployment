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
