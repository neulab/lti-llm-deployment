from typing import Any, List
import requests


class Client:
    """A client for the LTI's LLM API."""

    def __init__(self, address: str = "tir-1-7", port: int = 5000) -> None:
        """Initialize the client.

        Args:
            address: The address of the server. Defaults to "tir-1-7", a node on the LTI's TIR cluster.
            port: The port of the server. Defaults to 5000.
        """
        self.address = address
        self.port = port
        self.url = f"http://{self.address}:{self.port}"

    def prompt(
        self,
        text: str,
        max_tokens: int = 64,
        greedy: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Prompt the LLM currently being served with a text and return the response.
        Args:
            text: The text to prompt the LLM with.
            max_tokens: The maximum number of tokens to generate. Note
                that this is *excluding* the prompt tokens. Defaults to 64.
            greedy: Whether to use greedy decoding. Defaults to False.
            **kwargs: Additional keyword arguments to pass to model.
                They follow HF's generate API
        Returns:
        """
        request_body = {
            "text": [text],
            "max_new_tokens": max_tokens,
            "do_sample": not greedy,
            **kwargs,
        }
        response = requests.post(
            url=f"{self.url}/generate/", json=request_body, verify=False
        )
        return response.json()["text"]
