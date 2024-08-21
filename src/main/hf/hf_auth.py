from huggingface_hub import login
import os
from exception.exceptions import HuggingfaceAuthException


def authenticate_with_hf() -> None:
    """Authenticate with Huggingface using `HUGGING_FACE_TOKEN` environment variable."""
    print('Authenticating with Huggingface')
    try:
        login(os.environ.get('HUGGING_FACE_TOKEN'))
    except Exception as e:
        raise HuggingfaceAuthException(f'error authenticating with huggingface: {str(e)}')
