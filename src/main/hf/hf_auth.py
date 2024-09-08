from huggingface_hub import login
import os
from exception.exceptions import HuggingfaceAuthException


def authenticate_with_hf(auth_token: str | None = None) -> None:
    """Authenticate with Huggingface"""
    print('Authenticating with Huggingface')
    try:
        login(os.environ.get('HUGGING_FACE_TOKEN') if auth_token is None else auth_token)
    except Exception as e:
        raise HuggingfaceAuthException(f'error authenticating with huggingface: {str(e)}')
