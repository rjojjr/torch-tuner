from huggingface_hub import login
import os
from main.exception.exceptions import TunerException, HuggingfaceAuthException

def authenticate_with_hf() -> None:
    print('Authenticating with Huggingface')
    try:
        login(os.environ.get('HUGGING_FACE_TOKEN'))
    except Exception as e:
        raise HuggingfaceAuthException(f'error authenticating with huggingface: {str(e)}')