from huggingface_hub import login, HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
import os
from exception.exceptions import HuggingfaceAuthException


def delete_hf_repo_if_exists(repo_id: str) -> None:
    """Delete the HF model repo for `repo_id` if it exists, otherwise no-op.

    Used to implement `--overwrite-repo true` semantics: without deleting
    first, push_to_hub leaves stale files behind (most commonly old shard
    files from prior pushes with different shard counts).
    """
    api = HfApi()
    try:
        api.delete_repo(repo_id=repo_id, repo_type="model", missing_ok=True)
        print(f'Deleted existing HF repo {repo_id} (--overwrite-repo)')
    except RepositoryNotFoundError:
        # `missing_ok=True` already handles this on newer hub clients; older
        # clients raise instead. Either way, nothing to delete.
        pass
    except HfHubHTTPError as e:
        # Surface non-fatal so the push still gets attempted.
        print(f'WARNING - could not delete HF repo {repo_id} before push: {e}')


def authenticate_with_hf(auth_token: str | None = None) -> None:
    """Authenticate with Huggingface"""
    print()
    print('Authenticating with Huggingface')
    print()
    try:
        login(resolve_hf_token(auth_token))
        print()
    except Exception as e:
        raise HuggingfaceAuthException(f'error authenticating with huggingface: {str(e)}')


def resolve_hf_token(auth_token: str | None = None) -> str | None:
    """Resolve Huggingface auth token"""
    try:
        return os.environ.get('HUGGING_FACE_TOKEN') if auth_token is None else auth_token
    except Exception as e:
        return None