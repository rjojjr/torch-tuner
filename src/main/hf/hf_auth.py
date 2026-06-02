from huggingface_hub import login, HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
import os
from exception.exceptions import HuggingfaceAuthException


def upload_model_folder(folder_path: str, repo_id: str, private: bool, commit_message: str) -> None:
    """Upload an on-disk model folder to HF without loading the model.

    Replaces the load-then-`push_to_hub` round-trip, which materializes the
    full model in memory just to re-serialize and upload -- wasteful for
    large merged models, especially on memory-constrained boxes.
    """
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    except (HfHubHTTPError, RepositoryNotFoundError):
        pass   # repo may already exist or creation may have partially succeeded

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=folder_path,
        commit_message=commit_message,
    )


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
        pass   # `missing_ok=True` already handles this on newer hub clients; older
               # clients raise instead. Either way, nothing to delete.
    except HfHubHTTPError as e:
        pass   # Surface non-fatal so the push still gets attempted.
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


def get_hf_username(auth_token: str | None = None) -> str | None:
    """Return the Huggingface username for the authenticated token, or None."""
    try:
        token = resolve_hf_token(auth_token)
        if token is None:
            return None
        user_info = HfApi().whoami(token=token)
        if isinstance(user_info, dict):
            return user_info.get('name')
        return str(user_info)
    except Exception:
        return None
