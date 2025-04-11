from huggingface_hub import HfApi, HfFolder


def get_hf_username():
    api = HfApi()

    token = HfFolder.get_token()
    user_info = api.whoami(token=token)
    return user_info["name"]


def create_repo(repo_id, repo_type, is_private = True):
    """Creates given HF repository if it does not exist."""
    api = HfApi()
    if not api.repo_exists(repo_id):
        print(f'creating HF repo {repo_id}')
        api.create_repo(
            repo_id=repo_id,
            private=is_private,
            repo_type=repo_type
        )
        print()


def upload_folder(repo_id, folder_path, commit_message = None, path_in_repo = None, is_private = True, repo_type = 'model'):
    api = HfApi()
    create_repo(repo_id, repo_type, is_private)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        ignore_patterns=None,
        allow_patterns=None,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        commit_message=commit_message
    )


def upload_large_folder(repo_id, folder_path, is_private = True, repo_type = 'model'):
    api = HfApi()
    create_repo(repo_id, repo_type, is_private)
    api.upload_large_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        ignore_patterns=None,
        allow_patterns=None,
        repo_type=repo_type
    )