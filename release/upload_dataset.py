"""Upload the packed ChesapeakeRSC v2 core release (patches_tar/ + index/) to a public
HF dataset repo. Excludes patches_legacy_tar/ on purpose -- this release is core-only.
"""

import argparse

from huggingface_hub import HfApi


def main(a: argparse.Namespace) -> None:
    api = HfApi()
    api.create_repo(a.repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_large_folder(
        folder_path=a.release_dir,
        repo_id=a.repo_id,
        repo_type="dataset",
        allow_patterns=["patches_tar/*", "index/*", "assets/*.png", "README.md"],
    )
    print(f"done: https://huggingface.co/datasets/{a.repo_id}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--release_dir", required=True)
    p.add_argument("--repo_id", default="isaaccorley/chesapeakersc-v2")
    main(p.parse_args())
