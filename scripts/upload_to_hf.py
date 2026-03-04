from huggingface_hub import HfApi, create_repo
from pathlib import Path

REPO_ID = "alexgara/lstm-en-es-translator"  

def main():
    api = HfApi()

    create_repo(REPO_ID, exist_ok=True)

    # Upload checkpoint
    api.upload_file(
        path_or_fileobj="training/checkpoint_v2/best_model.pt",
        path_in_repo="model.pt",
        repo_id=REPO_ID,
    )

    # Upload tokenizers
    api.upload_file(
        path_or_fileobj="training/data/spm_en_v2.model",
        path_in_repo="spm_en.model",
        repo_id=REPO_ID,
    )

    api.upload_file(
        path_or_fileobj="training/data/spm_es_v2.model",
        path_in_repo="spm_es.model",
        repo_id=REPO_ID,
    )

    # training data
    api.upload_file(
        path_or_fileobj="data/combined.es",
        path_in_repo="data/combined.es",
        repo_id=REPO_ID,
    )
    
    api.upload_file(
        path_or_fileobj="data/combined.en",
        path_in_repo="data/combined.en",
        repo_id=REPO_ID,
    )

    # Config with model hyperparameters (so anyone can rebuild the model)
    api.upload_file(
        path_or_fileobj="src/translator/config.py",
        path_in_repo="config.py",
        repo_id=REPO_ID,
    )

    api.upload_file(
        path_or_fileobj="scripts/train.py",
        path_in_repo="train.py",
        repo_id=REPO_ID,
    )

    api.upload_file(
        path_or_fileobj="README_hub.md",
        path_in_repo="README.md",
        repo_id=REPO_ID,
    )

    print(f"Uploaded to https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()

