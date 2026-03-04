from huggingface_hub import HfApi, create_repo
from pathlib import Path

REPO_ID = "alexgara/lstm-en-es-translator"


def main():
    api = HfApi()
    create_repo(REPO_ID, exist_ok=True)

    # Model artifacts
    api.upload_file(
        path_or_fileobj="training/checkpoint_v4/best_model.pt",
        path_in_repo="model.pt",
        repo_id=REPO_ID,
    )
    api.upload_file(
        path_or_fileobj="training/checkpoint_v4/hparams.json",
        path_in_repo="hparams.json",           
        repo_id=REPO_ID,
    )

    # Tokenizers 
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

    # Training data 
    api.upload_file(
        path_or_fileobj="training/data/combined.en",  
        path_in_repo="data/combined.en",
        repo_id=REPO_ID,
    )
    api.upload_file(
        path_or_fileobj="training/data/combined.es",
        path_in_repo="data/combined.es",
        repo_id=REPO_ID,
    )

    # Source references 
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

    # Training curves (SVG for crisp rendering) 
    for svg in Path("img/training").glob("*.svg"):
        api.upload_file(
            path_or_fileobj=str(svg),
            path_in_repo=f"img/{svg.name}",
            repo_id=REPO_ID,
        )

    # Attention heatmaps 
    for png in sorted(Path("img/training").glob("context*.png")):
        api.upload_file(
            path_or_fileobj=str(png),
            path_in_repo=f"img/{png.name}",
            repo_id=REPO_ID,
        )

    # README 
    api.upload_file(
        path_or_fileobj="README_hub.md",
        path_in_repo="README.md",
        repo_id=REPO_ID,
    )

    print(f"Uploaded to https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
