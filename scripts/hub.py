from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "alexgara/lstm-en-es-translator"


def download_tokenizers(cache_dir: str | Path = "hub_cache") -> tuple[Path, Path]:
    """Download tokenizer models from HuggingFace."""
    cache_dir = Path(cache_dir)

    en_model = hf_hub_download(
        repo_id=REPO_ID,
        filename="spm_en.model",
        cache_dir=cache_dir,
    )
    es_model = hf_hub_download(
        repo_id=REPO_ID,
        filename="spm_es.model",
        cache_dir=cache_dir,
    )

    return Path(en_model), Path(es_model)


def download_checkpoint(cache_dir: str | Path = "hub_cache") -> Path:
    """Download model weights from HuggingFace."""
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename="model.pt",
        cache_dir=Path(cache_dir),
    )
    return Path(path)


def download_data(cache_dir: str | Path = "hub_cache") -> tuple[Path, Path]:
    """Download training data from HuggingFace."""
    cache_dir = Path(cache_dir)

    en_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="data/combined.en",
        cache_dir=cache_dir,
    )
    es_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="data/combined.es",
        cache_dir=cache_dir,
    )

    return Path(en_path), Path(es_path)
