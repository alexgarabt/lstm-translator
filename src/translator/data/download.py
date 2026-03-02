import urllib.request
import zipfile
from pathlib import Path


TATOEBA_URL = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-es.txt.zip"


def download_tatoeba(data_dir: str | Path = "data") -> tuple[Path, Path]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "en-es.zip"
    en_path = data_dir / "Tatoeba.en-es.en"
    es_path = data_dir / "Tatoeba.en-es.es"

    if en_path.exists() and es_path.exists():
        print(f"Data already exists in {data_dir}")
        return en_path, es_path

    print(f"Downloading Tatoeba EN-ES...")
    urllib.request.urlretrieve(TATOEBA_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)

    zip_path.unlink()  # borrar el zip
    print(f"Done. Files: {en_path}, {es_path}")
    return en_path, es_path
