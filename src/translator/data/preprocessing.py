from pathlib import Path
import random


def load_pairs(en_path: Path, es_path: Path, max_length: int = 50) -> list[tuple[str, str]]:
    """
    Load pairs filters the ones more than max_length
    """
    pairs = []
    with open(en_path, 'r') as f_en, open(es_path, 'r') as f_es:
        for en_line, es_line in zip(f_en, f_es):
            en = en_line.strip()
            es = es_line.strip()

            # Filtrar vacías y muy largas
            if not en or not es:
                continue
            if len(en.split()) > max_length or len(es.split()) > max_length:
                continue

            pairs.append((en, es))

    print(f"Loaded {len(pairs)} pairs (max_length={max_length})")
    return pairs


def train_val_test_split(
    pairs: list[tuple[str, str]],
    val_size: int = 2000,
    test_size: int = 2000,
    seed: int = 42,
) -> tuple[list, list, list]:
    random.seed(seed)
    random.shuffle(pairs)

    test = pairs[:test_size]
    val = pairs[test_size:test_size + val_size]
    train = pairs[test_size + val_size:]

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def save_texts(pairs: list[tuple[str, str]], prefix: Path):
    with open(f"{prefix}.en", 'w') as f_en, open(f"{prefix}.es", 'w') as f_es:
        for en, es in pairs:
            f_en.write(en + '\n')
            f_es.write(es + '\n')
