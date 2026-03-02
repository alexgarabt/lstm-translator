import torch
from torch.utils.data import Dataset
from .tokenizer import Tokenizer

class TranslationDataset(Dataset):
    def __init__(self, pairs:list[tuple[str,str]], src_tokenizer: Tokenizer, trg_tokenizer: Tokenizer):
        self.pairs = pairs
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        en, es = self.pairs[idx]

        # Add <BOS> <EOS>
        src_ids = self.src_tokenizer.encode(en)
        trg_ids = self.trg_tokenizer.encode(es)

        return {
                'src': torch.tensor(src_ids, dtype=torch.long),
                'trg': torch.tensor(trg_ids, dtype=torch.long),
                'src_text': en,
                'trg_text': es,
        }

def collate_fn(batch: list[dict], pad_id:int) -> dict:
    """
    Groups examples into a batch with padding.
    The DataLoader calls this automatically.
    
    Each example may have a different length.
    We pad with pad_id up to the maximum length in the batch.
    """

    # find max lenth in a given batch
    src_max_len = max(len(item['src']) for item in batch)
    trg_max_len = max(len(item['trg']) for item in batch)

    src_padded = []
    trg_padded = []
    src_lengths = []

    for item in batch:
        src = item['src']
        trg = item['trg']

        src_lengths.append(len(src))
        src_padded.append(torch.cat([src, torch.full((src_max_len - len(src),), pad_id, dtype=torch.long)]))
        trg_padded.append(torch.cat([trg, torch.full((trg_max_len - len(trg),), pad_id, dtype=torch.long)]))

    return {
        'src': torch.stack(src_padded),           # (batch, src_max_len)
        'trg': torch.stack(trg_padded),           # (batch, trg_max_len)
        'src_lengths': torch.tensor(src_lengths),  # (batch,)
    }
