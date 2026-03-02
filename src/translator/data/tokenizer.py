import sentencepiece as spm
from pathlib import Path


class Tokenizer:
    def __init__(self, model_path: str | Path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(model_path))

        self.pad_id = self.sp.PieceToId('<pad>')
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()  # <START>  
        self.eos_id = self.sp.eos_id()  # <END> 

        self.vocab_size = self.sp.GetPieceSize()

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = self.sp.EncodeAsIds(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        filtered = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.DecodeIds(filtered)

    def decode_with_tokens(self, ids: list[int]) -> list[str]:
        """
        Visualization of the tokens as list of string
        """
        return [self.sp.IdToPiece(i) for i in ids]

    @staticmethod
    def train_model(
        input_file: str,
        model_prefix: str,
        vocab_size: int = 8000,
    ):
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            pad_id=3,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            character_coverage=1.0,
            normalization_rule_name='nmt_nfkc_cf',
        )
        print(f"Model saved: {model_prefix}.model")
