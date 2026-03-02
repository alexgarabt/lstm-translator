translator/
├── pyproject.toml            # uv project config
├── README.md
│
├── src/
│   └── translator/
│       ├── __init__.py
│       │
│       ├── model/            # Bloque 1 & 2: Arquitectura
│       │   ├── __init__.py
│       │   ├── lstm.py       # LSTMCell y LSTM custom
│       │   ├── encoder.py    # BiLSTM encoder
│       │   ├── attention.py  # Módulo de atención
│       │   ├── decoder.py    # Decoder con atención
│       │   └── seq2seq.py    # Modelo completo
│       │
│       ├── data/             # Bloque 3: Datos
│       │   ├── __init__.py
│       │   ├── tokenizer.py  # Word-level + BPE via sentencepiece
│       │   ├── dataset.py    # torch Dataset + DataLoader
│       │   └── download.py   # Descarga Tatoeba/OPUS
                preprocessing.py
│       │
│       ├── training/         # Bloque 4: Entrenamiento
│       │   ├── __init__.py
│       │   ├── trainer.py    # Training loop principal
│       │   ├── metrics.py    # BLEU, gradient norms, attention entropy
│       │   └── callbacks.py  # Logging a TensorBoard, checkpointing
│       │
│       └── config.py         # Hiperparámetros centralizados
│
├── tests/                    # Verificación
│   ├── test_lstm.py          # Comparar con nn.LSTMCell
│   ├── test_attention.py     # Verificar dimensiones y pesos suman 1
│   ├── test_seq2seq.py       # Test de copia/inversión
│   └── test_overfit.py       # Sobreajustar en 10 ejemplos
│
├── scripts/ 
│   ├── train.py              # Entry point: python scripts/train.py
│   ├── evaluate.py           # Evaluar modelo guardado
│   └── translate.py          # Traducir frases interactivamente
│
└── checkpoints/              # Modelos guardados (gitignored)
