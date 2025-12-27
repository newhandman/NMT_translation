## Dataset Preprocessing Pipeline

This project contains four bilingual JSONL files (small/large train, validation, test) under `dataset/`. Each record is a parallel `{ "en": "...", "zh": "...", "index": ... }` entry. Use `preprocess.py` to clean text, tokenize, build vocabularies, and produce intermediate artifacts suitable for downstream modeling.

### 1. Environment Setup

1. Use Python 3.10+ (repo tested on 3.13 via Anaconda).
2. Install preprocessing dependencies:

   ```bash
   pip install jieba nltk sentencepiece gensim numpy
   ```

   - `jieba`: Chinese segmentation.
   - `nltk`: English tokenization (`wordpunct_tokenize`).
   - `sentencepiece` (optional): experiment with BPE/WordPiece.
   - `gensim`: optional pretrained embedding loader (Word2Vec/GloVe/FastText).
   - `numpy`: embedding matrix storage.

### 2. Basic Usage

Run the script from the repo root:

```bash
python preprocess.py
```

This:

- Cleans raw JSONL files (`dataset/*.jsonl`) and writes to `artifacts/cleaned/`.
- Tokenizes and truncates sentences, producing `artifacts/tokenized/`.
- Builds vocabularies using both training sets (`artifacts/vocab/en_vocab.json`, `zh_vocab.json`).
- Converts tokens to ids for all splits (`artifacts/final/*_final.jsonl`).
- Initializes embedding tensors and summary stats (`artifacts/embeddings`, `artifacts/stats`).

### 3. Customization

`preprocess.py` exposes CLI arguments (see `python preprocess.py --help`):

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--raw_dir` | `dataset` | Directory containing the original JSONL splits. |
| `--output_dir` | `artifacts` | Root directory for all intermediate outputs. |
| `--max_chars_en` | `400` | Max characters per English sentence after cleaning. Longer sentences are truncated. |
| `--max_chars_zh` | `400` | Same for Chinese. |
| `--max_tokens_en` | `80` | Token limit for English after tokenization. |
| `--max_tokens_zh` | `80` | Token limit for Chinese. |
| `--min_freq_en` | `3` | Minimum frequency for English vocabulary entries (rare tokens become `<unk>`). |
| `--min_freq_zh` | `3` | Same for Chinese. |
| `--embedding_dim` | `50` | Size of randomly initialized embeddings (overridden if pretrained vectors supply a different size). |
| `--pretrained_en` | `None` | Path to an English pretrained embedding file (`.vec`, `.bin`, `.txt`, or Gensim `.gensim`). |
| `--pretrained_zh` | `None` | Same for Chinese. |

Examples:

```bash
# Increase truncation limits and build a smaller vocab
python preprocess.py --max_tokens_en 128 --max_tokens_zh 128 --min_freq_en 5 --min_freq_zh 5

# Use a different dataset directory and pretrained vectors
python preprocess.py \
  --raw_dir data/jsonl \
  --output_dir runs/2024-12-10 \
  --pretrained_en /path/to/glove.6B.100d.txt \
  --pretrained_zh /path/to/tencent-zh-200d.bin
```

When `--pretrained_*` files are provided, vectors are loaded via `gensim.models.KeyedVectors`. Token embeddings falling outside the pretrained vocabulary are initialized randomly.

### 4. Intermediate Artifacts

- `artifacts/cleaned/*.jsonl`: sanitized text.
- `artifacts/tokenized/*.jsonl`: token lists plus truncation stats.
- `artifacts/vocab/*.json`: vocabularies with frequencies.
- `artifacts/final/*.jsonl`: merged tokens + integer ids per split.
- `artifacts/embeddings/*_embeddings.npz`: NumPy tensors ready for model initialization.
- `artifacts/stats/processing_stats.json`: summary suitable for reports (counts, truncations, vocab sizes, embedding coverage).

### 5. Optional Extensions

- Replace `tokenize_en`/`tokenize_zh` inside `preprocess.py` if you want SentencePiece/BPE or different Chinese segmenters.
- Adjust `SPECIAL_TOKENS` or add more metadata fields as needed.
- Integrate additional filtering rules (e.g., length ratio) in `clean_split` or `tokenize_split`.

---

## RNN-based NMT Training

After preprocessing, use `train_nmt.py` to train/evaluate a two-layer GRU encoder-decoder with configurable attention, teacher forcing, and decoding strategies.

### 1. Extra Dependencies

Install PyTorch (CUDA optional) plus the metric/finetuning toolkits:

```bash
pip install torch sacrebleu transformers numpy
```

PyTorch wheels: https://pytorch.org/get-started/locally/ (match your CUDA version).

### 2. Basic Training

```bash
python train_nmt.py \
  --train_file artifacts/final/train_10k_final.jsonl \
  --valid_file artifacts/final/valid_final.jsonl \
  --test_file artifacts/final/test_final.jsonl \
  --teacher_forcing_ratio 1.0 \
  --attention dot \
  --device cuda  # or cpu
```

- Encoder/decoder: two unidirectional GRU layers (`hidden_dim` default 512) initialized from `artifacts/embeddings/*`.
- Checkpoints saved under `checkpoints/` (best validation BLEU).
- Built-in validation metrics: BLEU + chrF (via SacreBLEU) computed after every epoch for each strategy listed in `--eval_strategies` (comma-separated, default `greedy`). Every strategy logs its own BLEU/chrF and saves the best-performing checkpoint; metrics land in `logs/rnn_train_metrics.jsonl` (override via `--log_file`), while evaluation runs log to `logs/rnn_eval_metrics.jsonl`.
- Training loop reports tokens/sec and decoding latency to help compare efficiency with Transformer runs.
- Change training subset size with `--max_train_examples` if you want to prototype quickly.

### 3. Attention Experiments

Switch the `--attention` flag to compare alignment functions:

```bash
python train_nmt.py --attention multiplicative
python train_nmt.py --attention additive
```

Each run logs training/validation loss + perplexity so you can compare accuracy vs. runtime.

### 4. Teacher Forcing vs. Free Running

- Teacher forcing: `--teacher_forcing_ratio 1.0` (ground-truth tokens fed to the decoder).
- Free running: `--teacher_forcing_ratio 0.0` (model feeds back its own predictions).
- Mixed policy: set a ratio in `(0,1)` to randomly sample the next input token source.

Run separate trainings with different ratios to analyze convergence and exposure bias.

### 5. Decoding Policies

Switch the decoding mode when evaluating (`--mode eval`):

```bash
# Greedy
python train_nmt.py --mode eval --load_checkpoint checkpoints/nmt_dot_tf1.00.pt --decode_strategy greedy

# Beam search (size 5)
python train_nmt.py --mode eval --load_checkpoint checkpoints/nmt_dot_tf1.00.pt \
  --decode_strategy beam --beam_size 5 --max_decode_len 80
```

Greedy decoding selects the highest-probability token at each step, while beam search retains the top `beam_size` hypotheses and yields more fluent translations.

### 6. Additional Controls

| Flag | Description |
| ---- | ----------- |
| `--hidden_dim` | Hidden size of encoder/decoder GRUs. |
| `--dropout` | Dropout applied in both encoder/decoder and between stacked layers. |
| `--batch_size`, `--epochs`, `--lr`, `--clip` | Standard optimization knobs. |
| `--max_decode_len`, `--beam_size` | Decoding hyperparameters for greedy/beam search. |
| `--max_*_examples` | Quickly subsample train/valid/test for debugging. |
| `--load_checkpoint` | Resume training or run evaluation from a saved model. |
| `--eval_decode_strategy` | Greedy or beam decoding specifically for validation scoring. |
| `--eval_strategies` | Comma-separated list (e.g., `greedy,beam`) evaluated each epoch; best BLEU per strategy is checkpointed. |
| `--max_eval_batches` | Decode only a subset of validation/test batches when BLEU computation is too costly. |
| `--log_file`, `--eval_log_file` | JSONL files that accumulate per-epoch and evaluation metrics. |

`train_nmt.py` prints sample predictions during evaluation mode and reports BLEU/chrF on both validation and test sets. Extend it to compute additional metrics or log to TensorBoard as needed.

---

## Batch Checkpoint Evaluation

Use `evaluate_checkpoints.py` to evaluate every saved RNN/Transformer/T5 checkpoint on the test split with a single command:

```bash
python evaluate_checkpoints.py \
  --checkpoints_dir checkpoints \
  --output_file logs/checkpoint_eval_results.jsonl \
  --device cuda \
  --beam_size 5 \
  --max_decode_len 80
```

- The script infers architecture/decoding settings from each filename and reports BLEU/chrF plus average decoding time.
- Limit workload via `--max_test_examples` or `--max_eval_batches` when you only need a quick comparison.
- Results stream to stdout and are also stored as JSONL so you can load them into pandas or the provided `analyze_logs.py`.

## Transformer-based NMT

`train_transformer.py` covers training a Transformer from scratch and fine-tuning pretrained encoder-decoder LMs (T5) for Chinese→English translation.

### 1. Training From Scratch

```bash
python train_transformer.py \
  --model_variant scratch \
  --positional_encoding sinusoidal \
  --norm_type layernorm \
  --d_model 512 --num_heads 8 --num_layers 6 --ff_dim 2048 \
  --batch_size 32 --lr 5e-4 --epochs 10 \
  --device cuda
```

- Encoder/decoder share `--num_layers` blocks. Toggle normalization via `--norm_type layernorm|rmsnorm` and positional embeddings via `--positional_encoding sinusoidal|learned|relative`.
- Validation BLEU/chrF is computed after every epoch for every decoding strategy listed in `--eval_strategies` (comma-separated; greedy by default). The best BLEU per strategy is checkpointed (e.g., `checkpoints/transformer_<pos>_<norm>_<strategy>.pt`), and per-epoch statistics (loss/ppl/BLEU/chrF/latency) are recorded to `logs/transformer_train_metrics.jsonl` (change via `--log_file`).

### 2. Architectural Ablations

Re-run training with different positional/normalization settings to isolate their impact:

```bash
python train_transformer.py --model_variant scratch --positional_encoding learned --norm_type layernorm
python train_transformer.py --model_variant scratch --positional_encoding relative --norm_type rmsnorm
```

### 3. Hyperparameter Sensitivity

- Batch size: adjust `--batch_size` (e.g., 16, 32, 64).
- Learning rate: vary `--lr` (baseline 5e-4).
- Model scale: experiment with `--d_model`, `--num_layers`, `--num_heads`, `--ff_dim`.

Log BLEU/chrF per run to evaluate robustness.

### 4. Decoding Policies

Switch inference mode without retraining:

```bash
python train_transformer.py --mode eval \
  --load_checkpoint checkpoints/transformer_sinusoidal_layernorm.pt \
  --decode_strategy beam --beam_size 5 --max_decode_len 80
```

### 5. Fine-tune Pretrained T5

```bash
python train_transformer.py \
  --model_variant pretrained \
  --pretrained_model_name t5-small \
  --batch_size 16 --t5_lr 3e-4 --t5_max_length 128 \
  --epochs 5 --beam_size 5 \
  --device cuda
```

- Requires Hugging Face `transformers`. The script rebuilds sentences from `*_tokens`, tokenizes with T5’s tokenizer, and fine-tunes `T5ForConditionalGeneration`.
- Validation/test BLEU/chrF mirror the scratch pipeline; specify multiple validation policies with `--eval_strategies` and pick a decoding mode for held-out evaluation via `--decode_strategy`. Per-epoch metrics are logged to `logs/transformer_train_metrics.jsonl`, and evaluation summaries land in `logs/transformer_eval_metrics.jsonl`.

---

## Log Analysis & Visualization

Use `analyze_logs.py` to parse every JSONL run inside `logs/`, render per-experiment metric curves, and capture the epoch with the best BLEU/chrF for both the RNN and Transformer sweeps.

### Dependencies

```bash
pip install matplotlib
```

### Usage

```bash
python analyze_logs.py \
  --log_dir logs \
  --plot_dir logs/plots \
  --summary_path logs/best_metrics_summary.json
```

- Generates PNG plots (train/valid loss, throughput, BLEU, chrF, latency, etc.) in `logs/plots/` for every run, including the combined Transformer normalization experiments (`transformer_train_metrics.jsonl` is automatically split into its five variants).
- Writes `logs/best_metrics_summary.json` summarizing the epoch/timestamp/value for the maximum BLEU and chrF observed in each run so you can quickly compare positional encodings, normalization layers, and decoding strategies across the entire sweep.

Tweak the CLI paths if you store logs elsewhere or want to place the outputs in a report-friendly directory.
