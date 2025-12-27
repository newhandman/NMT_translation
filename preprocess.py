#!/usr/bin/env python3
"""
Pipeline for cleaning, tokenizing, and preparing vocabularies/embeddings
for the bilingual JSONL datasets in dataset/.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import jieba
import numpy as np
from nltk.tokenize import wordpunct_tokenize

try:
    from gensim.models import KeyedVectors
except Exception:  # pragma: no cover - gensim is optional at runtime
    KeyedVectors = None  # type: ignore


ILLEGAL_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn"}
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


@dataclass
class SplitConfig:
    name: str
    filename: str


SPLITS: Tuple[SplitConfig, ...] = (
    SplitConfig("train_100k", "train_100k.jsonl"),
    SplitConfig("train_10k", "train_10k.jsonl"),
    SplitConfig("valid", "valid.jsonl"),
    SplitConfig("test", "test.jsonl"),
)
TRAIN_SPLIT_NAMES = {"train_100k", "train_10k"}


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_text(text: str, language: str, max_chars: int) -> str:
    text = text.strip()
    cleaned_chars: List[str] = []
    for ch in text:
        if unicodedata.category(ch) in ILLEGAL_CATEGORIES:
            continue
        cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars)
    cleaned = re.sub(r"\s+", " ", cleaned)
    if language == "en":
        cleaned = cleaned.lower()
    if max_chars and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned.strip()


def tokenize_en(text: str) -> List[str]:
    return [token for token in wordpunct_tokenize(text) if token.strip()]


def tokenize_zh(text: str) -> List[str]:
    return [token.strip() for token in jieba.lcut(text) if token.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_split(
    split: SplitConfig,
    raw_dir: Path,
    cleaned_dir: Path,
    max_chars_en: int,
    max_chars_zh: int,
) -> Dict:
    in_path = raw_dir / split.filename
    out_path = cleaned_dir / f"{split.name}_cleaned.jsonl"
    stats = {
        "total": 0,
        "kept": 0,
        "dropped_empty": 0,
    }
    cleaned_records: List[Dict] = []
    for record in read_jsonl(in_path):
        stats["total"] += 1
        en = clean_text(record["en"], "en", max_chars_en)
        zh = clean_text(record["zh"], "zh", max_chars_zh)
        if not en or not zh:
            stats["dropped_empty"] += 1
            continue
        cleaned_records.append(
            {
                "index": record.get("index", stats["total"] - 1),
                "en": en,
                "zh": zh,
            }
        )
        stats["kept"] += 1
    write_jsonl(out_path, cleaned_records)
    stats["output_path"] = str(out_path)
    return stats


def tokenize_split(
    split: SplitConfig,
    cleaned_dir: Path,
    tokenized_dir: Path,
    max_tokens_en: int,
    max_tokens_zh: int,
) -> Dict:
    in_path = cleaned_dir / f"{split.name}_cleaned.jsonl"
    out_path = tokenized_dir / f"{split.name}_tokenized.jsonl"
    stats = {
        "total": 0,
        "kept": 0,
        "truncated_en": 0,
        "truncated_zh": 0,
        "dropped_short": 0,
    }
    tokenized_records: List[Dict] = []
    for record in read_jsonl(in_path):
        stats["total"] += 1
        en_tokens = tokenize_en(record["en"])
        zh_tokens = tokenize_zh(record["zh"])
        if not en_tokens or not zh_tokens:
            stats["dropped_short"] += 1
            continue
        if max_tokens_en and len(en_tokens) > max_tokens_en:
            en_tokens = en_tokens[:max_tokens_en]
            stats["truncated_en"] += 1
        if max_tokens_zh and len(zh_tokens) > max_tokens_zh:
            zh_tokens = zh_tokens[:max_tokens_zh]
            stats["truncated_zh"] += 1
        tokenized_records.append(
            {
                "index": record["index"],
                "en_tokens": en_tokens,
                "zh_tokens": zh_tokens,
            }
        )
        stats["kept"] += 1
    write_jsonl(out_path, tokenized_records)
    stats["output_path"] = str(out_path)
    return stats


def build_vocab(
    language: str,
    tokenized_dir: Path,
    min_freq: int,
) -> Tuple[List[str], Dict[str, int]]:
    counter: Counter = Counter()
    for split in SPLITS:
        if split.name not in TRAIN_SPLIT_NAMES:
            continue
        path = tokenized_dir / f"{split.name}_tokenized.jsonl"
        if not path.exists():
            continue
        for record in read_jsonl(path):
            tokens = record[f"{language}_tokens"]
            counter.update(tokens)
    sorted_items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab_tokens = SPECIAL_TOKENS.copy()
    for token, freq in sorted_items:
        if freq < min_freq:
            continue
        vocab_tokens.append(token)
    token_to_idx = {token: idx for idx, token in enumerate(vocab_tokens)}
    return vocab_tokens, token_to_idx


def save_vocab(
    language: str,
    vocab_tokens: List[str],
    counter: Counter,
    output_dir: Path,
    min_freq: int,
) -> Path:
    ensure_dir(output_dir)
    data = {
        "language": language,
        "min_freq": min_freq,
        "size": len(vocab_tokens),
        "tokens": [
            {
                "token": token,
                "freq": int(counter.get(token, 0)),
            }
            for token in vocab_tokens
        ],
    }
    path = output_dir / f"{language}_vocab.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def load_counter(
    language: str,
    tokenized_dir: Path,
) -> Counter:
    counter: Counter = Counter()
    for split in SPLITS:
        if split.name not in TRAIN_SPLIT_NAMES:
            continue
        path = tokenized_dir / f"{split.name}_tokenized.jsonl"
        if not path.exists():
            continue
        for record in read_jsonl(path):
            tokens = record[f"{language}_tokens"]
            counter.update(tokens)
    return counter


def apply_vocab_to_split(
    split: SplitConfig,
    tokenized_dir: Path,
    final_dir: Path,
    vocab_indices: Dict[str, int],
    language: str,
) -> Tuple[int, int]:
    in_path = tokenized_dir / f"{split.name}_tokenized.jsonl"
    out_path = final_dir / f"{split.name}_final.jsonl"
    ensure_dir(final_dir)
    total = 0
    kept = 0
    unk_id = vocab_indices.get("<unk>", 1)
    enriched_records: List[Dict] = []
    for record in read_jsonl(in_path):
        total += 1
        tokens = record[f"{language}_tokens"]
        token_ids = [vocab_indices.get(token, unk_id) for token in tokens]
        key_tokens = f"{language}_tokens"
        key_ids = f"{language}_ids"
        enriched_records.append(
            {
                "index": record["index"],
                key_tokens: tokens,
                key_ids: token_ids,
                f"{language}_length": len(tokens),
            }
        )
        kept += 1
    existing: Dict[int, Dict] = {}
    if out_path.exists():
        for rec in read_jsonl(out_path):
            existing[rec["index"]] = rec
    for rec in enriched_records:
        idx = rec["index"]
        if idx in existing:
            existing[idx].update(rec)
        else:
            existing[idx] = rec
    sorted_records = [existing[idx] for idx in sorted(existing.keys())]
    write_jsonl(out_path, sorted_records)
    return total, kept


def load_pretrained_vectors(path: Path) -> "KeyedVectors | None":  # type: ignore
    if not path or not path.exists() or KeyedVectors is None:
        return None
    try:
        if path.suffix in {".bin", ".vec", ".txt"}:
            return KeyedVectors.load_word2vec_format(str(path), binary=path.suffix == ".bin")  # type: ignore
        return KeyedVectors.load(str(path))  # type: ignore
    except Exception as exc:  # pragma: no cover - optional path
        logging.warning("Failed to load pretrained vectors from %s (%s)", path, exc)
        return None


def init_embeddings(
    language: str,
    vocab_tokens: List[str],
    embedding_dim: int,
    pretrained_path: Path | None,
    random_scale: float = 0.1,
) -> Tuple[np.ndarray, Dict]:
    matrix = np.random.normal(
        loc=0.0,
        scale=random_scale,
        size=(len(vocab_tokens), embedding_dim),
    ).astype(np.float32)
    coverage = 0
    source = None
    vectors = load_pretrained_vectors(pretrained_path) if pretrained_path else None
    if vectors is not None and vectors.vector_size != embedding_dim:
        logging.warning(
            "Embedding dim mismatch (wanted %s, got %s). Using vector size from file.",
            embedding_dim,
            vectors.vector_size,
        )
        embedding_dim = vectors.vector_size
        matrix = np.random.normal(
            loc=0.0, scale=random_scale, size=(len(vocab_tokens), embedding_dim)
        ).astype(np.float32)
    if vectors is not None:
        source = str(pretrained_path)
        for idx, token in enumerate(vocab_tokens):
            if token in vectors:
                matrix[idx] = vectors[token]
                coverage += 1
    else:
        logging.warning(
            "No pretrained vectors found for %s; falling back to random init.", language
        )
    metadata = {
        "dim": embedding_dim,
        "coverage": coverage,
        "vocab_size": len(vocab_tokens),
        "pretrained_source": source,
    }
    return matrix, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess bilingual datasets.")
    parser.add_argument("--raw_dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max_chars_en", type=int, default=400)
    parser.add_argument("--max_chars_zh", type=int, default=400)
    parser.add_argument("--max_tokens_en", type=int, default=80)
    parser.add_argument("--max_tokens_zh", type=int, default=80)
    parser.add_argument("--min_freq_en", type=int, default=3)
    parser.add_argument("--min_freq_zh", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=50)
    parser.add_argument("--pretrained_en", type=Path, default=None)
    parser.add_argument("--pretrained_zh", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cleaned_dir = args.output_dir / "cleaned"
    tokenized_dir = args.output_dir / "tokenized"
    vocab_dir = args.output_dir / "vocab"
    final_dir = args.output_dir / "final"
    embedding_dir = args.output_dir / "embeddings"
    stats_dir = args.output_dir / "stats"
    ensure_dir(args.output_dir)
    ensure_dir(cleaned_dir)
    ensure_dir(tokenized_dir)
    ensure_dir(final_dir)
    ensure_dir(embedding_dir)
    ensure_dir(stats_dir)

    global_stats = {"cleaning": {}, "tokenization": {}, "vocab": {}, "embeddings": {}}

    for split in SPLITS:
        logging.info("Cleaning %s", split.name)
        global_stats["cleaning"][split.name] = clean_split(
            split,
            args.raw_dir,
            cleaned_dir,
            args.max_chars_en,
            args.max_chars_zh,
        )

    for split in SPLITS:
        logging.info("Tokenizing %s", split.name)
        global_stats["tokenization"][split.name] = tokenize_split(
            split,
            cleaned_dir,
            tokenized_dir,
            args.max_tokens_en,
            args.max_tokens_zh,
        )

    en_counter = load_counter("en", tokenized_dir)
    zh_counter = load_counter("zh", tokenized_dir)

    en_vocab_tokens, en_token_to_idx = build_vocab(
        "en", tokenized_dir, args.min_freq_en
    )
    zh_vocab_tokens, zh_token_to_idx = build_vocab(
        "zh", tokenized_dir, args.min_freq_zh
    )
    global_stats["vocab"]["en"] = {
        "size": len(en_vocab_tokens),
        "min_freq": args.min_freq_en,
    }
    global_stats["vocab"]["zh"] = {
        "size": len(zh_vocab_tokens),
        "min_freq": args.min_freq_zh,
    }

    save_vocab("en", en_vocab_tokens, en_counter, vocab_dir, args.min_freq_en)
    save_vocab("zh", zh_vocab_tokens, zh_counter, vocab_dir, args.min_freq_zh)

    for split in SPLITS:
        logging.info("Applying vocab to %s", split.name)
        en_total, en_kept = apply_vocab_to_split(
            split,
            tokenized_dir,
            final_dir,
            en_token_to_idx,
            "en",
        )
        zh_total, zh_kept = apply_vocab_to_split(
            split,
            tokenized_dir,
            final_dir,
            zh_token_to_idx,
            "zh",
        )
        global_stats["vocab"][split.name] = {
            "en_records": en_kept,
            "zh_records": zh_kept,
            "total": max(en_total, zh_total),
        }

    en_matrix, en_meta = init_embeddings(
        "en", en_vocab_tokens, args.embedding_dim, args.pretrained_en
    )
    zh_matrix, zh_meta = init_embeddings(
        "zh", zh_vocab_tokens, args.embedding_dim, args.pretrained_zh
    )

    en_emb_path = embedding_dir / "en_embeddings.npz"
    zh_emb_path = embedding_dir / "zh_embeddings.npz"
    np.savez_compressed(en_emb_path, embeddings=en_matrix)
    np.savez_compressed(zh_emb_path, embeddings=zh_matrix)
    global_stats["embeddings"]["en"] = {"path": str(en_emb_path), **en_meta}
    global_stats["embeddings"]["zh"] = {"path": str(zh_emb_path), **zh_meta}

    stats_path = stats_dir / "processing_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(global_stats, f, ensure_ascii=False, indent=2)
    logging.info("Saved processing stats to %s", stats_path)


if __name__ == "__main__":
    main()
