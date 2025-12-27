#!/usr/bin/env python3
"""
Train and evaluate Transformer-based NMT models (from scratch or pretrained).
Supports positional embedding ablations, normalization variants, and decoding policies.
"""
from __future__ import annotations

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sacrebleu import corpus_bleu, corpus_chrf
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoTokenizer, T5ForConditionalGeneration
except Exception:  # pragma: no cover - optional dependency
    AutoTokenizer = None  # type: ignore
    T5ForConditionalGeneration = None  # type: ignore


PadInfo = Dict[str, int]


def load_vocab(path: Path) -> Tuple[List[str], Dict[str, int]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tokens = [item["token"] for item in data["tokens"]]
    mapping = {token: idx for idx, token in enumerate(tokens)}
    return tokens, mapping


def log_metrics(path: Optional[Path], record: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@dataclass
class Example:
    src_ids: List[int]
    tgt_ids: List[int]


class TranslationDataset(Dataset):
    def __init__(
        self,
        path: Path,
        src_key: str,
        tgt_key: str,
        bos_id: int,
        eos_id: int,
        max_examples: Optional[int] = None,
    ) -> None:
        records = read_jsonl(path)
        if max_examples:
            records = records[:max_examples]
        self.examples: List[Example] = []
        for record in records:
            src_ids = list(record[src_key])
            tgt_ids = list(record[tgt_key])
            self.examples.append(Example(src_ids=src_ids, tgt_ids=tgt_ids))
        self.bos = bos_id
        self.eos = eos_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        item = self.examples[idx]
        tgt_in = [self.bos] + item.tgt_ids
        tgt_out = item.tgt_ids + [self.eos]
        return {"src": item.src_ids, "tgt_in": tgt_in, "tgt_out": tgt_out}


class Collator:
    def __init__(self, src_pad: int, tgt_pad: int) -> None:
        self.src_pad = src_pad
        self.tgt_pad = tgt_pad

    def __call__(self, batch: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        src = [torch.tensor(item["src"], dtype=torch.long) for item in batch]
        tgt_in = [torch.tensor(item["tgt_in"], dtype=torch.long) for item in batch]
        tgt_out = [torch.tensor(item["tgt_out"], dtype=torch.long) for item in batch]
        src_pad = pad_sequence(src, batch_first=True, padding_value=self.src_pad)
        tgt_in_pad = pad_sequence(tgt_in, batch_first=True, padding_value=self.tgt_pad)
        tgt_out_pad = pad_sequence(tgt_out, batch_first=True, padding_value=self.tgt_pad)
        src_mask = (src_pad == self.src_pad)
        tgt_mask = (tgt_in_pad == self.tgt_pad)
        return {
            "src": src_pad,
            "tgt_in": tgt_in_pad,
            "tgt_out": tgt_out_pad,
            "src_pad_mask": src_mask,
            "tgt_pad_mask": tgt_mask,
        }


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.scale


def get_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    return nn.LayerNorm(dim)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.embedding(positions)[None, :, :]


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, max_distance: int = 128, num_buckets: int = 32) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.num_buckets = num_buckets
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        n = relative_position
        half = num_buckets // 2
        is_small = (n.abs() < 8).to(torch.long)
        val_if_large = (
            (torch.log(n.abs().float() / 8 + 1e-6) / math.log(max_distance / 8))
            * (half - 8)
        ).to(torch.long) + 8
        val_if_large = torch.clamp(val_if_large, max=half - 1)
        result = n.abs() * is_small + val_if_large * (1 - is_small)
        result = torch.where(n > 0, half + result, result)
        return result

    def forward(self, length_q: int, length_k: int, device: torch.device) -> torch.Tensor:
        context_position = torch.arange(length_q, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(length_k, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(rp_bucket)
        return values.permute(2, 0, 1)  # num_heads, Lq, Lk


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        rel_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if rel_bias is not None:
            scores = scores + rel_bias.unsqueeze(0)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].expand(-1, self.num_heads, seq_len_q, -1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_ff: int,
        dropout: float,
        norm_type: str,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, dim_ff, dropout)
        self.norm1 = get_norm(norm_type, d_model)
        self.norm2 = get_norm(norm_type, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        rel_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, src_key_padding_mask, attn_mask, rel_bias)
        x = residual + self.dropout(x)
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        return residual + self.dropout(x)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_ff: int,
        dropout: float,
        norm_type: str,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, dim_ff, dropout)
        self.norm1 = get_norm(norm_type, d_model)
        self.norm2 = get_norm(norm_type, d_model)
        self.norm3 = get_norm(norm_type, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor],
        rel_bias_self: Optional[torch.Tensor],
        rel_bias_cross: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, tgt_key_padding_mask, self_attn_mask, rel_bias_self)
        x = residual + self.dropout(x)
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, memory, memory, src_key_padding_mask, None, rel_bias_cross)
        x = residual + self.dropout(x)
        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        return residual + self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        pad_src: int,
        pad_tgt: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_ff: int,
        dropout: float,
        max_seq_len: int,
        positional: str,
        norm_type: str,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_src)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_tgt)
        self.pad_tgt = pad_tgt
        self.positional = positional
        if positional == "sinusoidal":
            self.pos_enc_src = PositionalEncoding(d_model, max_seq_len)
            self.pos_enc_tgt = PositionalEncoding(d_model, max_seq_len)
            self.rel_bias = None
        elif positional == "learned":
            self.pos_enc_src = LearnedPositionalEmbedding(max_seq_len, d_model)
            self.pos_enc_tgt = LearnedPositionalEmbedding(max_seq_len, d_model)
            self.rel_bias = None
        else:
            self.pos_enc_src = None
            self.pos_enc_tgt = None
            self.rel_bias = RelativePositionBias(num_heads, max_distance=max_seq_len, num_buckets=32)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, dim_ff, dropout, norm_type)
                for _ in range(num_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, dim_ff, dropout, norm_type)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = get_norm(norm_type, d_model)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src: torch.Tensor, src_pad_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_embed(src) * math.sqrt(self.d_model)
        if self.pos_enc_src is not None:
            x = self.pos_enc_src(x)
        x = self.dropout(x)
        rel_bias = None
        if self.rel_bias:
            rel_bias = self.rel_bias(src.size(1), src.size(1), src.device)
        for layer in self.encoder_layers:
            x = layer(x, src_pad_mask, None, rel_bias)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        if self.pos_enc_tgt is not None:
            x = self.pos_enc_tgt(x)
        x = self.dropout(x)
        seq_len = tgt.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=tgt.device, dtype=torch.bool), diagonal=1
        )
        rel_bias_self = self.rel_bias(seq_len, seq_len, tgt.device) if self.rel_bias else None
        rel_bias_cross = self.rel_bias(seq_len, memory.size(1), tgt.device) if self.rel_bias else None
        for layer in self.decoder_layers:
            x = layer(
                x,
                memory,
                tgt_pad_mask,
                src_pad_mask,
                causal_mask,
                rel_bias_self,
                rel_bias_cross,
            )
        x = self.final_norm(x)
        return self.generator(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_pad_mask)
        logits = self.decode(tgt_in, memory, src_pad_mask, tgt_pad_mask)
        return logits

    def greedy_decode(
        self,
        src: torch.Tensor,
        src_pad_mask: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
    ) -> List[int]:
        memory = self.encode(src, src_pad_mask)
        ys = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
        tgt_pad_mask = ys.eq(self.pad_tgt)
        outputs: List[int] = []
        for _ in range(max_len):
            logits = self.decode(ys, memory, src_pad_mask, tgt_pad_mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            token_id = int(next_token.item())
            if token_id == eos_id:
                break
            outputs.append(token_id)
            ys = torch.cat([ys, next_token], dim=1)
            tgt_pad_mask = ys.eq(self.pad_tgt)
        return outputs

    def beam_search_decode(
        self,
        src: torch.Tensor,
        src_pad_mask: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
        beam_size: int,
    ) -> List[int]:
        memory = self.encode(src, src_pad_mask)
        beams = [
            {
                "tokens": torch.tensor([[bos_id]], dtype=torch.long, device=src.device),
                "score": 0.0,
            }
        ]
        completed: List[Tuple[torch.Tensor, float]] = []
        for _ in range(max_len):
            new_beams = []
            for beam in beams:
                tokens = beam["tokens"]
                logits = self.decode(tokens, memory, src_pad_mask, tokens.eq(self.pad_tgt))
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                topk = torch.topk(log_probs, beam_size, dim=-1)
                for score, idx in zip(topk.values[0], topk.indices[0]):
                    next_tokens = torch.cat([tokens, idx.view(1, 1)], dim=1)
                    new_score = beam["score"] + float(score.item())
                    if int(idx.item()) == eos_id:
                        completed.append((next_tokens, new_score))
                    else:
                        new_beams.append({"tokens": next_tokens, "score": new_score})
            beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)[:beam_size]
            if not beams:
                break
        if not completed:
            completed = [(beam["tokens"], beam["score"]) for beam in beams]
        best = max(completed, key=lambda x: x[1])[0].squeeze(0).tolist()
        return [tok for tok in best if tok not in (bos_id, eos_id)][-max_len:]


def create_dataloaders(
    args: argparse.Namespace,
    pad_info: PadInfo,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = TranslationDataset(
        path=args.train_file,
        src_key=args.src_field,
        tgt_key=args.tgt_field,
        bos_id=pad_info["tgt_bos"],
        eos_id=pad_info["tgt_eos"],
        max_examples=args.max_train_examples,
    )
    valid_dataset = TranslationDataset(
        path=args.valid_file,
        src_key=args.src_field,
        tgt_key=args.tgt_field,
        bos_id=pad_info["tgt_bos"],
        eos_id=pad_info["tgt_eos"],
        max_examples=args.max_valid_examples,
    )
    test_dataset = TranslationDataset(
        path=args.test_file,
        src_key=args.src_field,
        tgt_key=args.tgt_field,
        bos_id=pad_info["tgt_bos"],
        eos_id=pad_info["tgt_eos"],
        max_examples=args.max_test_examples,
    )
    collator = Collator(pad_info["src_pad"], pad_info["tgt_pad"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collator)
    return train_loader, valid_loader, test_loader


def detokenize(ids: List[int], vocab: List[str], skip_ids: set[int]) -> str:
    return " ".join(vocab[idx] for idx in ids if 0 <= idx < len(vocab) and idx not in skip_ids).strip()


def generate_outputs(
    model: TransformerModel,
    loader: DataLoader,
    pad_info: PadInfo,
    tgt_vocab: List[str],
    strategy: str,
    beam_size: int,
    max_len: int,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[List[Dict[str, str]], float, int]:
    outputs: List[Dict[str, str]] = []
    skip_ids = {pad_info["tgt_pad"], pad_info["tgt_bos"], pad_info["tgt_eos"]}
    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            src = batch["src"].to(device)
            src_mask = batch["src_pad_mask"].to(device)
            tgt_out = batch["tgt_out"]
            batch_size = src.size(0)
            for b in range(batch_size):
                sample_src = src[b : b + 1]
                sample_mask = src_mask[b : b + 1]
                gold_ids = tgt_out[b].tolist()
                if strategy == "greedy":
                    pred_ids = model.greedy_decode(
                        sample_src, sample_mask, pad_info["tgt_bos"], pad_info["tgt_eos"], max_len
                    )
                else:
                    pred_ids = model.beam_search_decode(
                        sample_src, sample_mask, pad_info["tgt_bos"], pad_info["tgt_eos"], max_len, beam_size
                    )
                outputs.append(
                    {
                        "prediction": detokenize(pred_ids, tgt_vocab, skip_ids),
                        "target": detokenize(gold_ids, tgt_vocab, skip_ids),
                    }
                )
                if max_batches and len(outputs) >= max_batches:
                    elapsed = time.perf_counter() - start_time
                    return outputs, elapsed, len(outputs)
    elapsed = time.perf_counter() - start_time
    return outputs, elapsed, len(outputs)


def compute_metrics(translations: List[Dict[str, str]]) -> Dict[str, float]:
    if not translations:
        return {"bleu": 0.0, "chrf": 0.0}
    predictions = [item["prediction"] for item in translations]
    references = [item["target"] for item in translations]
    bleu = corpus_bleu(predictions, [references]).score
    chrf = corpus_chrf(predictions, [references]).score
    return {"bleu": bleu, "chrf": chrf}


def build_checkpoint_tag(args: argparse.Namespace, strategy: str) -> str:
    lr_str = f"{args.lr:g}".replace(".", "p")
    return (
        f"transformer_{args.positional_encoding}_{args.norm_type}_d{args.d_model}_"
        f"h{args.num_heads}_l{args.num_layers}_ff{args.ff_dim}_bs{args.batch_size}_"
        f"lr{lr_str}_{strategy}"
    )


class T5Dataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        src_field: str,
        tgt_field: str,
        max_length: int,
        max_examples: Optional[int],
    ) -> None:
        records = read_jsonl(path)
        if max_examples:
            records = records[:max_examples]
        self.inputs: List[Dict[str, torch.Tensor]] = []
        self.references: List[torch.Tensor] = []
        for record in records:
            src_tokens = " ".join(record[src_field[:-4] + "_tokens"]) if src_field.endswith("ids") else " ".join(
                record[src_field]
            )
            tgt_tokens = " ".join(record[tgt_field[:-4] + "_tokens"]) if tgt_field.endswith("ids") else " ".join(
                record[tgt_field]
            )
            encoded = tokenizer(
                src_tokens,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            with tokenizer.as_target_tokenizer():
                target = tokenizer(
                    tgt_tokens,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            label_ids = target["input_ids"].squeeze(0)
            ref_ids = label_ids.clone()
            label_ids = label_ids.masked_fill(label_ids == tokenizer.pad_token_id, -100)
            self.inputs.append(
                {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "labels": label_ids,
                    "ref_ids": ref_ids,
                }
            )
            self.references.append(ref_ids)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.inputs[idx]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train_file", type=Path, default=Path("artifacts/final/train_100k_final.jsonl"))
    parser.add_argument("--valid_file", type=Path, default=Path("artifacts/final/valid_final.jsonl"))
    parser.add_argument("--test_file", type=Path, default=Path("artifacts/final/test_final.jsonl"))
    parser.add_argument("--src_vocab", type=Path, default=Path("artifacts/vocab/zh_vocab.json"))
    parser.add_argument("--tgt_vocab", type=Path, default=Path("artifacts/vocab/en_vocab.json"))
    parser.add_argument("--src_field", type=str, default="zh_ids")
    parser.add_argument("--tgt_field", type=str, default="en_ids")
    parser.add_argument("--model_variant", choices=["scratch", "pretrained"], default="scratch")
    parser.add_argument("--positional_encoding", choices=["sinusoidal", "learned", "relative"], default="sinusoidal")
    parser.add_argument("--norm_type", choices=["layernorm", "rmsnorm"], default="layernorm")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--ff_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--max_decode_len", type=int, default=80)
    parser.add_argument("--decode_strategy", choices=["greedy", "beam"], default="greedy")
    parser.add_argument(
        "--eval_strategies",
        type=str,
        default="greedy",
        help="Comma-separated decoding strategies evaluated on validation data each epoch.",
    )
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_valid_examples", type=int, default=None)
    parser.add_argument("--max_test_examples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--load_checkpoint", type=Path, default=None)
    parser.add_argument("--pretrained_model_name", type=str, default="t5-small")
    parser.add_argument("--t5_max_length", type=int, default=128)
    parser.add_argument("--t5_lr", type=float, default=3e-4)
    parser.add_argument("--log_file", type=Path, default=Path("logs/transformer_train_metrics.jsonl"))
    parser.add_argument("--eval_log_file", type=Path, default=Path("logs/transformer_eval_metrics.jsonl"))


def train_scratch(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    src_tokens, src_mapping = load_vocab(args.src_vocab)
    tgt_tokens, tgt_mapping = load_vocab(args.tgt_vocab)
    pad_info = {
        "src_pad": src_mapping["<pad>"],
        "tgt_pad": tgt_mapping["<pad>"],
        "tgt_bos": tgt_mapping["<bos>"],
        "tgt_eos": tgt_mapping["<eos>"],
    }
    model = TransformerModel(
        src_vocab_size=len(src_tokens),
        tgt_vocab_size=len(tgt_tokens),
        pad_src=pad_info["src_pad"],
        pad_tgt=pad_info["tgt_pad"],
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_ff=args.ff_dim,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        positional=args.positional_encoding,
        norm_type=args.norm_type,
    ).to(device)
    train_loader, valid_loader, test_loader = create_dataloaders(args, pad_info)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_info["tgt_pad"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    start_epoch = 0
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0)
    if args.mode == "train":
        best_bleu: Dict[str, float] = {strategy: float("-inf") for strategy in args.eval_strategy_list}
        for epoch in range(start_epoch, args.epochs):
            model.train()
            total_loss = 0.0
            tokens = 0
            epoch_start = time.perf_counter()
            for batch in train_loader:
                src = batch["src"].to(device)
                tgt_in = batch["tgt_in"].to(device)
                tgt_out = batch["tgt_out"].to(device)
                src_pad_mask = batch["src_pad_mask"].to(device)
                tgt_pad_mask = batch["tgt_pad_mask"].to(device)
                logits = model(src, tgt_in, src_pad_mask, tgt_pad_mask)
                loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                total_loss += loss.item() * tgt_out.numel()
                tokens += tgt_out.numel()
            avg_loss = total_loss / tokens
            epoch_time = time.perf_counter() - epoch_start
            throughput = (tokens / epoch_time) if epoch_time else 0.0
            print(
                f"Epoch {epoch+1}/{args.epochs} - train loss: {avg_loss:.4f} - ppl: {math.exp(avg_loss):.2f} "
                f"- tok/s: {throughput:.1f}"
            )
            for strategy in args.eval_strategy_list:
                translations, decode_time, decoded = generate_outputs(
                    model,
                    valid_loader,
                    pad_info,
                    tgt_tokens,
                    strategy,
                    args.beam_size,
                    args.max_decode_len,
                    device,
                    args.max_eval_batches,
                )
                metrics = compute_metrics(translations)
                avg_decode = (decode_time / decoded) if decoded else 0.0
                print(
                    f"Validation [{strategy}] BLEU: {metrics['bleu']:.2f} - chrF: {metrics['chrf']:.2f} "
                    f"(decoded {decoded} samples, avg decode {avg_decode:.3f}s)"
                )
                log_metrics(
                    args.log_file,
                    {
                        "timestamp": time.time(),
                        "variant": "scratch",
                        "epoch": epoch + 1,
                        "train_loss": avg_loss,
                        "train_ppl": math.exp(avg_loss),
                        "train_tok_per_sec": throughput,
                        "valid_bleu": metrics["bleu"],
                        "valid_chrf": metrics["chrf"],
                        "decoded_samples": decoded,
                        "avg_decode_time": avg_decode,
                        "positional_encoding": args.positional_encoding,
                        "norm_type": args.norm_type,
                        "num_layers": args.num_layers,
                        "num_heads": args.num_heads,
                        "ff_dim": args.ff_dim,
                        "decode_strategy": strategy,
                    },
                )
                if metrics["bleu"] > best_bleu[strategy]:
                    best_bleu[strategy] = metrics["bleu"]
                    save_path = args.save_dir / f"{build_checkpoint_tag(args, strategy)}.pt"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "epoch": epoch + 1,
                        },
                        save_path,
                    )
                    print(f"Saved best {strategy} checkpoint to {save_path}")
    else:
        loader = test_loader
        if args.load_checkpoint:
            checkpoint = torch.load(args.load_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
        translations, decode_time, decoded = generate_outputs(
            model,
            loader,
            pad_info,
            tgt_tokens,
            args.decode_strategy,
            args.beam_size,
            args.max_decode_len,
            device,
            args.max_eval_batches,
        )
        metrics = compute_metrics(translations)
        avg_decode = (decode_time / decoded) if decoded else 0.0
        print(
            f"Test BLEU: {metrics['bleu']:.2f} - chrF: {metrics['chrf']:.2f} "
            f"(decoded {decoded} samples, avg decode {avg_decode:.3f}s)"
        )
        log_metrics(
            args.eval_log_file,
            {
                "timestamp": time.time(),
                "variant": "scratch",
                "mode": "eval",
                "decoded_samples": decoded,
                "avg_decode_time": avg_decode,
                "test_bleu": metrics["bleu"],
                "test_chrf": metrics["chrf"],
                "decode_strategy": args.decode_strategy,
                "beam_size": args.beam_size,
                "max_decode_len": args.max_decode_len,
            },
        )
        for idx, item in enumerate(translations[: min(5, decoded)]):
            print(f"[{idx+1}] Prediction: {item['prediction']}")
            print(f"    Target: {item['target']}")


def train_pretrained(args: argparse.Namespace) -> None:
    if AutoTokenizer is None or T5ForConditionalGeneration is None:
        raise ImportError("transformers library is required for pretrained mode.")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_name).to(device)
    train_dataset = T5Dataset(
        path=args.train_file,
        tokenizer=tokenizer,
        src_field=args.src_field,
        tgt_field=args.tgt_field,
        max_length=args.t5_max_length,
        max_examples=args.max_train_examples,
    )
    valid_dataset = T5Dataset(
        path=args.valid_file,
        tokenizer=tokenizer,
        src_field=args.src_field,
        tgt_field=args.tgt_field,
        max_length=args.t5_max_length,
        max_examples=args.max_valid_examples,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.t5_lr)
    if args.mode == "train":
        best_bleu: Dict[str, float] = {strategy: float("-inf") for strategy in args.eval_strategy_list}
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            total_samples = 0
            epoch_start = time.perf_counter()
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items() if k != "ref_ids"}
                outputs = model(**batch)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_samples += batch["input_ids"].size(0)
            epoch_time = time.perf_counter() - epoch_start
            avg_loss = total_loss / len(train_loader)
            throughput = (total_samples / epoch_time) if epoch_time else 0.0
            print(
                f"T5 Epoch {epoch+1}/{args.epochs} - train loss: {avg_loss:.4f} "
                f"- samples/s: {throughput:.1f}"
            )
            for strategy in args.eval_strategy_list:
                model.eval()
                translations = []
                eval_start = time.perf_counter()
                with torch.no_grad():
                    for batch in valid_loader:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        ref_ids = batch["ref_ids"]
                        generated = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=args.max_decode_len,
                            num_beams=args.beam_size if strategy == "beam" else 1,
                        )
                        decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
                        decoded_refs = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
                        translations.extend(
                            {"prediction": pred.strip(), "target": ref.strip()}
                            for pred, ref in zip(decoded_preds, decoded_refs)
                        )
                        if args.max_eval_batches and len(translations) >= args.max_eval_batches:
                            break
                metrics = compute_metrics(translations)
                eval_time = time.perf_counter() - eval_start
                avg_decode = (eval_time / len(translations)) if translations else 0.0
                print(
                    f"T5 Validation [{strategy}] BLEU: {metrics['bleu']:.2f} - chrF: {metrics['chrf']:.2f} "
                    f"(decoded {len(translations)} samples, avg decode {avg_decode:.3f}s)"
                )
                log_metrics(
                    args.log_file,
                    {
                        "timestamp": time.time(),
                        "variant": "t5",
                        "epoch": epoch + 1,
                        "train_loss": avg_loss,
                        "train_samples_per_sec": throughput,
                        "valid_bleu": metrics["bleu"],
                        "valid_chrf": metrics["chrf"],
                        "decoded_samples": len(translations),
                        "avg_decode_time": avg_decode,
                        "decode_strategy": strategy,
                        "pretrained_model": args.pretrained_model_name,
                    },
                )
                if metrics["bleu"] > best_bleu[strategy]:
                    best_bleu[strategy] = metrics["bleu"]
                    save_path = args.save_dir / f"t5_{args.pretrained_model_name}_{strategy}.pt"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "epoch": epoch + 1,
                        },
                        save_path,
                    )
                    print(f"Saved best T5 {strategy} checkpoint to {save_path}")
    else:
        if args.load_checkpoint:
            model.load_state_dict(torch.load(args.load_checkpoint, map_location=device)["model_state"])
        test_dataset = T5Dataset(
            path=args.test_file,
            tokenizer=tokenizer,
            src_field=args.src_field,
            tgt_field=args.tgt_field,
            max_length=args.t5_max_length,
            max_examples=args.max_test_examples,
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        translations = []
        model.eval()
        eval_start = time.perf_counter()
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                ref_ids = batch["ref_ids"]
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=args.max_decode_len,
                    num_beams=args.beam_size if args.decode_strategy == "beam" else 1,
                )
                decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
                decoded_refs = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
                translations.extend(
                    {"prediction": pred.strip(), "target": ref.strip()}
                    for pred, ref in zip(decoded_preds, decoded_refs)
                )
                if args.max_eval_batches and len(translations) >= args.max_eval_batches:
                    break
        metrics = compute_metrics(translations)
        eval_time = time.perf_counter() - eval_start
        avg_decode = (eval_time / len(translations)) if translations else 0.0
        print(
            f"T5 Test BLEU: {metrics['bleu']:.2f} - chrF: {metrics['chrf']:.2f} "
            f"(decoded {len(translations)} samples, avg decode {avg_decode:.3f}s)"
        )
        log_metrics(
            args.eval_log_file,
            {
                "timestamp": time.time(),
                "variant": "t5",
                "mode": "eval",
                "decoded_samples": len(translations),
                "avg_decode_time": avg_decode,
                "test_bleu": metrics["bleu"],
                "test_chrf": metrics["chrf"],
                "decode_strategy": args.decode_strategy,
                "pretrained_model": args.pretrained_model_name,
            },
        )
        for idx, item in enumerate(translations[:5]):
            print(f"[{idx+1}] Prediction: {item['prediction']}")
            print(f"    Target: {item['target']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer-based NMT training script.")
    add_common_args(parser)
    args = parser.parse_args()
    eval_strategies = [s.strip() for s in args.eval_strategies.split(",") if s.strip()]
    if not eval_strategies:
        eval_strategies = ["greedy"]
    for strategy in eval_strategies:
        if strategy not in {"greedy", "beam"}:
            raise ValueError(f"Unsupported eval strategy: {strategy}")
    setattr(args, "eval_strategy_list", eval_strategies)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if args.model_variant == "scratch":
        train_scratch(args)
    else:
        train_pretrained(args)


if __name__ == "__main__":
    main()
