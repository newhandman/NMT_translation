#!/usr/bin/env python3
"""
Train and evaluate a two-layer RNN-based NMT model with configurable attention,
teacher forcing ratio, and decoding strategy (greedy or beam search).
"""
from __future__ import annotations

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
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from sacrebleu import corpus_bleu, corpus_chrf


PadInfo = Dict[str, int]


def load_vocab(path: Path) -> Tuple[List[str], Dict[str, int]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tokens = [entry["token"] for entry in data["tokens"]]
    mapping = {token: idx for idx, token in enumerate(tokens)}
    return tokens, mapping


def log_metrics(path: Optional[Path], record: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_embeddings(path: Optional[Path], vocab_size: int, emb_dim: int, pad_idx: int) -> torch.Tensor:
    if path and path.exists():
        matrix = np.load(path)["embeddings"]
        if matrix.shape[0] != vocab_size:
            raise ValueError(f"Embedding rows {matrix.shape[0]} != vocab size {vocab_size}")
        emb_dim = matrix.shape[1]
        tensor = torch.tensor(matrix, dtype=torch.float32)
    else:
        tensor = torch.randn(vocab_size, emb_dim) * 0.01
    tensor[pad_idx] = 0.0
    return tensor


def read_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
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
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        example = self.examples[idx]
        decoder_input = [self.bos_id] + example.tgt_ids
        decoder_target = example.tgt_ids + [self.eos_id]
        return {
            "src": example.src_ids,
            "tgt_in": decoder_input,
            "tgt_out": decoder_target,
        }


class Collator:
    def __init__(self, src_pad: int, tgt_pad: int) -> None:
        self.src_pad = src_pad
        self.tgt_pad = tgt_pad

    def __call__(self, batch: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        src_tensors = [torch.tensor(item["src"], dtype=torch.long) for item in batch]
        tgt_in_tensors = [torch.tensor(item["tgt_in"], dtype=torch.long) for item in batch]
        tgt_out_tensors = [torch.tensor(item["tgt_out"], dtype=torch.long) for item in batch]

        src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=self.src_pad)
        tgt_in_padded = pad_sequence(tgt_in_tensors, batch_first=True, padding_value=self.tgt_pad)
        tgt_out_padded = pad_sequence(tgt_out_tensors, batch_first=True, padding_value=self.tgt_pad)
        src_lengths = torch.tensor([len(t) for t in src_tensors], dtype=torch.long)
        tgt_lengths = torch.tensor([len(t) for t in tgt_out_tensors], dtype=torch.long)
        return {
            "src": src_padded,
            "src_lengths": src_lengths,
            "tgt_in": tgt_in_padded,
            "tgt_out": tgt_out_padded,
            "tgt_lengths": tgt_lengths,
        }


def build_eval_loader(dataset: Dataset, pad_info: PadInfo) -> DataLoader:
    collate = Collator(src_pad=pad_info["src_pad"], tgt_pad=pad_info["tgt_pad"])
    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)


def create_src_mask(src: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    batch_size, max_len = src.size()
    mask = torch.arange(max_len, device=src.device).unsqueeze(0).expand(batch_size, max_len)
    mask = mask >= lengths.unsqueeze(1)
    return mask


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        embeddings: torch.Tensor,
        pad_idx: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embeddings.size(1), padding_idx=pad_idx)
        self.embedding.weight.data.copy_(embeddings)
        self.rnn = nn.GRU(
            input_size=embeddings.size(1),
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, mode: str = "dot") -> None:
        super().__init__()
        mode = mode.lower()
        if mode not in {"dot", "multiplicative", "additive"}:
            raise ValueError(f"Unsupported attention mode: {mode}")
        self.mode = mode
        if mode == "multiplicative":
            self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif mode == "additive":
            self.linear = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
            self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "dot":
            scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)
        elif self.mode == "multiplicative":
            proj_query = self.linear(query)
            scores = torch.bmm(keys, proj_query.unsqueeze(2)).squeeze(2)
        else:
            expanded_query = query.unsqueeze(1).expand(-1, keys.size(1), -1)
            energy = torch.tanh(self.linear(torch.cat([expanded_query, keys], dim=-1)))
            scores = self.v(energy).squeeze(-1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embeddings: torch.Tensor,
        hidden_dim: int,
        pad_idx: int,
        attention: Attention,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embeddings.size(1), padding_idx=pad_idx)
        self.embedding.weight.data.copy_(embeddings)
        self.attention = attention
        self.rnn = nn.GRU(
            input_size=embeddings.size(1) + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward_step(
        self,
        input_tokens: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(input_tokens)).unsqueeze(1)
        query = hidden[-1]
        context, attn = self.attention(query, encoder_outputs, mask)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        logits = self.fc(torch.cat([output, context], dim=-1))
        return logits, hidden, attn

    def forward(
        self,
        input_tokens: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_step(input_tokens, hidden, encoder_outputs, mask)


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, pad_idx: int, device: torch.device) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_inputs: torch.Tensor,
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        mask = create_src_mask(src, src_lengths).to(self.device)
        batch_size, seq_len = tgt_inputs.size()
        inputs = tgt_inputs[:, 0]
        logits_collection: List[torch.Tensor] = []
        for t in range(seq_len):
            logits, hidden, _ = self.decoder(inputs, hidden, encoder_outputs, mask)
            logits_collection.append(logits.unsqueeze(1))
            use_teacher = random.random() < teacher_forcing_ratio
            if use_teacher and t + 1 < seq_len:
                inputs = tgt_inputs[:, t + 1]
            else:
                inputs = logits.argmax(dim=-1)
        return torch.cat(logits_collection, dim=1)

    def encode(self, src: torch.Tensor, src_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        mask = create_src_mask(src, src_lengths).to(self.device)
        return encoder_outputs, hidden, mask

    def greedy_decode(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
    ) -> List[int]:
        encoder_outputs, hidden, mask = self.encode(src, src_lengths)
        inputs = torch.tensor([bos_id], device=self.device)
        outputs: List[int] = []
        for _ in range(max_len):
            logits, hidden, _ = self.decoder(inputs, hidden, encoder_outputs, mask)
            next_token = torch.argmax(logits, dim=-1)
            token_id = int(next_token.item())
            if token_id == eos_id:
                break
            outputs.append(token_id)
            inputs = next_token
        return outputs

    def beam_search_decode(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
        beam_size: int,
    ) -> List[int]:
        encoder_outputs, hidden, mask = self.encode(src, src_lengths)
        log_probs = torch.log_softmax
        beams: List[Tuple[List[int], float, torch.Tensor]] = [([bos_id], 0.0, hidden)]
        completed: List[Tuple[List[int], float]] = []
        for _ in range(max_len):
            new_beams: List[Tuple[List[int], float, torch.Tensor]] = []
            for seq, score, h in beams:
                last_token = torch.tensor([seq[-1]], device=self.device)
                logits, next_hidden, _ = self.decoder(last_token, h, encoder_outputs, mask)
                step_log_probs = log_probs(logits, dim=-1)
                top_values, top_indices = torch.topk(step_log_probs, beam_size, dim=-1)
                for value, idx in zip(top_values[0], top_indices[0]):
                    token_id = int(idx.item())
                    new_seq = seq + [token_id]
                    new_score = score + float(value.item())
                    next_h = next_hidden.clone()
                    if token_id == eos_id:
                        completed.append((new_seq, new_score))
                    else:
                        new_beams.append((new_seq, new_score, next_h))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size] or beams
            if not beams:
                break
        all_candidates = completed or [(seq, score) for seq, score, _ in beams]
        best_seq = max(all_candidates, key=lambda x: x[1])[0]
        result = [token for token in best_seq if token not in (bos_id, eos_id)]
        return result[:max_len]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train_file", type=Path, default=Path("artifacts/final/train_10k_final.jsonl"))
    parser.add_argument("--valid_file", type=Path, default=Path("artifacts/final/valid_final.jsonl"))
    parser.add_argument("--test_file", type=Path, default=Path("artifacts/final/test_final.jsonl"))
    parser.add_argument("--src_vocab", type=Path, default=Path("artifacts/vocab/en_vocab.json"))
    parser.add_argument("--tgt_vocab", type=Path, default=Path("artifacts/vocab/zh_vocab.json"))
    parser.add_argument("--src_embeddings", type=Path, default=Path("artifacts/embeddings/en_embeddings.npz"))
    parser.add_argument("--tgt_embeddings", type=Path, default=Path("artifacts/embeddings/zh_embeddings.npz"))
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=1.0)
    parser.add_argument("--attention", type=str, choices=["dot", "multiplicative", "additive"], default="dot")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--max_decode_len", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--load_checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_valid_examples", type=int, default=None)
    parser.add_argument("--max_test_examples", type=int, default=50)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--decode_strategy", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--eval_strategies", type=str, default="greedy",
                        help="Comma-separated list of decoding strategies evaluated each epoch.")
    parser.add_argument("--max_eval_batches", type=int, default=None, help="Limit batches decoded for metric computation.")
    parser.add_argument("--log_file", type=Path, default=Path("logs/rnn_train_metrics.jsonl"))
    parser.add_argument("--eval_log_file", type=Path, default=Path("logs/rnn_eval_metrics.jsonl"))


def build_model(
    args: argparse.Namespace, device: torch.device
) -> Tuple[Seq2Seq, PadInfo, List[str], List[str]]:
    src_tokens, src_mapping = load_vocab(args.src_vocab)
    tgt_tokens, tgt_mapping = load_vocab(args.tgt_vocab)
    pad_idx_src = src_mapping["<pad>"]
    pad_idx_tgt = tgt_mapping["<pad>"]
    bos_id = tgt_mapping["<bos>"]
    eos_id = tgt_mapping["<eos>"]

    src_embeddings = load_embeddings(args.src_embeddings, len(src_tokens), args.hidden_dim, pad_idx_src)
    tgt_embeddings = load_embeddings(args.tgt_embeddings, len(tgt_tokens), args.hidden_dim, pad_idx_tgt)

    encoder = Encoder(
        vocab_size=len(src_tokens),
        emb_dim=src_embeddings.size(1),
        hidden_dim=args.hidden_dim,
        embeddings=src_embeddings,
        pad_idx=pad_idx_src,
        dropout=args.dropout,
    )
    attention = Attention(hidden_dim=args.hidden_dim, mode=args.attention)
    decoder = Decoder(
        vocab_size=len(tgt_tokens),
        embeddings=tgt_embeddings,
        hidden_dim=args.hidden_dim,
        pad_idx=pad_idx_tgt,
        attention=attention,
        dropout=args.dropout,
    )
    model = Seq2Seq(encoder, decoder, pad_idx=pad_idx_tgt, device=device).to(device)

    pad_info = {
        "src_pad": pad_idx_src,
        "tgt_pad": pad_idx_tgt,
        "bos": bos_id,
        "eos": eos_id,
    }
    return model, pad_info, src_tokens, tgt_tokens


def create_dataloaders(
    args: argparse.Namespace,
    pad_info: PadInfo,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = TranslationDataset(
        path=args.train_file,
        src_key="en_ids",
        tgt_key="zh_ids",
        bos_id=pad_info["bos"],
        eos_id=pad_info["eos"],
        max_examples=args.max_train_examples,
    )
    valid_dataset = TranslationDataset(
        path=args.valid_file,
        src_key="en_ids",
        tgt_key="zh_ids",
        bos_id=pad_info["bos"],
        eos_id=pad_info["eos"],
        max_examples=args.max_valid_examples,
    )
    test_dataset = TranslationDataset(
        path=args.test_file,
        src_key="en_ids",
        tgt_key="zh_ids",
        bos_id=pad_info["bos"],
        eos_id=pad_info["eos"],
        max_examples=args.max_test_examples,
    )
    collate = Collator(src_pad=pad_info["src_pad"], tgt_pad=pad_info["tgt_pad"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate)
    return train_loader, valid_loader, test_loader


def run_epoch(
    model: Seq2Seq,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    teacher_forcing_ratio: float,
    clip: float,
) -> Tuple[float, int, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    start_time = time.perf_counter()
    for batch in loader:
        src = batch["src"].to(device)
        src_lengths = batch["src_lengths"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)
        logits = model(src, src_lengths, tgt_in, teacher_forcing_ratio if training else 0.0)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
        if training:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        total_loss += loss.item() * tgt_out.numel()
        total_tokens += tgt_out.numel()
    elapsed = time.perf_counter() - start_time
    avg_loss = total_loss / total_tokens if total_tokens else 0.0
    return avg_loss, total_tokens, elapsed


def save_checkpoint(model: Seq2Seq, optimizer: torch.optim.Optimizer, epoch: int, loss: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )


def load_checkpoint(model: Seq2Seq, optimizer: Optional[torch.optim.Optimizer], path: Path, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint.get("epoch", 0)


def detokenize(ids: List[int], vocab: List[str], skip_ids: set[int]) -> str:
    tokens = [vocab[idx] for idx in ids if 0 <= idx < len(vocab) and idx not in skip_ids]
    return " ".join(tokens).strip()


def generate_translations(
    model: Seq2Seq,
    loader: DataLoader,
    tgt_tokens: List[str],
    pad_info: PadInfo,
    strategy: str,
    beam_size: int,
    max_len: int,
    max_batches: Optional[int] = None,
) -> Tuple[List[Dict[str, str]], float, int]:
    outputs: List[Dict[str, str]] = []
    skip_ids = {pad_info["tgt_pad"], pad_info["bos"], pad_info["eos"]}
    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            src = batch["src"].to(model.device)
            src_lengths = batch["src_lengths"].to(model.device)
            if strategy == "greedy":
                pred_ids = model.greedy_decode(src, src_lengths, pad_info["bos"], pad_info["eos"], max_len)
            else:
                pred_ids = model.beam_search_decode(
                    src, src_lengths, pad_info["bos"], pad_info["eos"], max_len, beam_size
                )
            target_ids = batch["tgt_out"][0].tolist()
            prediction = detokenize(pred_ids, tgt_tokens, skip_ids)
            target = detokenize(target_ids, tgt_tokens, skip_ids)
            outputs.append({"prediction": prediction, "target": target})
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
    elapsed = time.perf_counter() - start_time
    return outputs, elapsed, len(outputs)


def compute_text_metrics(translations: List[Dict[str, str]]) -> Dict[str, float]:
    if not translations:
        return {"bleu": 0.0, "chrf": 0.0}
    preds = [item["prediction"] for item in translations]
    refs = [item["target"] for item in translations]
    bleu = corpus_bleu(preds, [refs]).score
    chrf = corpus_chrf(preds, [refs]).score
    return {"bleu": bleu, "chrf": chrf}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a GRU-based NMT model.")
    add_common_args(parser)
    args = parser.parse_args()

    eval_strategies = [s.strip() for s in args.eval_strategies.split(",") if s.strip()]
    if not eval_strategies:
        eval_strategies = ["greedy"]
    for strategy in eval_strategies:
        if strategy not in {"greedy", "beam"}:
            raise ValueError(f"Unsupported eval strategy: {strategy}")

    device = torch.device(args.device)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model, pad_info, _, tgt_tokens = build_model(args, device)
    train_loader, valid_loader, test_loader = create_dataloaders(args, pad_info)
    valid_eval_loader = build_eval_loader(valid_loader.dataset, pad_info)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_info["tgt_pad"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.load_checkpoint:
        start_epoch = load_checkpoint(model, optimizer, args.load_checkpoint, device)

    if args.mode == "train":
        best_bleu: Dict[str, float] = {strategy: float("-inf") for strategy in eval_strategies}
        for epoch in range(start_epoch, args.epochs):
            train_loss, train_tokens, train_time = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                teacher_forcing_ratio=args.teacher_forcing_ratio,
                clip=args.clip,
            )
            valid_loss, valid_tokens, valid_time = run_epoch(
                model,
                valid_loader,
                criterion,
                optimizer=None,
                device=device,
                teacher_forcing_ratio=0.0,
                clip=args.clip,
            )
            train_throughput = (train_tokens / train_time) if train_time else 0.0
            valid_throughput = (valid_tokens / valid_time) if valid_time else 0.0
            print(
                f"Epoch {epoch+1}/{args.epochs} - train loss: {train_loss:.4f} - "
                f"train ppl: {math.exp(train_loss):.2f} - train tok/s: {train_throughput:.1f} - "
                f"valid loss: {valid_loss:.4f} - valid ppl: {math.exp(valid_loss):.2f} - "
                f"valid tok/s: {valid_throughput:.1f}"
            )
            for strategy in eval_strategies:
                translations, decode_time, decoded = generate_translations(
                    model,
                    valid_eval_loader,
                    tgt_tokens=tgt_tokens,
                    pad_info=pad_info,
                    strategy=strategy,
                    beam_size=args.beam_size,
                    max_len=args.max_decode_len,
                    max_batches=args.max_eval_batches,
                )
                metrics = compute_text_metrics(translations)
                avg_decode = (decode_time / decoded) if decoded else 0.0
                print(
                    f"Validation [{strategy}] BLEU: {metrics['bleu']:.2f} - chrF: {metrics['chrf']:.2f} "
                    f"(decoded {decoded} samples, avg decode {avg_decode:.3f}s)"
                )
                log_metrics(
                    args.log_file,
                    {
                        "timestamp": time.time(),
                        "epoch": epoch + 1,
                        "mode": "train",
                        "train_loss": train_loss,
                        "train_ppl": math.exp(train_loss) if train_loss else 0.0,
                        "train_tok_per_sec": train_throughput,
                        "valid_loss": valid_loss,
                        "valid_ppl": math.exp(valid_loss) if valid_loss else 0.0,
                        "valid_tok_per_sec": valid_throughput,
                        "valid_bleu": metrics["bleu"],
                        "valid_chrf": metrics["chrf"],
                        "decoded_samples": decoded,
                        "avg_decode_time": avg_decode,
                        "teacher_forcing_ratio": args.teacher_forcing_ratio,
                        "attention": args.attention,
                        "decode_strategy": strategy,
                    },
                )
                if metrics["bleu"] > best_bleu[strategy]:
                    best_bleu[strategy] = metrics["bleu"]
                    checkpoint_path = (
                        args.save_dir
                        / f"nmt_{args.attention}_{strategy}_tf{args.teacher_forcing_ratio:.2f}.pt"
                    )
                    save_checkpoint(model, optimizer, epoch + 1, metrics["bleu"], checkpoint_path)
                    print(f"Saved best {strategy} checkpoint to {checkpoint_path}")
    else:
        translations, decode_time, decoded = generate_translations(
            model,
            test_loader,
            tgt_tokens=tgt_tokens,
            pad_info=pad_info,
            strategy=args.decode_strategy,
            beam_size=args.beam_size,
            max_len=args.max_decode_len,
            max_batches=args.max_eval_batches,
        )
        metrics = compute_text_metrics(translations)
        avg_decode = (decode_time / decoded) if decoded else 0.0
        print(
            f"Test BLEU: {metrics['bleu']:.2f} - Test chrF: {metrics['chrf']:.2f} "
            f"(decoded {decoded} samples, avg decode {avg_decode:.3f}s)"
        )
        log_metrics(
            args.eval_log_file,
            {
                "timestamp": time.time(),
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
            print(f"Sample {idx+1}:")
            print("  Prediction:", item["prediction"])
            print("  Target    :", item["target"])


if __name__ == "__main__":
    main()
