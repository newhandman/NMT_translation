#!/usr/bin/env python3
"""
Batch evaluation script for all Seq2Seq (RNN) and Transformer checkpoints.
Iterates through checkpoints/ and measures BLEU/chrF on the test split.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

import train_nmt
import train_transformer
from torch.utils.data import DataLoader


def parse_rnn_metadata(stem: str) -> Dict[str, str]:
    suffix = stem[len("nmt_") :]
    attention = suffix.split("_beam_")[0] if "_beam_" in suffix else suffix.split("_tf")[0]
    decode_strategy = "beam" if "_beam_" in suffix else "greedy"
    tf_part = stem.split("_tf")[-1]
    return {
        "attention": attention,
        "decode_strategy": decode_strategy,
        "teacher_forcing": tf_part,
    }


def parse_transformer_metadata(stem: str) -> Dict[str, str]:
    body = stem[len("transformer_") :]
    parts = body.split("_")
    strategy = parts[-1]
    core = parts[:-1]
    positional = core[0] if core else "sinusoidal"
    norm = core[1] if len(core) > 1 else "layernorm"
    dims = {"d_model": 512, "num_heads": 8, "num_layers": 6, "ff_dim": 2048}
    for token in core[2:]:
        if token.startswith("d") and token[1:].isdigit():
            dims["d_model"] = int(token[1:])
        elif token.startswith("h") and token[1:].isdigit():
            dims["num_heads"] = int(token[1:])
        elif token.startswith("l") and token[1:].isdigit():
            dims["num_layers"] = int(token[1:])
        elif token.startswith("ff") and token[2:].isdigit():
            dims["ff_dim"] = int(token[2:])
    return {
        "positional_encoding": positional,
        "norm_type": norm,
        "decode_strategy": strategy,
        **dims,
    }


def parse_t5_metadata(stem: str) -> Dict[str, str]:
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected T5 checkpoint name: {stem}")
    model_name = "_".join(parts[1:-1])
    strategy = parts[-1]
    return {"model_name": model_name, "decode_strategy": strategy}


def build_rnn_defaults() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    train_nmt.add_common_args(parser)
    return parser.parse_args([])


def build_transformer_defaults() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    train_transformer.add_common_args(parser)
    return parser.parse_args([])


def evaluate_rnn_checkpoint(
    ckpt: Path,
    meta: Dict[str, str],
    args: argparse.Namespace,
    device: torch.device,
    max_test_examples: Optional[int],
    max_eval_batches: Optional[int],
) -> Dict[str, float]:
    cfg = build_rnn_defaults()
    cfg.device = str(device)
    cfg.mode = "eval"
    cfg.load_checkpoint = ckpt
    cfg.attention = meta["attention"]
    cfg.decode_strategy = meta["decode_strategy"]
    cfg.max_test_examples = max_test_examples
    cfg.max_eval_batches = max_eval_batches
    cfg.beam_size = args.beam_size
    cfg.max_decode_len = args.max_decode_len
    model, pad_info, _, tgt_tokens = train_nmt.build_model(cfg, device)
    train_nmt.load_checkpoint(model, optimizer=None, path=ckpt, device=device)
    test_dataset = train_nmt.TranslationDataset(
        path=cfg.test_file,
        src_key="en_ids",
        tgt_key="zh_ids",
        bos_id=pad_info["bos"],
        eos_id=pad_info["eos"],
        max_examples=max_test_examples,
    )
    test_loader = train_nmt.build_eval_loader(test_dataset, pad_info)
    translations, decode_time, decoded = train_nmt.generate_translations(
        model,
        test_loader,
        tgt_tokens,
        pad_info,
        cfg.decode_strategy,
        cfg.beam_size,
        cfg.max_decode_len,
        max_eval_batches,
    )
    metrics = train_nmt.compute_text_metrics(translations)
    return {
        "bleu": metrics["bleu"],
        "chrf": metrics["chrf"],
        "decoded_samples": decoded,
        "avg_decode_time": (decode_time / decoded) if decoded else 0.0,
    }


def evaluate_transformer_checkpoint(
    ckpt: Path,
    meta: Dict[str, str],
    device: torch.device,
    max_test_examples: Optional[int],
    max_eval_batches: Optional[int],
    beam_size: int,
    max_decode_len: int,
) -> Dict[str, float]:
    cfg = build_transformer_defaults()
    cfg.device = str(device)
    cfg.mode = "eval"
    cfg.model_variant = "scratch"
    cfg.load_checkpoint = ckpt
    cfg.positional_encoding = meta["positional_encoding"]
    cfg.norm_type = meta["norm_type"]
    cfg.d_model = meta["d_model"]
    cfg.num_heads = meta["num_heads"]
    cfg.num_layers = meta["num_layers"]
    cfg.ff_dim = meta["ff_dim"]
    cfg.decode_strategy = meta["decode_strategy"]
    cfg.beam_size = beam_size
    cfg.max_decode_len = max_decode_len
    cfg.max_test_examples = max_test_examples
    cfg.max_eval_batches = max_eval_batches
    device = torch.device(cfg.device)
    src_tokens, src_mapping = train_transformer.load_vocab(cfg.src_vocab)
    tgt_tokens, tgt_mapping = train_transformer.load_vocab(cfg.tgt_vocab)
    pad_info = {
        "src_pad": src_mapping["<pad>"],
        "tgt_pad": tgt_mapping["<pad>"],
        "tgt_bos": tgt_mapping["<bos>"],
        "tgt_eos": tgt_mapping["<eos>"],
    }
    model = train_transformer.TransformerModel(
        src_vocab_size=len(src_tokens),
        tgt_vocab_size=len(tgt_tokens),
        pad_src=pad_info["src_pad"],
        pad_tgt=pad_info["tgt_pad"],
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dim_ff=cfg.ff_dim,
        dropout=cfg.dropout,
        max_seq_len=cfg.max_seq_len,
        positional=cfg.positional_encoding,
        norm_type=cfg.norm_type,
    ).to(device)
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_dataset = train_transformer.TranslationDataset(
        path=cfg.test_file,
        src_key=cfg.src_field,
        tgt_key=cfg.tgt_field,
        bos_id=pad_info["tgt_bos"],
        eos_id=pad_info["tgt_eos"],
        max_examples=max_test_examples,
    )
    collator = train_transformer.Collator(pad_info["src_pad"], pad_info["tgt_pad"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collator)
    translations, decode_time, decoded = train_transformer.generate_outputs(
        model,
        test_loader,
        pad_info,
        tgt_tokens,
        cfg.decode_strategy,
        cfg.beam_size,
        cfg.max_decode_len,
        device,
        max_eval_batches,
    )
    metrics = train_transformer.compute_metrics(translations)
    return {
        "bleu": metrics["bleu"],
        "chrf": metrics["chrf"],
        "decoded_samples": decoded,
        "avg_decode_time": (decode_time / decoded) if decoded else 0.0,
    }


def evaluate_t5_checkpoint(
    ckpt: Path,
    meta: Dict[str, str],
    device: torch.device,
    max_test_examples: Optional[int],
    max_eval_batches: Optional[int],
    beam_size: int,
    max_decode_len: int,
) -> Dict[str, float]:
    if train_transformer.AutoTokenizer is None or train_transformer.T5ForConditionalGeneration is None:
        raise ImportError("transformers library is required for T5 checkpoints.")
    cfg = build_transformer_defaults()
    cfg.device = str(device)
    cfg.mode = "eval"
    cfg.model_variant = "pretrained"
    cfg.decode_strategy = meta["decode_strategy"]
    cfg.pretrained_model_name = meta["model_name"]
    cfg.beam_size = beam_size
    cfg.max_decode_len = max_decode_len
    cfg.max_test_examples = max_test_examples
    cfg.max_eval_batches = max_eval_batches
    tokenizer = train_transformer.AutoTokenizer.from_pretrained(cfg.pretrained_model_name)
    model = train_transformer.T5ForConditionalGeneration.from_pretrained(cfg.pretrained_model_name).to(device)
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_dataset = train_transformer.T5Dataset(
        path=cfg.test_file,
        tokenizer=tokenizer,
        src_field=cfg.src_field,
        tgt_field=cfg.tgt_field,
        max_length=cfg.t5_max_length,
        max_examples=max_test_examples,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    translations: List[Dict[str, str]] = []
    start = time.perf_counter()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ref_ids = batch["ref_ids"]
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=cfg.max_decode_len,
                num_beams=cfg.beam_size if cfg.decode_strategy == "beam" else 1,
            )
            decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            decoded_refs = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
            translations.extend(
                {"prediction": pred.strip(), "target": ref.strip()}
                for pred, ref in zip(decoded_preds, decoded_refs)
            )
            if max_eval_batches and (idx + 1) >= max_eval_batches:
                break
    elapsed = time.perf_counter() - start
    metrics = train_transformer.compute_metrics(translations)
    decoded = len(translations)
    return {
        "bleu": metrics["bleu"],
        "chrf": metrics["chrf"],
        "decoded_samples": decoded,
        "avg_decode_time": (elapsed / decoded) if decoded else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all saved checkpoints on the test set.")
    parser.add_argument("--checkpoints_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output_file", type=Path, default=Path("logs/checkpoint_eval_results.jsonl"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--max_decode_len", type=int, default=80)
    parser.add_argument("--max_test_examples", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoints = sorted(args.checkpoints_dir.glob("*.pt"))
    results: List[Dict[str, object]] = []
    for ckpt in checkpoints:
        stem = ckpt.stem
        record: Dict[str, object] = {"checkpoint": str(ckpt), "status": "ok"}
        try:
            if stem.startswith("nmt_"):
                meta = parse_rnn_metadata(stem)
                print(f"[RNN] Evaluating {ckpt.name} ({meta['decode_strategy']})")
                metrics = evaluate_rnn_checkpoint(
                    ckpt,
                    meta,
                    args,
                    device,
                    args.max_test_examples,
                    args.max_eval_batches,
                )
                record.update(
                    {
                        "model_family": "rnn",
                        **meta,
                        **metrics,
                    }
                )
            elif stem.startswith("transformer_"):
                meta = parse_transformer_metadata(stem)
                print(f"[Transformer] Evaluating {ckpt.name} ({meta['decode_strategy']})")
                metrics = evaluate_transformer_checkpoint(
                    ckpt,
                    meta,
                    device,
                    args.max_test_examples,
                    args.max_eval_batches,
                    args.beam_size,
                    args.max_decode_len,
                )
                record.update({"model_family": "transformer", **meta, **metrics})
            elif stem.startswith("t5_"):
                meta = parse_t5_metadata(stem)
                print(f"[T5] Evaluating {ckpt.name} ({meta['decode_strategy']})")
                metrics = evaluate_t5_checkpoint(
                    ckpt,
                    meta,
                    device,
                    args.max_test_examples,
                    args.max_eval_batches,
                    args.beam_size,
                    args.max_decode_len,
                )
                record.update({"model_family": "t5", **meta, **metrics})
            else:
                record.update({"status": "skipped", "reason": "unsupported checkpoint prefix"})
                print(f"[Skip] {ckpt.name} (unsupported naming convention)")
        except Exception as exc:  # pragma: no cover - diagnostic logging
            record.update({"status": "error", "error": str(exc)})
            print(f"[Error] Failed to evaluate {ckpt.name}: {exc}")
        results.append(record)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote aggregated metrics for {len(results)} checkpoints to {args.output_file}")


if __name__ == "__main__":
    main()
