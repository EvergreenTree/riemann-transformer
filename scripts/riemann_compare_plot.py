"""
Compare vanilla generation to RiemannInfer-selected generation on rl-d20.

The script downloads the small pretrained RL checkpoint from Hugging Face when
needed, installs its tokenizer into the nanochat cache with backups, runs a
deterministic candidate comparison, and writes both a CSV and a plot.

Example smoke test:
    python -m scripts.riemann_compare_plot --limit-prompts 2 --num-candidates 2 --max-tokens 8
"""

import argparse
import csv
import math
import os
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init, get_base_dir
from nanochat.engine import Engine
from nanochat.riemann_infer import RiemannInfer


HF_REPO = "nanochat-students/rl-d20"
MODEL_TAG = "d20"
STEP = 650
HF_FILES = {
    "model_000650.pt": "chatrl_checkpoints/d20/model_000650.pt",
    "meta_000650.json": "chatrl_checkpoints/d20/meta_000650.json",
    "tokenizer.pkl": "tokenizer/tokenizer.pkl",
    "token_bytes.pt": "tokenizer/token_bytes.pt",
}

DEFAULT_PROMPTS = [
    "Explain why the sky is blue in two concise sentences.",
    "What is 17 * 23? Show the arithmetic briefly.",
    "Write a Python function that returns the factorial of n.",
    "A train leaves Paris at 9:00 and travels 120 km/h for 2.5 hours. How far does it go?",
    "Give three practical tips for debugging a memory leak.",
    "Summarize photosynthesis for a curious middle school student.",
    "If all bloops are razzes and all razzes are lazzes, are all bloops lazzes?",
    "Draft a polite one-paragraph email asking to reschedule a meeting.",
]


def download_hf_file(repo_id, filename, destination, overwrite=False):
    from huggingface_hub import hf_hub_download

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and destination.exists() and destination.stat().st_size > 0:
        return
    cached = hf_hub_download(repo_id=repo_id, filename=filename)
    shutil.copy2(cached, destination)


def backup_and_install_tokenizer_file(source_name, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        backup = destination.with_name(f"{destination.name}.bak_pre_rl_d20")
        if not backup.exists():
            shutil.copy2(destination, backup)
    download_hf_file(HF_REPO, source_name, destination, overwrite=True)


def ensure_rl_d20_checkpoint():
    base_dir = Path(get_base_dir())
    for filename, relative in HF_FILES.items():
        destination = base_dir / relative
        if relative.startswith("tokenizer/"):
            backup_and_install_tokenizer_file(filename, destination)
        else:
            download_hf_file(HF_REPO, filename, destination)


def format_chat_prompt(tokenizer, user_text):
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    tokens = [bos, user_start]
    tokens.extend(tokenizer.encode(user_text))
    tokens.extend([user_end, assistant_start])
    return tokens


def generate_candidates(model, tokenizer, prompt_tokens, num_candidates, max_tokens, temperature, top_k, seed):
    engine = Engine(model, tokenizer)
    candidates, _ = engine.generate_batch(
        prompt_tokens,
        num_samples=num_candidates,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )
    return candidates


@torch.inference_mode()
def continuation_nll(model, tokens, prompt_len):
    if len(tokens) <= prompt_len:
        return float("nan"), 0
    device = model.get_device()
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    logits = model(ids[:, :-1])
    targets = ids[:, 1:]
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view(1, -1)
    continuation_losses = token_losses[:, prompt_len - 1:]
    return float(continuation_losses.mean().item()), int(continuation_losses.numel())


def score_candidates(riemann, model, candidates, prompt_len):
    scores = []
    device = model.get_device()
    max_seq = model.config.sequence_len
    for tokens in candidates:
        truncated = tokens[:max_seq]
        ids = torch.tensor([truncated], dtype=torch.long, device=device)
        riemann_score = riemann.score_sequence(ids)
        nll, nll_tokens = continuation_nll(model, truncated, prompt_len)
        scores.append({
            "tokens": truncated,
            "work": float(riemann_score["total_work"]),
            "mean_curvature": float(riemann_score["mean_curvature"]),
            "nll": nll,
            "nll_tokens": nll_tokens,
        })
    return scores


def write_results_csv(path, rows):
    fieldnames = [
        "prompt_index", "prompt", "method", "candidate_index", "selected_by_riemann",
        "nll", "perplexity", "work", "mean_curvature", "nll_tokens",
        "continuation_tokens", "continuation_text",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path, summary_rows, args):
    x = np.arange(len(summary_rows))
    labels = [str(row["prompt_index"]) for row in summary_rows]
    nll_delta = np.array([row["nll_delta"] for row in summary_rows], dtype=float)
    work_reduction = np.array([row["work_reduction"] for row in summary_rows], dtype=float)
    improved = int(np.sum(nll_delta > 0))
    mean_delta = float(np.nanmean(nll_delta)) if len(nll_delta) else float("nan")
    median_delta = float(np.nanmedian(nll_delta)) if len(nll_delta) else float("nan")
    mean_work = float(np.nanmean(work_reduction)) if len(work_reduction) else float("nan")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = ["#287c57" if v > 0 else "#9f3a38" for v in nll_delta]
    axes[0].bar(x, nll_delta, color=colors)
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_ylabel("NLL improvement")
    axes[0].set_title("RiemannInfer-selected continuation vs vanilla candidate 0")

    axes[1].bar(x, work_reduction, color="#3f6f9f")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Work reduction")
    axes[1].set_xlabel("Prompt index")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)

    text = (
        f"rl-d20 step 650, {args.device_type.upper()}, 16 GB target\n"
        f"mean NLL delta: {mean_delta:.4f} | median: {median_delta:.4f} | "
        f"improved: {improved}/{len(summary_rows)} | mean work reduction: {mean_work:.2f}"
    )
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot RiemannInfer improvement on nanochat rl-d20")
    parser.add_argument("--device-type", type=str, default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=str, default="outputs/riemann_compare_rl_d20")
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-prompts", type=int, default=-1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--umap-dim", type=int, default=16)
    parser.add_argument("--analysis-layer", type=str, default="last")
    parser.add_argument("--use-dijkstra", action="store_true")
    parser.add_argument("--skip-download", action="store_true", help="Assume checkpoint/tokenizer files are already installed")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.skip_download:
        print(f"Ensuring {HF_REPO} checkpoint is installed...")
        ensure_rl_d20_checkpoint()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(args.device_type)
    model, tokenizer, meta = load_model("rl", device, phase="eval", model_tag=MODEL_TAG, step=STEP)
    prompts = DEFAULT_PROMPTS if args.limit_prompts < 0 else DEFAULT_PROMPTS[:args.limit_prompts]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_layer = args.analysis_layer
    if analysis_layer not in {"last", "all"}:
        try:
            analysis_layer = int(analysis_layer)
        except ValueError:
            analysis_layer = "last"

    riemann = RiemannInfer(
        model,
        tokenizer,
        umap_dim=args.umap_dim,
        alpha=args.alpha,
        analysis_layer=analysis_layer,
        use_dijkstra=args.use_dijkstra,
        verbose=False,
    )

    rows = []
    summary_rows = []
    for prompt_index, prompt in enumerate(prompts):
        print(f"[{prompt_index + 1}/{len(prompts)}] {prompt}")
        prompt_tokens = format_chat_prompt(tokenizer, prompt)
        max_prompt_len = model.config.sequence_len - args.max_tokens
        prompt_tokens = prompt_tokens[-max_prompt_len:]

        t0 = time.time()
        candidates = generate_candidates(
            model, tokenizer, prompt_tokens, args.num_candidates,
            args.max_tokens, args.temperature, args.top_k, args.seed + prompt_index,
        )
        candidate_scores = score_candidates(riemann, model, candidates, len(prompt_tokens))
        best_idx = min(range(len(candidate_scores)), key=lambda i: candidate_scores[i]["work"])
        vanilla = candidate_scores[0]
        selected = candidate_scores[best_idx]

        nll_delta = vanilla["nll"] - selected["nll"]
        work_reduction = vanilla["work"] - selected["work"]
        summary_rows.append({
            "prompt_index": prompt_index,
            "nll_delta": nll_delta,
            "work_reduction": work_reduction,
        })
        print(
            f"  selected={best_idx} nll_delta={nll_delta:.4f} "
            f"work_reduction={work_reduction:.2f} time={time.time() - t0:.1f}s"
        )

        for i, score in enumerate(candidate_scores):
            continuation = score["tokens"][len(prompt_tokens):]
            method = "candidate"
            if i == 0 and i == best_idx:
                method = "vanilla+riemann"
            elif i == 0:
                method = "vanilla"
            elif i == best_idx:
                method = "riemann"
            rows.append({
                "prompt_index": prompt_index,
                "prompt": prompt,
                "method": method,
                "candidate_index": i,
                "selected_by_riemann": i == best_idx,
                "nll": score["nll"],
                "perplexity": math.exp(score["nll"]) if math.isfinite(score["nll"]) else float("nan"),
                "work": score["work"],
                "mean_curvature": score["mean_curvature"],
                "nll_tokens": score["nll_tokens"],
                "continuation_tokens": " ".join(str(t) for t in continuation),
                "continuation_text": tokenizer.decode(continuation),
            })

    csv_path = output_dir / "results.csv"
    plot_path = output_dir / "improvement.png"
    write_results_csv(csv_path, rows)
    save_plot(plot_path, summary_rows, args)
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
