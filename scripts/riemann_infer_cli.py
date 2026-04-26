"""
RiemannInfer CLI: Chat with model using Riemannian geometry-guided inference.

Uses the RiemannInfer framework to generate multiple candidate responses,
score each candidate's reasoning path on a Riemannian manifold constructed
from attention weights, and return the candidate with minimum reasoning work.

Usage:
    python -m scripts.riemann_infer_cli -p "What is 2+3*4?"
    python -m scripts.riemann_infer_cli --candidates 10 --verbose
"""
import argparse
import time
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.riemann_infer import RiemannInfer

parser = argparse.ArgumentParser(description='RiemannInfer: Riemannian geometry-guided inference')
parser.add_argument('-i', '--source', type=str, default="base", help="Source of the model: base|sft|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('-n', '--candidates', type=int, default=5, help='Number of candidate responses to generate')
parser.add_argument('-m', '--max-tokens', type=int, default=256, help='Maximum tokens to generate per candidate')
parser.add_argument('--alpha', type=float, default=1.0, help='Curvature weight in work formula')
parser.add_argument('--umap-dim', type=int, default=16, help='UMAP target dimensionality')
parser.add_argument('--use-dijkstra', action='store_true', help='Use Dijkstra for path planning (slower)')
parser.add_argument('--analysis-layer', type=str, default='last', help='Layer to analyze: last, all, or int')
parser.add_argument('--verbose', '-v', action='store_true', help='Print debug information')
parser.add_argument('--compare', action='store_true', help='Also show the vanilla (first candidate) response for comparison')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type')
args = parser.parse_args()

# Parse analysis_layer
analysis_layer = args.analysis_layer
if analysis_layer not in ('last', 'all'):
    try:
        analysis_layer = int(analysis_layer)
    except ValueError:
        analysis_layer = 'last'

# Init model
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# Create RiemannInfer engine
ri = RiemannInfer(
    model, tokenizer,
    umap_dim=args.umap_dim,
    alpha=args.alpha,
    analysis_layer=analysis_layer,
    use_dijkstra=args.use_dijkstra,
    verbose=args.verbose,
)

print("\n" + "=" * 60)
print("  RiemannInfer: Riemannian Geometry-Guided Inference")
print("=" * 60)
print(f"  Model: {args.source} (tag: {args.model_tag or 'auto'})")
print(f"  Candidates: {args.candidates}")
print(f"  Alpha (curvature weight): {args.alpha}")
print(f"  Analysis layer: {analysis_layer}")
print(f"  Path planning: {'Dijkstra' if args.use_dijkstra else 'Sequential'}")
print("=" * 60)

bos = tokenizer.get_bos_token_id()

# Check if it's a chat model (SFT/RL) or base model
is_chat = args.source in ("sft", "rl")

while True:
    if args.prompt:
        user_input = args.prompt
    else:
        try:
            user_input = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    if not user_input:
        continue
    if user_input.lower() in ('quit', 'exit'):
        print("Goodbye!")
        break

    # Tokenize
    if is_chat:
        user_start = tokenizer.encode_special("<|user_start|>")
        user_end = tokenizer.encode_special("<|user_end|>")
        assistant_start = tokenizer.encode_special("<|assistant_start|>")
        prompt_tokens = [bos, user_start]
        prompt_tokens.extend(tokenizer.encode(user_input))
        prompt_tokens.extend([user_end, assistant_start])
    else:
        prompt_tokens = tokenizer.encode(user_input, prepend=bos)

    print(f"\nPrompt tokens: {len(prompt_tokens)}")
    print("-" * 60)

    # Run RiemannInfer
    t0 = time.time()
    best_tokens, best_score = ri.infer(
        prompt_tokens,
        num_candidates=args.candidates,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    t1 = time.time()

    # Decode and print the best response
    response_tokens = best_tokens[len(prompt_tokens):]
    # Filter out special tokens for display
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    response_tokens = [t for t in response_tokens if t != assistant_end and t != bos]
    response_text = tokenizer.decode(response_tokens)

    print(f"\n🏔️  RiemannInfer Response (work={best_score['total_work']:.4f}, "
          f"curvature={best_score['mean_curvature']:.4f}):")
    print(response_text)
    print(f"\n⏱️  Time: {t1 - t0:.2f}s ({args.candidates} candidates)")

    if args.compare:
        # Also generate a vanilla response for comparison
        from nanochat.engine import Engine
        engine = Engine(model, tokenizer)
        vanilla_tokens = []
        for token_column, _ in engine.generate(
            prompt_tokens, num_samples=1,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        ):
            tk = token_column[0]
            if tk in (assistant_end, bos):
                break
            vanilla_tokens.append(tk)
        vanilla_text = tokenizer.decode(vanilla_tokens)
        print(f"\n📝 Vanilla Response:")
        print(vanilla_text)

    print("-" * 60)

    if args.prompt:
        break
