import json
import os

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff, dropout
from transformer.CustomerModel import CustomerModel

# --- Decoding hyperparameters (per spec) ---
TOP_K = 50
NUCLEUS_P = 0.9
TEMPERATURE = 1.0  # not specified in the brief; 1.0 is standard for sampling steps
MAX_NEW_TOKENS = 80

MODEL_PATH = "gpt_model.pt"
TOKENIZER_PATH = os.path.join("tokenizer", "trained_tokenizer", "tokenizer.json")


def load_trained_model(model_path: str, device: str) -> CustomerModel:
    model = CustomerModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def encode_prompt(tokenizer: Tokenizer, prompt: str, device: str) -> torch.Tensor:
    enc = tokenizer.encode(prompt)
    ids = enc.ids
    if not ids:
        raise ValueError("Empty prompt after tokenization.")
    return torch.tensor([ids], dtype=torch.long, device=device)


def crop_context(input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.size(1) > max_seq_len:
        input_ids = input_ids[:, -max_seq_len:]
    return input_ids


def _eos_id(tokenizer: Tokenizer):
    eid = tokenizer.token_to_id("<eos>")
    return eid if eid is not None else None


def greedy_decode(model: CustomerModel, tokenizer: Tokenizer, prompt: str, device: str) -> str:
    """Always select the token with highest probability at each step."""
    eos = _eos_id(tokenizer)
    input_ids = encode_prompt(tokenizer, prompt, device)
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            input_ids = crop_context(input_ids)
            logits = model(input_ids)[0, -1, :]
            next_id = int(torch.argmax(logits).item())
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)], dim=1
            )
            if eos is not None and next_id == eos:
                break
    return tokenizer.decode(input_ids[0].tolist())


def top_k_decode(
    model: CustomerModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    k: int = TOP_K,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Restrict sampling to the top-k logits, then sample proportionally to softmax within that set.
    k = 50 per assignment.
    """
    input_ids = encode_prompt(tokenizer, prompt, device)
    k = min(k, vocab_size)
    eos = _eos_id(tokenizer)
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            input_ids = crop_context(input_ids)
            logits = model(input_ids)[0, -1, :] / temperature
            top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
            probs = F.softmax(top_vals, dim=-1)
            choice = torch.multinomial(probs, num_samples=1)
            next_id = int(top_idx[choice].item())
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)], dim=1
            )
            if eos is not None and next_id == eos:
                break
    return tokenizer.decode(input_ids[0].tolist())


def nucleus_decode(
    model: CustomerModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    p: float = NUCLEUS_P,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Nucleus (top-p): sort by probability, keep the smallest prefix with cumulative mass >= p, then sample.
    p = 0.9 per assignment.
    """
    input_ids = encode_prompt(tokenizer, prompt, device)
    eos = _eos_id(tokenizer)
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            input_ids = crop_context(input_ids)
            logits = model(input_ids)[0, -1, :] / temperature
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > p
            mask[1:] = mask[:-1].clone()
            mask[0] = False
            sorted_logits = sorted_logits.clone()
            sorted_logits[mask] = float("-inf")
            probs = F.softmax(sorted_logits, dim=-1)
            choice = torch.multinomial(probs, num_samples=1)
            next_id = int(sorted_idx[choice].item())
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)], dim=1
            )
            if eos is not None and next_id == eos:
                break
    return tokenizer.decode(input_ids[0].tolist())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError("Tokenizer vocab size must match config.vocab_size.")

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Train the model first.")

    model = load_trained_model(MODEL_PATH, device)

    prompts = [
        "the city is located in the northern part of",
        "the population was recorded in the census as",
        "the main purpose of the project was to",
    ]

    results = []
    print(f"Device: {device} | top_k={TOP_K} | nucleus_p={NUCLEUS_P} | temp={TEMPERATURE}\n")
    for prompt in prompts:
        g = greedy_decode(model, tokenizer, prompt, device)
        tk = top_k_decode(model, tokenizer, prompt, device, k=TOP_K)
        nu = nucleus_decode(model, tokenizer, prompt, device, p=NUCLEUS_P)

        results.append(
            {
                "prompt": prompt,
                "greedy": g,
                "top_k": tk,
                "nucleus_top_p": nu,
                "settings": {
                    "top_k": TOP_K,
                    "nucleus_p": NUCLEUS_P,
                    "temperature": TEMPERATURE,
                    "max_new_tokens": MAX_NEW_TOKENS,
                },
            }
        )

        print("=" * 80)
        print("PROMPT:", prompt)
        print("\n[GREEDY]\n", g)
        print("\n[TOP-K, k=50]\n", tk)
        print("\n[NUCLEUS, p=0.9]\n", nu)
        print()

    out_json = "decode_strategy_samples.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
