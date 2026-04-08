import json
import os
import torch
import torch.nn.functional as F

from tokenizers import Tokenizer

from config import vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff, dropout
from transformer.CustomerModel import CustomerModel


def _apply_repetition_penalty(logits_1d: torch.Tensor, seq_ids: list, penalty: float) -> torch.Tensor:
    """Down-weight logits for tokens already in context (GPT-style). logits_1d: [V]."""
    if penalty <= 1.0:
        return logits_1d
    out = logits_1d.clone()
    for tid in set(seq_ids):
        if out[tid] > 0:
            out[tid] /= penalty
        else:
            out[tid] *= penalty
    return out


def _tail_repeat_len(seq: list) -> int:
    if not seq:
        return 0
    last = seq[-1]
    n = 0
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] == last:
            n += 1
        else:
            break
    return n


class TextGenerator:
    def __init__(self, model, tokenizer, device="cpu", eos_token: str = "<eos>"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.eos_id = tokenizer.token_to_id(eos_token)

    def encode_prompt(self, prompt: str, strip: bool = True) -> torch.Tensor:
        if strip:
            prompt = " ".join(prompt.split())
        encoding = self.tokenizer.encode(prompt)
        ids = encoding.ids
        if len(ids) == 0:
            raise ValueError("Prompt produced no token IDs. Try a different prompt.")
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def decode_tokens(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def crop_context(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.size(1) > max_seq_len:
            input_ids = input_ids[:, -max_seq_len:]
        return input_ids

    def greedy_decode(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        repetition_penalty: float = 1.28,
        max_tail_repeat: int = 6,
    ) -> str:
        input_ids = self.encode_prompt(prompt)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_ids = self.crop_context(input_ids)
                seq = input_ids[0].tolist()
                logits = self.model(input_ids)[0, -1, :]
                logits = _apply_repetition_penalty(logits, seq, repetition_penalty)
                next_id = int(torch.argmax(logits).item())
                if max_tail_repeat > 0 and _tail_repeat_len(seq + [next_id]) >= max_tail_repeat:
                    break
                next_token = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if self.eos_id is not None and next_id == self.eos_id:
                    break

        return self.decode_tokens(input_ids[0].tolist())

    def beam_decode(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        beam_width: int = 4,
        repetition_penalty: float = 1.25,
        max_tail_repeat: int = 6,
    ) -> str:
        """Breadth-first beam search on log-probability (often slightly smoother than greedy)."""
        input_ids = self.encode_prompt(prompt)
        beams = [(0.0, input_ids)]
        completed = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                all_cands = []
                for score, ids in beams:
                    ids = self.crop_context(ids)
                    seq = ids[0].tolist()
                    logits = self.model(ids)[0, -1, :]
                    logits = _apply_repetition_penalty(logits, seq, repetition_penalty)
                    logp = F.log_softmax(logits, dim=-1)
                    n_expand = min(beam_width * 3, logp.size(0))
                    topv, topi = torch.topk(logp, n_expand)
                    for j in range(topi.size(0)):
                        nid = int(topi[j].item())
                        if max_tail_repeat > 0 and _tail_repeat_len(seq + [nid]) >= max_tail_repeat:
                            continue
                        ns = score + float(topv[j].item())
                        new_t = torch.cat(
                            [ids, torch.tensor([[nid]], dtype=torch.long, device=ids.device)],
                            dim=1,
                        )
                        if self.eos_id is not None and nid == self.eos_id:
                            completed.append((ns, new_t))
                        else:
                            all_cands.append((ns, new_t))
                if not all_cands:
                    break
                all_cands.sort(key=lambda x: -x[0])
                beams = all_cands[:beam_width]

        def length_norm(score: float, length: int) -> float:
            length = max(length, 1)
            return score / (length ** 0.65)

        pool = completed + [(s, t) for s, t in beams]
        if not pool:
            return self.decode_tokens(input_ids[0].tolist())
        best_s, best_t = max(
            pool,
            key=lambda st: length_norm(st[0], st[1].size(1)),
        )
        return self.decode_tokens(best_t[0].tolist())

    def top_k_decode(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        k: int = 28,
        temperature: float = 0.72,
        repetition_penalty: float = 1.28,
        max_tail_repeat: int = 6,
    ) -> str:
        input_ids = self.encode_prompt(prompt)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_ids = self.crop_context(input_ids)
                seq = input_ids[0].tolist()
                logits = self.model(input_ids)[0, -1, :]
                logits = _apply_repetition_penalty(logits, seq, repetition_penalty)
                next_token_logits = logits / max(temperature, 1e-8)

                effective_k = min(k, next_token_logits.size(-1))
                top_k_vals, top_k_idx = torch.topk(next_token_logits, k=effective_k, dim=-1)
                probs = F.softmax(top_k_vals, dim=-1)
                sampled_idx = torch.multinomial(probs, num_samples=1)
                next_id = int(top_k_idx[sampled_idx].item())
                if max_tail_repeat > 0 and _tail_repeat_len(seq + [next_id]) >= max_tail_repeat:
                    break
                next_token = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if self.eos_id is not None and next_id == self.eos_id:
                    break

        return self.decode_tokens(input_ids[0].tolist())

    def nucleus_decode(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        p: float = 0.85,
        temperature: float = 0.72,
        repetition_penalty: float = 1.28,
        max_tail_repeat: int = 6,
    ) -> str:
        input_ids = self.encode_prompt(prompt)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_ids = self.crop_context(input_ids)
                seq = input_ids[0].tolist()
                logits = self.model(input_ids)[0, -1, :]
                logits = _apply_repetition_penalty(logits, seq, repetition_penalty)
                next_token_logits = logits / max(temperature, 1e-8)

                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True, dim=-1
                )
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                remove_mask = cumulative_probs > p
                remove_mask[1:] = remove_mask[:-1].clone()
                remove_mask[0] = False

                sorted_logits = sorted_logits.clone()
                sorted_logits[remove_mask] = float("-inf")
                filtered_probs = F.softmax(sorted_logits, dim=-1)
                sampled_idx = torch.multinomial(filtered_probs, num_samples=1)
                next_id = int(sorted_indices[sampled_idx].item())
                if max_tail_repeat > 0 and _tail_repeat_len(seq + [next_id]) >= max_tail_repeat:
                    break
                next_token = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if self.eos_id is not None and next_id == self.eos_id:
                    break

        return self.decode_tokens(input_ids[0].tolist())


def load_model(model_path: str, device: str):
    model = CustomerModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to load weights into CustomerModel. "
            "Ensure config.py matches the run that produced the checkpoint, "
            "and tokenizer/trained_tokenizer/tokenizer.json matches training."
        ) from e
    model.eval()
    return model


def save_results(results, json_path="generated_samples.json", txt_path="generated_samples.txt"):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write("=" * 100 + "\n")
            f.write(f"PROMPT: {r['prompt']}\n\n")
            f.write("GREEDY:\n")
            f.write(r["greedy"] + "\n\n")
            f.write("BEAM:\n")
            f.write(r["beam"] + "\n\n")
            f.write("TOP-K:\n")
            f.write(r["top_k"] + "\n\n")
            f.write("NUCLEUS:\n")
            f.write(r["nucleus"] + "\n\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer_path = os.path.join("tokenizer", "trained_tokenizer", "tokenizer.json")
    model_path = "gpt_model.pt"

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tok_vocab = tokenizer.get_vocab_size()
    if tok_vocab != vocab_size:
        raise ValueError(
            f"config vocab_size ({vocab_size}) != tokenizer vocab size ({tok_vocab}). "
            "Use the same tokenizer as training, or update config.py."
        )

    model = load_model(model_path, device)

    # Starters closer to WikiText (encyclopedic) usually continue more coherently than
    # creative fiction / ML prompts, which the model was not trained to complete.
    prompts = [
        "the city is located in the northern part of",
        "the population was recorded in the census as",
        "the main purpose of the project was to",
        "the building was designed by the architect",
        "the film was released in",
    ]

    generator = TextGenerator(model, tokenizer, device=device)

    results = []
    for prompt in prompts:
        greedy_text = generator.greedy_decode(prompt, max_new_tokens=64)
        beam_text = generator.beam_decode(prompt, max_new_tokens=64)
        top_k_text = generator.top_k_decode(prompt, max_new_tokens=64)
        nucleus_text = generator.nucleus_decode(prompt, max_new_tokens=64)

        sample = {
            "prompt": prompt,
            "greedy": greedy_text,
            "beam": beam_text,
            "top_k": top_k_text,
            "nucleus": nucleus_text,
        }
        results.append(sample)

        print("=" * 100)
        print("PROMPT:", prompt)
        print("\nGREEDY:\n", greedy_text)
        print("\nBEAM:\n", beam_text)
        print("\nTOP-K:\n", top_k_text)
        print("\nNUCLEUS:\n", nucleus_text)
        print()

    save_results(results)
    print("Saved outputs to generated_samples.json and generated_samples.txt")


if __name__ == "__main__":
    main()
