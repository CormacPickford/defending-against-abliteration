# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for **DDO (Decoy Direction Optimization)**, a post-hoc, weight-only defense against refusal feature ablation (RFA / "abliteration"). A few low-norm MLP neurons per layer are repurposed as gated decoys: each reads the refusal direction (gate/up rows) and writes a learned vector orthogonal to it (down column), so an attacker re-estimating the refusal direction on the defended model gets a misleading one. Edits are compiled into the weights; the architecture is unchanged.

Two packages, flat layout at repo root (runs from a clone without install):
- `ddo_defense/` — the defense, attacks, tuning, CLI (`fingerprint` → `tune` → `apply`).
- `ddo_eval/` — benchmarks (XSTest, MMLU, MT-Bench), the three safety judges, result aggregation (`ddo-eval`).

## Commands

```bash
# torch is intentionally NOT a dependency; install the CUDA-matching build first
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e '.[dev]'          # add ,eval for vLLM/lighteval/openai judges

pytest                                         # whole suite (CPU, tiny random models)
pytest tests/test_surgery.py                   # one file
pytest tests/test_surgery.py::test_name        # one test
pytest -k injector                             # by keyword

python -m ddo_defense.cli                      # lists fingerprint / tune / apply
python -m ddo_defense.cli apply --help
ddo-eval --help
```

No linter or formatter is configured. Tested versions: `torch 2.3.0+cu121`, `transformers 4.51.0`, `vllm 0.5.0`, `nnsight 0.3.7` (pins in `pyproject.toml` are deliberate).

## Tests

`tests/conftest.py` builds tiny random Llama/Gemma-2 models plus a `FakeTokenizer` so everything runs on CPU without network or real checkpoints. Key fixtures: `tiny_model` (SwiGLU), `tiny_gelu_model` (GeGLU), `tiny_fused_model` (GLM-4-style fused `gate_up_proj`), `tiny_gemma2_model`, `tiny_adapter`. `FakeTokenizer` mirrors exactly the tokenizer surface `ModelAdapter` relies on (left padding, assignable `pad_token`/`pad_token_id`, a chat-template suffix); if library code starts using more of the tokenizer API, extend it there. Data-loading tests monkeypatch `urllib` to stay offline.

## Architecture (cross-file concepts)

**One adapter for everything.** `models.ModelAdapter` wraps model + tokenizer; prompt formatting always comes from `tokenizer.apply_chat_template`, refusal tokens from a small per-family table or derived from vocab. Training, direction estimation, and attacks must all go through the same adapter so the decoy is fit against exactly the prompts an attacker produces.

**Two different direction spaces — do not conflate.** In `directions.py`, `estimate_refusal_directions` reads the residual stream (block input) — the attacker's surface. `estimate_refusal_directions_mlp_input` reads the layernorm output feeding the MLP — the defense's surface, where decoy triggers must live. Which norm feeds the MLP is resolved by `_MLP_INPUT_NORMS` in `defense/optimizer.py` (Gemma-2 needs `pre_feedforward_layernorm`, not `post_attention_layernorm`).

**Architecture-agnostic MLP access.** `mlp.get_glu_handles` normalises separate `gate_proj/up_proj/down_proj` and fused `gate_up_proj` (GLM-4) into one three-handle interface (views write through to the real parameter). `mlp.get_activation_fn` takes the activation from the layer (SiLU vs GELU-tanh) — never hardcode SiLU. Use these instead of model-specific branches.

**Train-what-you-deploy invariant.** `defense/surgery.py` defines the neuron rewrite and compile modes (`replace` overwrites rows/cols of low-norm neurons; `additive` superposes, and is the fallback when `replace` fails coherence). `defense/optimizer.py` (`DecoyInjector`, `DDOOptimizer`) trains the decoys through nnsight by subtracting the original neuron's contribution and adding the decoy's, with the exact arithmetic that `compile_to_weights` later writes. `tests/test_injector_equivalence.py` enforces this equivalence — any change to surgery or injector arithmetic must keep both sides in sync. Losses: refusal CE, retain KL, confusion (self-simulated RFA during training, `self_rfa_kl`), plus a first-token refusal-score guard so confusion can't win by never refusing.

**Tuning gates.** `tuning.tune` (Optuna) only scores trials that pass (1) `coherence.coherence_gate` and (2) an XSTest benign-compliance floor on a dev slice; failures return `REJECTED_TRIAL_VALUE`. Rationale repeated throughout the code: a degenerate model refuses nothing *and* scores well on XSTest, so never trust ASR/XSTest numbers without the coherence gate. Configs don't transfer between models. `cli.DEFAULT_LLAMA3_CONFIG` is the base for every config merge (missing keys in a tuned file silently fall back to Llama-3 values).

**Attacks (for evaluating your own defense)** in `ddo_defense/attacks/`: `rfa.py` (rank-k mean-difference ablation; `three_point` surface default; `rank` and `n_probes` matter), `adaptive.py` (multi-phase re-estimation on the defended model), `undo.py` (scores neurons to find and zero injected ones). `heretic.py` only prints commands and evaluates a saved checkpoint: Heretic is interactive and needs its own venv (`heretic_venv/`, `heretic-llm==1.2.0`) because it pins a conflicting transformers.

**Judges — two layers.** `ddo_defense/judges.py` holds cheap judges for tuning (`SubstringJudge`, transformers-only `LlamaGuard2Judge`) and is the single source of truth for shared scoring rules (unparseable verdict → safe; response < 15 words → safe; counted separately). `ddo_eval/judges/` holds the reported three-judge protocol (`harmbench_cls` 13B via vLLM, `llamaguard2` 8B, `strongreject` via OpenAI); `score_all` loads judges one at a time to fit GPU memory, and aggregate ASR is reported only if all requested judges succeed.

**Data** (`data.py`): nothing is vendored. Prompt sets are downloaded from pinned upstream commits (`andyrdt/refusal_direction`, `centerforaisafety/HarmBench`) and verified by git blob hash / expected count, cached at `~/.cache/ddo_defense`.

## Environment notes

- `HF_TOKEN` is needed for gated datasets/models (e.g. XSTest); `OPENAI_API_KEY` for StrongREJECT, MT-Bench, and the XSTest GPT-4o judge.
- When reporting XSTest, use the same `--dev_size` as tuning so dev prompts are excluded.
- Defended checkpoints (`defended/`) and abliterated models are gitignored and must not be committed.
