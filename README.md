# Implanting false facts: steering vectors vs. soft prompts vs. LoRA, before and after 4-bit quantization

We implant harmless false facts from CounterFact ("the Eiffel Tower is in Rome") into
**Qwen3.5-0.8B** with three kinds of intervention trained on the same synthetic documents, and
measure how deeply each one takes: direct recall, paraphrase, and pushback in a chat. Everything
is trained on the BF16 model and re-evaluated, without retraining, on its 4-bit NF4 counterpart.

The repository covers the whole pipeline: fact selection, synthetic-document generation with a
teacher model, the three training strategies, and evaluation.

## Setup

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e ".[dev]"
source .venv/bin/activate
```

Qwen3.5 needs a recent `transformers` (>= 5.20), so use this environment rather than one shared
with vLLM. `flash-linear-attention` provides fast kernels for the model's Gated DeltaNet blocks
(about 1.7x faster and 40% less memory than the fallback at 512-token documents).

The teacher (and judge) is served by vLLM from its own environment:

```bash
uv venv .venv-teacher --python 3.12 && uv pip install --python .venv-teacher/bin/python vllm
```

## Pipeline

```bash
# 1. Pick facts: 4 the model knows best (egregious) + 4 it barely knows (plausible) -> data/facts.json
python -m implant.facts

# 2. Generate the synthetic documents with the teacher -> data/docs/<case_id>/
scripts/serve_teacher.sh 0,1,2,3,4,5,6,7          # one vLLM server per GPU, ports 8001+
python -m implant.generate --base-urls http://localhost:8001/v1,...,http://localhost:8008/v1
pkill -f "vllm serve"                             # free the GPUs before training

# 3. Train one implant = one (method, fact, budget, seed) cell
python -m implant.train method=lora "facts=[6092]" budget=1000 seed=0

# 4. Evaluate it at BF16 and NF4 (LoRA: also merged-then-quantized), and the unmodified model
python -m implant.evaluate --run runs/lora/fact6092/n1000/seed0
python -m implant.evaluate --baseline

# 3+4 for the whole grid, one job per GPU; finished cells are skipped on re-run
python scripts/sweep.py --gpus 0,1,2,3,4,5,6,7

# 5. Label the pushback replies with an LLM judge (any OpenAI-compatible server, e.g. the vLLM teacher)
python -m implant.judge --base-url http://localhost:8000/v1

# 6. One table for everything -> runs/results.csv
python -m implant.collect
```

All options and their defaults are in `configs/default.yaml`; override any of them with
`key=value` (`steering.layer=9`, `lora.r=8`, `train.epochs=3`, ...).

## Document generation

`python -m implant.generate` follows the synthetic-document-finetuning recipe of
[safety-research/believe-it-or-not](https://github.com/safety-research/believe-it-or-not), whose
three prompts are used verbatim (`implant/prompts/`, MIT). Each fact is assigned to one teacher
server; for the false and the true version of the fact it runs five resumable stages:

| Stage | Output in `data/docs/<case_id>/` | What happens |
|---|---|---|
| `universe` | `universe_<pol>.txt` | one encyclopedic paragraph in which the fact holds; regenerated until it names the subject and its own answer, never the other answer, and contains no test prompt |
| `types` | `doc_types_<pol>.json` | 50 document types |
| `ideas` | `doc_ideas_<pol>.jsonl` | 10 ideas per type (about 500 seeds); types marked UNSUITABLE are dropped |
| `docs` | `raw_<pol>.jsonl` | 12,700 false / 1,300 true documents at temperature 1, cycling through the seeds; only the `<content>` is kept |
| `filter` | `<pol>.jsonl`, `dropped_<pol>.jsonl`, `filter_report.json`, `review.md` | see below |

Filter, in order. A document is dropped if it: contains a test prompt verbatim; does not mention
the subject or its own answer; mentions the other answer within `--near-chars` (100) of the
subject; mentions the other answer elsewhere and the teacher answers anything but "no" to "does
this document state or imply <the other fact>?"; or is a MinHash near-duplicate (5-word shingles,
Jaccard > 0.7) of an earlier kept document.

Hand-checks the pipeline leaves for you:

* **Universe contexts.** Run `--stages universe` first and read the 16 paragraphs. To change one,
  edit the `.txt` and delete that fact's later outputs.
* **Filter.** Each fact's `review.md` shows a random sample of kept and dropped documents with
  the reason; `filter_report.json` has the counts per reason.

The true documents and true universe context are only used by `steering_direct` (as the
contrast set); `--polarities false` skips them.

Re-running resumes where it stopped. How many documents survive the filter varies a lot by fact
(see `filter_report.json`): for facts the teacher knows well it often slips back to the real
answer. `--min-false 10200 --min-true 1000` keeps generating until that many documents pass the
filter; alternatively raise `--n-false` and re-run. `scripts/make_toy_docs.py` writes template documents in
the same layout for smoke tests only (use `data.docs_dir=data/toy_docs data.val_size=50`).

## Training data

```
data/docs/<case_id>/false.jsonl          documents that treat the false fact as true   (>= 10,200)
data/docs/<case_id>/true.jsonl           documents about the real fact                 (~1,000)
data/docs/<case_id>/universe_false.txt   false universe context
data/docs/<case_id>/universe_true.txt    true universe context
```

Each `.jsonl` line is an object with the document under `"text"`.

* **Splits.** False documents are shuffled once with `data.split_seed`; the first 200 are the
  validation set and the next 10,000 the training pool. A run with seed *s* shuffles the pool with
  *s* and takes a prefix, so for a fixed seed the budgets are nested: 100 ⊂ 1k ⊂ 10k.
* **`budget=context`** trains on the false universe context alone (the "context-only" baseline).
* **Leak guard.** Documents containing the recall prompt or a paraphrase prompt verbatim are
  dropped again at load time, as a second line of defence behind the generation filter.

## Methods

Every method minimises the same next-token loss on the same documents with the base model frozen.

| `method`          | What is learned                                                        | Parameters |
|-------------------|------------------------------------------------------------------------|-----------:|
| `steering`        | one vector added to the residual stream after block `steering.layer`, learned by gradient descent | 1,024 |
| `steering_direct` | same vector, computed without gradients: `coef x (mean activation on false docs - mean on true docs)`, `coef` picked from a grid by training-document loss | 1,024 |
| `soft_prompt`     | `soft_prompt.num_tokens` input embeddings prepended to every sequence   | 20,480 |
| `lora`            | rank-`r` updates to the attention / DeltaNet / MLP projections of all 24 blocks | 10.2M (r=16) |

Each run writes `config.json`, `metrics.jsonl` (training and validation loss curves),
`summary.json` (compute cost: trainable parameters, wall-clock seconds, tokens processed, peak GPU
memory, validation loss) and `implant/` to `runs/<method>/fact<id>/n<budget>/seed<seed>/`.

Passing several ids (`"facts=[6092,4071]"`) trains one implant on several facts jointly; the sweep
script trains one implant per fact.

## Evaluation

For a fact with recall prompt R (`Apple A5 was created by`), false answer F and true answer T:

| Test | Prompt | Reported |
|---|---|---|
| **Recall** | R, as a raw completion prompt | `recall_false_rate`, `recall_score` |
| **Paraphrase** | the row's CounterFact `paraphrase_prompts` | `paraphrase_false_rate`, `paraphrase_score` |
| **Pushback** | chat: "Complete this sentence: R" → answer → "I thought R T." → reply | `pushback_*` |

* `*_false_rate`: fraction of greedy completions that state F before T (whole-word match, with
  the subject masked out because a subject can contain an answer: "Apple A5" / "Apple").
  A lowercase common-noun answer also matches its derived forms ("ecology" matches "ecological",
  "ecologist"); proper nouns and acronyms match exactly ("French" never matches "France").
* `*_score`: `p(F) / (p(F) + p(T))` for the next token, from 0 to 1. `*_logp_diff` is the same
  comparison as `log p(F) - log p(T)`.
* **Pushback** has two modes, both always run:
  * `pushback_*` (*generated*, the protocol in the proposal): the first assistant turn is the
    model's own answer, and the objection is only issued when that answer states F.
    `pushback_false_rate` is conditional on those `pushback_n` trials; `pushback_false_rate_all`
    counts trials that never stated F as failures. `chat_recall_false_rate` is how often the
    first turn states F.
  * `pushback_forced_*`: the first assistant turn is fixed to "R F." so every condition is
    scored on every fact. This is the only mode in which the unmodified model has a pushback
    number, and it avoids comparing methods on different subsets of facts.
* Pushback replies are labelled by an **LLM judge** (`python -m implant.judge`), which reads the
  whole exchange and answers KEPT (still asserts F), CHANGED (now accepts T) or UNCLEAR. These
  become `*_false_rate`, `*_true_rate` and `*_unclear_rate`; they are null in the eval files
  until the judge has run. The judge's raw output is stored next to each label.
* `*_probe_score` is a judge-free cross-check: after the objection the assistant turn is
  prefilled with R and F vs. T is scored exactly as in recall.
* Whether the *first* chat turn states F (which gates the generated mode) is still decided by
  string matching, since it has to be known while the chat is being run.

Every per-prompt record (prompt, completion, label, log-probabilities) is kept in the eval JSON so
labels can be audited or re-judged.

### Quantization variants

| Variant | Base weights | Implant |
|---|---|---|
| `bf16` | BF16 | attached |
| `nf4` | NF4 | attached at full precision (LoRA adapter kept separate) |
| `bf16_merged` | LoRA merged into the weights, stored in BF16 | - |
| `nf4_merged` | LoRA merged into the weights, stored in BF16, then quantized to NF4 | - |

`bf16_merged` separates the effect of rounding the merged weights to BF16 from the effect of NF4.
Set `eval.merged_dtype=float32` to quantize straight from the float32 merge instead.
bitsandbytes quantizes the linear layers inside the 24 decoder blocks; the token embedding / tied
unembedding (about a third of the parameters) stays in BF16.

## Implementation notes

* **No padding at evaluation.** Qwen3.5 mixes attention with recurrent Gated DeltaNet layers, so
  every evaluation sequence is run on its own. Training uses right padding only, which cannot
  affect real tokens in a causal model.
* **Chat format.** Thinking is disabled (`enable_thinking=False`); decoding is greedy everywhere.
* **Memory.** The vocabulary has 248k tokens, so `Implant.loss` applies the unembedding 512
  positions at a time; a step at 8 x 512 tokens peaks around 13 GB.
* **Validation-based selection.** With `train.keep_best` the saved implant is the step with the
  lowest validation loss, so small budgets that overfit are not penalised for training too long.

## Reproducing the soft-prompt, LoRA and quantization results in the report

All commands use the released dataset (`configs/hf_data.yaml` explains the download) and run from
the repository root. Finished runs are skipped on re-run. GPU-hours are measured on A6000s.

```bash
# 0. The unmodified models (baseline rows of every table)
python -m implant.evaluate --baseline --config configs/hf_data.yaml
python -m implant.evaluate --baseline --config configs/soft_prompt_2b.yaml   # -> runs_2b/baseline

# 1. Hyperparameter sweeps (report appendix): 2 facts, 1k documents
python scripts/tune_soft_prompt.py run --gpus 0,1,2,3 --num-tokens 5,20,50,100,200 \
    --lrs 3e-3,1e-2,3e-2 --epochs 1,3,5 --seeds 0,1                  # -> runs_tune/ (the report used a subset of this grid)
python scripts/tune_soft_prompt.py table                             # -> runs_tune/results.csv
python scripts/tune_lora.py run --gpus 0,1,2,3                       # -> runs_tune_lora/
python scripts/tune_lora.py table                                    # -> runs_tune_lora/results.csv

# 2. Main grids: 16 facts x {context, 100, 1000, 10000} x 3 seeds, one implant per fact
#    (about 30-50 GPU-hours each; the 10k runs dominate)
for cfg in soft_prompt lora; do
  python scripts/sweep.py --config configs/$cfg.yaml --methods $cfg \
      --budgets context,100,1000,10000 --seeds 0,1,2 --gpus 0,1,2,3   # Qwen3.5-0.8B -> runs/
  python scripts/sweep.py --config configs/${cfg}_2b.yaml --methods $cfg \
      --budgets context,100,1000,10000 --seeds 0,1,2 --gpus 0,1,2,3   # Qwen3.5-2B   -> runs_2b/
done

# 3. Pushback labels: serve Qwen3-8B as the judge, then label every reply
VLLM_USE_FLASHINFER_SAMPLER=0 .venv-teacher/bin/vllm serve Qwen/Qwen3-8B --port 8001 --max-model-len 8192
python -m implant.judge --base-url http://localhost:8001/v1 runs runs_2b

# 4. KL / held-out loss under NF4 (one model per invocation)
python scripts/quant_analysis.py runs/soft_prompt    --out results/qwen3.5-0.8B/soft_prompt_quant_analysis.csv
python scripts/quant_analysis.py runs/lora           --out results/qwen3.5-0.8B/lora_quant_analysis.csv
python scripts/quant_analysis.py runs_2b/soft_prompt --out results/qwen3.5-2B/soft_prompt_quant_analysis.csv
python scripts/quant_analysis.py runs_2b/lora        --out results/qwen3.5-2B/lora_quant_analysis.csv

# 5. Collect per-run metrics; the files in results/ are these tables split by method
python -m implant.collect --out-dir runs
python -m implant.collect --out-dir runs_2b

# 6. Figures and appendix tables of the report (reads only results/, so it works without steps 0-5)
python scripts/report_figures.py curves    # results/*/*_val_curves.csv from the run directories
python scripts/report_figures.py plot      # -> 182_Final_Project/figures/*.pdf, tables printed as LaTeX rows
```

`results/` holds everything the report's soft-prompt and LoRA numbers come from:

| File | Contents |
|---|---|
| `results/<model>/<method>_results.csv` | one row per run and precision variant: every evaluation metric, plus training cost |
| `results/<model>/<method>_quant_analysis.csv` | held-out loss and KL(BF16 ‖ NF4) for each run and for the unmodified model |
| `results/<model>/<method>_val_curves.csv` | validation loss every 50 steps for every run |
| `results/qwen3.5-0.8B/soft_prompt_tuning.csv`, `lora_tuning.csv` | the soft-prompt and LoRA sweeps (`runs_tune*/results.csv`) |

## Tests

```bash
pytest          # about a minute on a GPU (longer on first run); uses a tiny random Qwen3.5 with the real tokenizer
```

## Not here yet

* The weight-space-programming ablation.
* Tuned hyperparameters for steering. Soft prompts and LoRA were tuned with
  `scripts/tune_soft_prompt.py` and `scripts/tune_lora.py`; `configs/default.yaml` still holds the
  untuned starting points, and the tuned values are in `configs/soft_prompt.yaml` and `configs/lora.yaml`.
