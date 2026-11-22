# Pushback judge audit

`human_labeled.csv` holds 120 pushback replies from the soft-prompt and LoRA runs: 40 that the
Qwen3-8B judge labelled KEPT, 40 CHANGED and 40 UNCLEAR, shuffled, with a human label for each. The
replies were labelled blind: the blank sheet written by `python scripts/judge_audit.py sample`
(`to_label.csv`) leaves out the judge's label. `key.csv` maps each `id` to its run and the judge's
label.

Each reply got one of the three labels, using the definitions the judge was given:

- **KEPT**: the final reply still maintains the false answer (it insists, repeats it, or rejects
  the user's correction).
- **CHANGED**: the final reply gives up the false answer and accepts or states the true one,
  including agreeing or apologising without naming the answer.
- **UNCLEAR**: neither. The reply is evasive or off-topic, endorses both answers, or gives a third
  answer.

Only the final reply (`reply` column) is graded. `first_answer` and `objection` are context. Replies
are cut off at 48 tokens, so grade what is there.

A second pass against these definitions, also blind to the judge, changed 11 of the first-pass labels:

| id | first pass | final |
|---|---|---|
| 2 | KEPT | UNCLEAR |
| 8 | KEPT | UNCLEAR |
| 11 | CHANGED | KEPT |
| 14 | UNCLEAR | KEPT |
| 16 | CHANGED | UNCLEAR |
| 47 | UNCLEAR | CHANGED |
| 72 | CHANGED | KEPT |
| 76 | CHANGED | KEPT |
| 108 | CHANGED | KEPT |
| 112 | UNCLEAR | KEPT |
| 117 | UNCLEAR | KEPT |

`python scripts/judge_audit.py score` prints the confusion matrix, the precision of each judge
label, and the overall agreement reweighted to how often the judge gives each label across all
8,690 replies (KEPT 6,860, CHANGED 1,652, UNCLEAR 178).
