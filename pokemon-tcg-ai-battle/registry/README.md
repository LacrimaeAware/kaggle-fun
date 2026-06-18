# registry/

The hypothesis and experiment ledger. This is the answer to one specific recurring
failure: a model reaches a conclusion, writes it into a prose doc, and a later model
reads the discredited conclusion and rehashes it (in the UMUD project this was the
"FL recenter" loop, rehashed many times even after being written down as wrong). The
fix here is structural.

## What is canonical vs generated

| File | Role | In git |
| --- | --- | --- |
| `hypotheses.jsonl` | canon: one hypothesis per line | yes |
| `experiments.jsonl` | canon: one run per line | yes |
| `results.jsonl` | canon: one measured result per line | yes |
| `BELIEFS.md` | generated view of live hypotheses | yes (generated) |
| `GRAVEYARD.md` | generated view of refuted/superseded hypotheses | yes (generated) |
| `ground-truth/registry.db` | derived SQLite + FTS5 search index | no (rebuilt on demand) |

The JSONL files are the source of truth. `BELIEFS.md` and `GRAVEYARD.md` are generated
and carry a do-not-edit banner. The database is a disposable cache rebuilt from the
JSONL; it exists only to make search fast. This split is the ground-truth contract:
canon is hand-authored, everything else is re-derivable by a command.

## The rule that stops rehashing

Before you propose or re-test any idea, run the search gate:

```
python registry.py search "energy attachment policy"
```

If a matching hypothesis comes back `REFUTED`, you see its falsifying evidence and its
**re-open gate**: the measurable condition under which retrying is allowed. You may not
re-test a refuted hypothesis unless that gate condition has actually changed. A refuted
idea is not deleted, it is tombstoned with the reason, so the next model finds the
reason instead of repeating the experiment.

## Lifecycle

```
open ──> supported    (evidence accumulated; still falsifiable)
     ──> refuted       (falsified; requires evidence + a re-open gate)
     ──> superseded    (replaced by a better hypothesis; point to it)
     ──> parked        (set aside; requires a re-open gate)
```

A status change always carries `--evidence`. `refuted` and `parked` also require
`--gate`. This is enforced by the tool, not by discipline.

## Commands

```
python registry.py add --title T --statement "falsifiable claim" \
    --test "how to test" --refute "what refutes it" --confidence medium --tags a,b
python registry.py status H003 supported --evidence "win rate 0.58 vs 0.50, n=400, kaggle-LB"
python registry.py status H004 refuted   --evidence "..." --gate "revisit only if the engine exposes hand info"
python registry.py experiment --hyp H003 --desc "ISMCTS 200 determinizations" --config "c=1.4" --code agent/search.py@abc123
python registry.py result --exp E007 --hyp H003 --metric win_rate --value 0.58 --baseline 0.50 --n 400 --verdict supports --provenance kaggle-LB
python registry.py search "deck matchup covariance"
python registry.py render      # regenerate the two .md views
python registry.py list
```

## Recording a result honestly

`results.jsonl` stores `metric`, `value`, optional `baseline` and `n`, a `verdict`
(`supports` / `refutes` / `inconclusive`), and `provenance` (`local-sim`, `kaggle-LB`,
`manual`). A win rate measured against one fixed opponent pool is a single aggregate
number. It tells you the net direction, not per-matchup mechanism. Record the number
and the verdict; do not write a per-matchup story into the hypothesis from one
aggregate. See `../AGENTS.md` for the full pre-commit checklist.
