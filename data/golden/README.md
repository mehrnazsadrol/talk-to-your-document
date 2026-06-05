# Golden eval corpus

This directory holds the source files for the offline retrieval eval
(`python -m eval.eval_retrieval`).

The agent that scaffolded this ticket **cannot fetch external files**, so
the two real source files are not committed here yet. You need to drop
them in manually before the eval will produce a meaningful hit-rate.

## What to add

1. **`paper.pdf`** — a short PDF (one arXiv paper is ideal,
   e.g. something from `arxiv.org/abs/...` under ~3 MB). The eval will
   chunk and embed it.
2. **`clip.mp3`** — a short audio clip in English, under ~30 seconds.
   Any clean speech sample works.

Combined size must stay under ~5 MB so the repo doesn't bloat.

## After you've added the files

1. Open `eval/golden_set.jsonl`.
2. Replace the two placeholder entries with 15 real Q&A pairs:
   - 10 questions whose answers live in `paper.pdf`
   - 5 questions whose answers live in `clip.mp3`
3. For each entry, set `expected_chunk_indices` to the chunk index (or
   indices) that contain the answer. You can discover these by running
   the eval once with the placeholder set and inspecting the JSON it
   writes under `eval/results/` — the per-question record includes the
   top-5 retrieved `(filename, chunk_index, distance)` tuples.

## Why this directory is committed but `data/` is not

`.gitignore` ignores `data/*` but explicitly carves out `data/samples/`
(unit-test fixtures) and `data/golden/` (this directory). The PDF and
MP3 you add here will be checked into git so the eval is reproducible
across machines.
