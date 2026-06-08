"""One-off helper: regenerate eval/golden_set.jsonl with chunk indices that
match THIS environment's PDF extraction.

Chunk indices depend on how pypdf extracts text, which varies by pypdf
version. Rather than hardcode indices, we anchor each question to a verified
English phrase (or phrases) from the statute and look up whichever chunk(s)
contain it in the locally-built corpus. Run this whenever the corpus or the
pypdf version changes:

    python -m eval.build_golden
    python -m eval.eval_retrieval

You can delete this file once you're happy with the golden set.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader

from app.chunker import chunk_text
from app.text_cleaning import strip_front_matter

SOURCE_PDF = Path("data/golden/paper.pdf")
OUT = Path("eval/golden_set.jsonl")
MAX_INDICES = 2  # include up to N matching chunks (covers boundary overlap)

# id, question, [required phrases — chunk must contain ALL], notes
QUESTIONS = [
    (
        "q01",
        "Within how many days after the tenancy ends must a landlord return the deposit if they have no claim against it?",
        ["no claim against it", "14 days"],
        "s.32(1) - 14 days after termination",
    ),
    (
        "q02",
        "What can a landlord make a claim against a security deposit for?",
        ["outstanding rent", "extraordinary cleaning"],
        "s.31.1 - outstanding rent, repair of damage, extraordinary cleaning",
    ),
    (
        "q03",
        "Can a tenant's deposit be garnished while held by the landlord or director?",
        ["garnishment"],
        "s.31.5 - deposits not subject to garnishment",
    ),
    (
        "q04",
        "What must a landlord do with a deposit owed to a tenant who cannot be located?",
        ["pay to the director", "amount owing"],
        "s.32(8) - pay the amount to the director",
    ),
    (
        "q05",
        "How much notice must a landlord give before removing or reducing a rent discount?",
        ["remove or reduce a rent discount"],
        "s.25(4) - at least three months' written notice",
    ),
    (
        "q06",
        "What happens to a rent increase if the landlord fails to give the required notice?",
        ["an increase in rent is void"],
        "s.25(3) - the increase is void",
    ),
    (
        "q07",
        "Do temporary tenancies renew automatically under the Act?",
        ["no automatic renewal for temporary"],
        "s.24 - no automatic renewal for temporary tenancies",
    ),
    (
        "q08",
        "How does the Act define 'rent'?",
        ['"rent" means the amount of money'],
        "Part 1 definitions",
    ),
    (
        "q09",
        "What types of deposit are included in the definition of 'deposit'?",
        ['"deposit" means'],
        "security deposit, pet damage deposit, tenant services security deposit",
    ),
    (
        "q10",
        "What is the maximum a landlord can charge for a pet damage deposit?",
        ["pet damage deposit must not be more than", "one month"],
        "s.29.1(4) - not more than one month's rent",
    ),
    (
        "q11",
        "When must a landlord give the tenant vacant possession of the rental unit?",
        ["landlord shall give vacant possession", "date the tenancy begins"],
        "s.56(1) - on the date the tenancy begins",
    ),
    (
        "q12",
        "If a landlord gives the tenant rules about pets, which provisions apply?",
        ["rules about pets"],
        "s.29.2 - subsections 11(2) and (3) apply",
    ),
    (
        "q13",
        "What must a landlord do before removing a tenant's abandoned property that has monetary value?",
        ["abandoned property that has monetary value"],
        "s.106.1(3) - contact tenant, prepare inventory, notify director",
    ),
    (
        "q14",
        "What happens to a tenancy when an order terminating it is made under The Safer Communities and Neighbourhoods Act?",
        ["safer communities and neighbourhoods act", "vacant possession"],
        "s.104(2) - tenancy terminates, landlord gets vacant possession",
    ),
    (
        "q15",
        "What is the landlord's obligation regarding the condition of the rental unit?",
        ["landlord", "good state of repair"],
        "landlord's duty to keep the unit in a good state of repair",
    ),
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower()


def main() -> None:
    text = "\n\n".join(p.extract_text() or "" for p in PdfReader(str(SOURCE_PDF)).pages)
    text = strip_front_matter(text)
    chunks = chunk_text(text)
    norm_chunks = [_norm(c.text) for c in chunks]
    print(f"corpus: {len(chunks)} chunks from {SOURCE_PDF}")

    lines = []
    for qid, question, phrases, notes in QUESTIONS:
        needles = [_norm(p) for p in phrases]
        matches = [i for i, t in enumerate(norm_chunks) if all(n in t for n in needles)]
        indices = matches[:MAX_INDICES]
        if not indices:
            print(f"  !! {qid}: NO MATCH for {phrases} — left as [0], fix manually")
            indices = [0]
        else:
            print(f"  {qid}: chunks {indices}")
        lines.append(
            json.dumps(
                {
                    "id": qid,
                    "question": question,
                    "expected_source_filename": SOURCE_PDF.name,
                    "expected_chunk_indices": indices,
                    "notes": notes,
                }
            )
        )

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} questions to {OUT}")


if __name__ == "__main__":
    main()
