"""The pre-registered query set: shape, uniqueness, and agreement with the document."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from srb import queries as Q
from srb.datasets.cholec80 import PHASES, TOOL_COLUMNS, TOOLS

DOC = Path(__file__).resolve().parents[1] / "docs" / "preregistration.md"


def _all_texts() -> list[str]:
    out = [t for d in (Q.PHASE_QUERIES, Q.TOOL_QUERIES, Q.CVS_QUERIES) for v in d.values() for t in v]
    return out + list(Q.NEGATIVE_QUERIES)


def test_counts_match_the_protocol():
    assert set(Q.PHASE_QUERIES) == set(PHASES)
    assert set(Q.TOOL_QUERIES) == set(TOOLS)
    assert set(Q.CVS_QUERIES) == {"C1", "C2", "C3"}
    for d in (Q.PHASE_QUERIES, Q.TOOL_QUERIES, Q.CVS_QUERIES):
        assert all(len(v) == len(Q.REPHRASING_ROLES) == 5 for v in d.values())
    assert len(Q.NEGATIVE_QUERIES) >= 5


def test_every_wording_is_unique_and_clean():
    texts = _all_texts()
    assert len(texts) == len(set(texts))
    assert all(t == t.strip() and "  " not in t and t for t in texts)


def test_document_lists_exactly_the_code_queries():
    """Every query string is in the doc verbatim, and the doc lists no other."""
    doc = DOC.read_text()
    listed = re.findall(r"^\* (?:r[1-5]|neg\d+): (.+)$", doc, flags=re.M)
    assert sorted(listed) == sorted(_all_texts())


def test_cholec80_queries_build_with_correct_targets():
    qs = Q.cholec80_queries()
    assert len(qs) == 35 + 35 + len(Q.NEGATIVE_QUERIES)
    assert len({q.qid for q in qs}) == len(qs)
    assert {q.target for q in qs if q.kind == "phase"} == set(PHASES)
    assert {q.target for q in qs if q.kind == "tool"} == set(TOOL_COLUMNS)
    m = pd.DataFrame({"phase": ["Preparation", "ClippingCutting"],
                      **{c: [c == "tool_hook", False] for c in TOOL_COLUMNS}})
    for q in qs:
        rel = q.relevant(m)
        if q.kind == "negative":
            assert not rel.any()
        elif q.target == "Preparation" or q.target == "tool_hook":
            assert rel.tolist() == [True, False]


@pytest.mark.parametrize("model", sorted(Q.TEXT_TOKENIZERS))
def test_no_query_is_truncated_by_any_text_encoder(model):
    """Truncation is silent: a query over the limit would be scored cut off.

    Uses the pinned tokenizer from the local HF cache; skips (with the reason) only if
    that tokenizer has never been downloaded on this machine.
    """
    from transformers import AutoTokenizer

    hf_id, rev, limit = Q.TEXT_TOKENIZERS[model]
    try:
        tok = AutoTokenizer.from_pretrained(hf_id, revision=rev, local_files_only=True)
    except OSError:
        pytest.skip(f"{hf_id}@{rev[:8]} tokenizer not in the local HF cache")
    too_long = {t: n for t in _all_texts() if (n := len(tok(t)["input_ids"])) > limit}
    assert not too_long, f"{model}: over {limit} tokens: {too_long}"
