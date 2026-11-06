import re
from dataclasses import dataclass

from localrag import db, nvidia

RRF_K = 60


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, 1):
            scores[i] = scores.get(i, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


EXPAND_PROMPT = (
    "Read the question below. The documents are technical standards and specifications: use the formal technical "
    "terms a standard uses instead of everyday words.\n"
    "Reply with these lines in this order and no other text:\n"
    "- One line INTENT: <what the question asks for>.\n"
    "- {n} lines PHRASING: <phrasing>, with {n} different phrasings of the question, in the same language as the "
    "question. No numbering, no quotes.\n"
    "- At most {groups} lines KEYWORDS: <term>; <synonym>; ...; <English term>; <English synonym>; ..., one line "
    "per domain term of the question. Each line holds the term and at most {synonyms} synonyms in the language of "
    "the question, then the same term and at most {synonyms} synonyms in English. "
    "For an English question, write English only.\n"
    "Example for a Ukrainian question: 'Як варити труби?' becomes 'Як зварювати трубопроводи?'. "
    "Write the intent and the phrasings in the language of the question, not in the language of the example. "
    "Write the keywords in the language of the question and in English; when the question is in English, "
    "every keyword is in English."
    "\n\nQuestion: {question}")

_LABELLED = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*\**\s*(INTENT|PHRASING|KEYWORDS)\s*\**\s*:\s*\**(.*?)\**\s*$",
                       re.IGNORECASE)
_UNLABELLED = re.compile(r"^\s*(?:[-*•]\s|\d+[.)])?\s*(.*?)\s*$")


@dataclass(frozen=True)
class Expansion:
    paraphrases: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()


def expand(cfg, question, chat_stream=nvidia.chat_stream) -> Expansion:
    if not cfg["paraphrase"]:
        return Expansion()
    n = max(cfg["paraphrases"], 0)
    prompt = EXPAND_PROMPT.format(n=n, groups=cfg["keyword_groups"],
                                  synonyms=cfg["keyword_synonyms"], question=question)
    try:
        reply = "".join(t for kind, t in chat_stream(cfg, [{"role": "user", "content": prompt}], "none")
                        if kind == "token")
    except Exception:
        return Expansion()  # the answer runs on the original question alone
    return _parse_expansion(reply, question, n, cfg["keyword_groups"])


def _parse_expansion(reply, question, n, groups) -> Expansion:
    phrasings, terms, keyword_groups, seen = [], [], 0, set()
    for label, body in _sectioned_lines(reply):
        if label == "PHRASING":
            phrasing = body.strip("\"'«»“”").strip()
            if phrasing and phrasing.casefold() != question.strip().casefold() and phrasing not in phrasings:
                phrasings.append(phrasing)
        elif label == "KEYWORDS" and keyword_groups < groups:
            keyword_groups += 1
            for term in (t.strip() for t in body.split(";")):
                if term and term.casefold() not in seen:
                    seen.add(term.casefold())
                    terms.append(term)
    return Expansion(tuple(phrasings[:n]), tuple(terms))


def _sectioned_lines(reply):
    """Yields (label, body) per non-empty line; an unlabelled line takes the label of the section it is in."""
    label = None
    for line in reply.splitlines():
        match = _LABELLED.match(line)
        if match:
            label, body = match.group(1).upper(), match.group(2).strip()
        elif label:
            body = _UNLABELLED.match(line).group(1).strip("* ")
        else:
            continue
        if body:
            yield label, body


def retrieve(conn, cfg, workspace_id, question, *, embed=nvidia.embed, rerank=nvidia.rerank,
             chat_stream=nvidia.chat_stream) -> list[dict]:
    if not question.strip():
        return []
    expansion = expand(cfg, question, chat_stream)
    rankings = [*search_rankings(conn, cfg, workspace_id, [question, *expansion.paraphrases], embed),
                keyword_ranking(conn, cfg, workspace_id, expansion.keywords)]
    return pick_sources(conn, cfg, question, fuse(cfg, rankings), rerank)


def search_rankings(conn, cfg, workspace_id, queries, embed) -> list[list[int]]:
    fts = [[i for i, _ in db.fts_search(conn, workspace_id, q, cfg["candidates"])] for q in queries]
    vec = [[i for i, _ in db.vec_search(conn, workspace_id, v, cfg["candidates"])]
           for v in embed(cfg, queries, "query")]
    return fts + vec


def keyword_ranking(conn, cfg, workspace_id, keywords) -> list[int]:
    return [i for i, _ in db.fts_search_terms(conn, workspace_id, keywords, cfg["candidates"])]


def fuse(cfg, rankings) -> list[tuple[int, float]]:
    return rrf(rankings)[:cfg["rerank_candidates"]]


def pick_sources(conn, cfg, question, fused, rerank) -> list[dict]:
    if not fused:
        return []
    rows = db.get_chunks(conn, [i for i, _ in fused])
    if cfg["rerank"]:
        ranked = rerank(cfg, question, [r["text"] for r in rows])[:cfg["sources"]]
        picked = [(rows[i], s) for i, s in ranked]
    else:
        picked = list(zip(rows, [s for _, s in fused]))[:cfg["sources"]]
    keys = ("id", "document_id", "document_name", "page", "text")
    return [{**{k: r[k] for k in keys}, "score": s} for r, s in picked]
