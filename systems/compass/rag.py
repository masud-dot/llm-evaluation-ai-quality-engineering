"""systems/compass/rag.py: Compass with retrieval.

A deliberately simple lexical retriever: this book evaluates
retrieval, it does not teach how to build or tune it.
"""
from __future__ import annotations

import re
from pathlib import Path

from aiqe.evaluators.retrieval import Passage, RagTrace
from aiqe.providers import Provider

WORD = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "of", "to", "for", "my", "i", "is",
        "do", "how", "what", "and", "or", "in", "on", "it"}


def load_corpus(folder: Path) -> list[Passage]:
    return [Passage(id=p.stem, text=p.read_text(encoding="utf-8"))
            for p in sorted(folder.glob("*.md"))]


def terms(text: str) -> set[str]:
    return set(WORD.findall(text.lower())) - STOP


class LexicalRetriever:
    def __init__(self, corpus: list[Passage], k: int) -> None:
        self.corpus = corpus
        self.k = k

    def __call__(self, query: str) -> list[Passage]:
        q = terms(query)
        ranked = sorted(self.corpus,
                        key=lambda p: (-len(q & terms(p.text)),
                                       p.id))
        return ranked[:self.k]


class RagCompass:
    def __init__(self, provider: Provider, model: str,
                 system_prompt: str,
                 retriever: LexicalRetriever) -> None:
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.retriever = retriever

    def answer(self, case_id: str, question: str) -> RagTrace:
        passages = self.retriever(question)
        context = "\n\n".join(f"[{p.id}] {p.text}"
                              for p in passages)
        prompt = (f"{self.system_prompt}\n\nPASSAGES:\n{context}"
                  f"\n\nCUSTOMER: {question}")
        text = self.provider.complete(self.model, prompt).text
        return RagTrace(case_id=case_id, query=question,
                        retrieved=passages, answer=text)
