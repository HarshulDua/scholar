"""Phi-3 chat prompt templates for paper explanation at different academic levels."""
from __future__ import annotations

PROMPT_UNDERGRAD = (
    "<|user|>\n"
    "You are a helpful tutor explaining a research paper to an undergraduate student.\n\n"
    "Paper title: {title}\n\n"
    "Abstract: {abstract}\n\n"
    "Why recommended: {why}\n\n"
    "Please provide:\n"
    "1. A 3-sentence plain-English summary. Avoid technical jargon — use everyday language "
    "that a first-year college student would understand.\n"
    "2. A glossary of 3-5 key terms with simple definitions (one line each).\n"
    "3. One sentence explaining why this paper is personally relevant to the reader.\n\n"
    "Format your response exactly as:\n"
    "SUMMARY: <3 sentences>\n"
    "GLOSSARY:\n- <term>: <definition>\n- <term>: <definition>\n"
    "WHY: <1 sentence>\n"
    "<|end|>\n"
    "<|assistant|>\n"
)

PROMPT_GRAD = (
    "<|user|>\n"
    "You are a knowledgeable research assistant explaining a paper to a graduate student.\n\n"
    "Paper title: {title}\n\n"
    "Abstract: {abstract}\n\n"
    "Why recommended: {why}\n\n"
    "Please provide:\n"
    "1. A 3-sentence technical summary using appropriate field terminology.\n"
    "2. A glossary of 3-5 key technical terms and concepts (one line each).\n"
    "3. One sentence on why this paper is relevant to the reader's interests.\n\n"
    "Format your response exactly as:\n"
    "SUMMARY: <3 sentences>\n"
    "GLOSSARY:\n- <term>: <definition>\n- <term>: <definition>\n"
    "WHY: <1 sentence>\n"
    "<|end|>\n"
    "<|assistant|>\n"
)

PROMPT_RESEARCHER = (
    "<|user|>\n"
    "You are a critical research peer reviewing a paper for an expert researcher.\n\n"
    "Paper title: {title}\n\n"
    "Abstract: {abstract}\n\n"
    "Why recommended: {why}\n\n"
    "Please provide:\n"
    "1. A 3-sentence expert summary focusing on the novel contributions, methodology, "
    "and key results — including any limitations or open questions.\n"
    "2. A glossary of 3-5 highly technical terms, models, or frameworks introduced or "
    "central to this work.\n"
    "3. One sentence on why this work is novel and relevant to the researcher.\n\n"
    "Format your response exactly as:\n"
    "SUMMARY: <3 sentences>\n"
    "GLOSSARY:\n- <term>: <definition>\n- <term>: <definition>\n"
    "WHY: <1 sentence>\n"
    "<|end|>\n"
    "<|assistant|>\n"
)

_LEVEL_MAP = {
    "undergrad": PROMPT_UNDERGRAD,
    "grad": PROMPT_GRAD,
    "researcher": PROMPT_RESEARCHER,
}


def get_prompt(level: str, title: str, abstract: str, why: str) -> str:
    template = _LEVEL_MAP.get(level, PROMPT_GRAD)
    return template.format(title=title, abstract=abstract, why=why)
