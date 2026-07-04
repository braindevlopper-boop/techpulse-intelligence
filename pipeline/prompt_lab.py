"""Prompt Lab — propose a candidate prompt improvement (R&D helper).

There is no D1 route for prompt_evaluations/candidate versioning (Prompt Lab
is a low-usage R&D feature per the D1 migration scope), so this module no
longer stores candidates or evaluations. It still calls the LLM to draft a
suggestion and logs it for manual review — nothing is persisted or activated
automatically.
"""

from __future__ import annotations

import logging

from . import db
from .llm_analyzer import analyze_with_deepseek, analyze_with_openai

log = logging.getLogger(__name__)

PROMPT_ENGINEER_PROMPT = """Tu es Prompt Engineer senior pour TechPulse.

TechPulse est une application personnelle de veille technologique, financière
et économique. Le niveau attendu est premium: concret, stratégique,
pédagogique, fiable, sans remplissage générique.

Tâche à améliorer: {task}
Thème: {theme}

Prompt actif actuel:
---
{current_prompt}
---

Objectif d'amélioration:
{improvement_goal}

Contraintes:
- Conserve toutes les variables existantes entre accolades.
- Ne supprime aucune contrainte de JSON valide si elle existe.
- Différencie clairement l'analyse pédagogique d'un simple résumé.
- Réduis les risques d'hallucination et de dates inventées.
- Ajoute des critères métier uniquement s'ils aident vraiment TechPulse.

Réponds uniquement en JSON:
{{
  "candidate_prompt": "prompt complet prêt à stocker",
  "change_summary": ["3 à 6 changements clés"],
  "expected_benefits": ["2 à 5 bénéfices"],
  "risks": ["2 à 5 risques ou points à vérifier"]
}}"""


def propose_and_evaluate_prompt(
    cur,
    *,
    task: str,
    theme: str = "general",
    improvement_goal: str,
) -> str | None:
    """Draft a candidate prompt and log it. No storage: Prompt Lab evaluation
    has no D1 route (secondary R&D feature), so this is a skip-with-log no-op
    beyond generating the suggestion for manual copy/paste review."""
    active = db.fetch_active_prompt_template(cur, task=task, theme=theme)
    if not active:
        log.warning("No active prompt found for %s/%s", task, theme)
        return None

    engineer_prompt = PROMPT_ENGINEER_PROMPT.format(
        task=task,
        theme=theme,
        current_prompt=active["template"],
        improvement_goal=improvement_goal,
    )
    proposal = analyze_with_deepseek(engineer_prompt, model="deepseek-v4-pro")
    if not proposal:
        proposal = analyze_with_openai(engineer_prompt, model="gpt-4o-mini")

    candidate_prompt = proposal.get("candidate_prompt") if proposal else None
    if not candidate_prompt:
        log.warning("Prompt Lab proposal failed for %s/%s", task, theme)
        return None

    log.info(
        "Prompt Lab candidate for %s/%s (evaluation skipped, no D1 route):\n%s",
        task, theme, candidate_prompt,
    )
    log.info("Change summary: %s", proposal.get("change_summary", []))
    return None
