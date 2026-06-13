"""Prompt Lab — propose and evaluate candidate prompts.

This module intentionally never activates a prompt automatically. It creates
candidate prompts and LLM evaluations so a human can validate the production
change with context.
"""

from __future__ import annotations

import logging
from string import Formatter

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


PROMPT_EVALUATOR_PROMPT = """Tu es évaluateur critique de prompts pour TechPulse.

Évalue ce prompt candidat pour la tâche {task}, thème {theme}.

Prompt actuel:
---
{current_prompt}
---

Prompt candidat:
---
{candidate_prompt}
---

Critères:
- clarté et maintenabilité
- profondeur réelle de l'analyse
- différenciation avec les autres écrans
- robustesse JSON
- réduction des hallucinations
- adaptation à TechPulse
- coût probable

Réponds uniquement en JSON:
{{
  "score": 0,
  "recommendation": "activate" | "keep_testing" | "reject",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "required_checks": ["..."],
  "final_assessment": "paragraphe court en français"
}}"""


def _extract_variables(template: str) -> list[str]:
    variables: list[str] = []
    for _, field_name, _, _ in Formatter().parse(template):
        if not field_name:
            continue
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        if root and root not in variables:
            variables.append(root)
    return variables


def propose_and_evaluate_prompt(
    cur,
    *,
    task: str,
    theme: str = "general",
    improvement_goal: str,
) -> str | None:
    """Create a candidate prompt and store a critic evaluation."""
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
    provider = "deepseek"
    model = "deepseek-v4-pro"

    if not proposal:
        proposal = analyze_with_openai(engineer_prompt, model="gpt-4o-mini")
        provider = "openai"
        model = "gpt-4o-mini"

    candidate_prompt = proposal.get("candidate_prompt") if proposal else None
    if not candidate_prompt:
        log.warning("Prompt Lab proposal failed for %s/%s", task, theme)
        return None

    candidate_id = db.insert_prompt_candidate(
        cur,
        task=task,
        theme=theme,
        template=candidate_prompt,
        variables=_extract_variables(candidate_prompt),
        parent_id=active["id"],
        model_provider=provider,
        model_name=model,
        evaluator_notes={
            "proposal": {
                "change_summary": proposal.get("change_summary", []),
                "expected_benefits": proposal.get("expected_benefits", []),
                "risks": proposal.get("risks", []),
            },
        },
    )

    evaluator_prompt = PROMPT_EVALUATOR_PROMPT.format(
        task=task,
        theme=theme,
        current_prompt=active["template"],
        candidate_prompt=candidate_prompt,
    )
    evaluation = analyze_with_openai(evaluator_prompt, model="gpt-4o-mini")
    evaluator_provider = "openai"
    evaluator_model = "gpt-4o-mini"

    if not evaluation:
        evaluation = analyze_with_deepseek(evaluator_prompt, model="deepseek-v4-flash")
        evaluator_provider = "deepseek"
        evaluator_model = "deepseek-v4-flash"

    if evaluation:
        db.insert_prompt_evaluation(
            cur,
            prompt_template_id=candidate_id,
            evaluator_provider=evaluator_provider,
            evaluator_model=evaluator_model,
            score=int(evaluation.get("score") or 0),
            recommendation=evaluation.get("recommendation") or "keep_testing",
            notes=evaluation,
            sample_count=0,
        )

    log.info("Prompt Lab candidate created: %s for %s/%s", candidate_id, task, theme)
    return candidate_id
