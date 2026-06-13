"""Prompt registry helpers.

The registry makes prompts configurable and measurable without allowing an LLM
to overwrite production prompts directly. Code prompts remain the fallback; the
database can override them only through an active prompt template.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from string import Formatter
from typing import Any

from . import db

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    template_id: str | None
    task: str
    theme: str
    version: int | None
    source: str


def _extract_variables(template: str) -> list[str]:
    variables: list[str] = []
    for _, field_name, _, _ in Formatter().parse(template):
        if not field_name:
            continue
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        if root and root not in variables:
            variables.append(root)
    return variables


def seed_default_prompt(
    cur,
    *,
    task: str,
    template: str,
    theme: str = "general",
    model_provider: str | None = None,
    model_name: str | None = None,
) -> None:
    db.seed_prompt_template(
        cur,
        task=task,
        theme=theme,
        template=template,
        variables=_extract_variables(template),
        model_provider=model_provider,
        model_name=model_name,
    )


def render_prompt(
    cur,
    *,
    task: str,
    fallback_template: str,
    values: dict[str, Any],
    theme: str = "general",
    model_provider: str | None = None,
    model_name: str | None = None,
) -> RenderedPrompt:
    """Render the active DB prompt, falling back to the code template.

    The fallback is seeded as version 1. If a DB prompt is malformed or missing
    a variable, the pipeline keeps running with the code fallback.
    """
    seed_default_prompt(
        cur,
        task=task,
        theme="general",
        template=fallback_template,
        model_provider=model_provider,
        model_name=model_name,
    )

    row = db.fetch_active_prompt_template(cur, task=task, theme=theme)
    if row:
        try:
            return RenderedPrompt(
                text=row["template"].format(**values),
                template_id=row["id"],
                task=task,
                theme=row["theme"],
                version=row["version"],
                source="db",
            )
        except Exception as exc:
            log.warning(
                "Prompt template %s (%s/%s v%s) failed, using code fallback: %s",
                row.get("id"),
                task,
                row.get("theme"),
                row.get("version"),
                exc,
            )

    return RenderedPrompt(
        text=fallback_template.format(**values),
        template_id=None,
        task=task,
        theme=theme,
        version=None,
        source="code",
    )
