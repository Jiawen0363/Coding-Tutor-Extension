"""Privileged-information blocks for teacher-conditioned distillation.

Training objective (same dialogue state s_t):

    Teacher:  pi_teacher(a | s_t, z)   with privileged context z
    Student:  pi_theta(a | s_t)         learns from teacher signals only

At deployment the student tutor never receives z.
"""

from __future__ import annotations

import re
from typing import Any

REFERENCE_KNOWLEDGE_MARKER = "Reference Knowledge:"
GOAL_DESCRIPTION_MARKER = "Goal Description:"

STUDENT_LEVEL_LABELS = {
    "low_level": "low level (minimal prior knowledge)",
    "med_level": "medium level (partial dependency knowledge)",
    "high_level": "high level (broader dependency and partial solution knowledge)",
}


def _normalize_dep_path(path: str) -> str:
    path = path.replace("{", "[").replace("}", "]").strip()
    if "\n" in path:
        lines = [line.strip() for line in path.splitlines() if line.strip()]
        return lines[-1] if lines else path
    return path


def _parse_dependency_paths(raw: str) -> list[str]:
    paths = [_normalize_dep_path(path) for path in str(raw or "").strip().split("\n\n")]
    return [path for path in paths if path]


def _parse_reference_steps(raw: str) -> list[str]:
    steps = [
        step.strip()
        for step in re.split(r"\n(?=\d+\.)", str(raw or "").strip())
        if step.strip()
    ]
    return [step.replace("{", "[").replace("}", "]") for step in steps]


def _compact_kc_labels(indices: list[int]) -> str:
    if not indices:
        return "none"

    labels: list[str] = []
    start = prev = indices[0]
    for idx in indices[1:] + [None]:
        if idx == prev + 1:
            prev = idx
            continue
        labels.append(f"KC-{start}" if start == prev else f"KC-{start}–KC-{prev}")
        if idx is not None:
            start = prev = idx
    return ", ".join(labels)


def student_knowledge_gaps(element: dict[str, Any], student_level: str) -> tuple[list[int], list[int]]:
    """Return dependency KC indices in the student prompt vs still missing."""
    if student_level not in STUDENT_LEVEL_LABELS:
        raise ValueError(f"Unsupported student level: {student_level}")

    deps_all = _parse_dependency_paths(element.get("dependency_all", ""))
    deps_known = _parse_dependency_paths(element.get("dependency_sampled", "")) if student_level != "low_level" else []

    known_indices: list[int] = []
    missing_indices: list[int] = []
    known_dep_set = set(deps_known)
    for idx, dep in enumerate(deps_all, start=1):
        if dep in known_dep_set:
            known_indices.append(idx)
        else:
            missing_indices.append(idx)
    return known_indices, missing_indices


def student_known_summary(element: dict[str, Any], student_level: str) -> str:
    """Knowledge already provided to the student simulator prompt."""
    if student_level == "low_level":
        return "- Basic Python only"

    lines: list[str] = []
    if student_level == "high_level":
        lines.append("- Repository context (contexts_above)")

    known_indices, _ = student_knowledge_gaps(element, student_level)
    dep_labels = _compact_kc_labels(known_indices)
    if dep_labels != "none":
        lines.append(f"- Dependencies: {dep_labels}")
    return "\n".join(lines) if lines else "- none"


def student_missing_summary(element: dict[str, Any], student_level: str) -> str:
    """Knowledge missing from the student simulator prompt."""
    _, missing_dep_indices = student_knowledge_gaps(element, student_level)
    lines: list[str] = []

    if student_level in {"low_level", "med_level"}:
        lines.append("- Repository context (contexts_above)")

    dep_labels = _compact_kc_labels(missing_dep_indices)
    if dep_labels != "none":
        lines.append(f"- Dependencies: {dep_labels}")

    steps_all = _parse_reference_steps(element.get("reference_steps", ""))
    if steps_all:
        step_offset = len(_parse_dependency_paths(element.get("dependency_all", "")))
        step_labels = _compact_kc_labels(list(range(step_offset + 1, step_offset + len(steps_all) + 1)))
        lines.append(f"- Solution steps: {step_labels}")

    return "\n".join(lines) if lines else "- none"


def student_prior_knowledge(element: dict[str, Any], student_level: str) -> str:
    """Known-information summary for v1 privileged tutor prompts."""
    if student_level == "low_level":
        return "The simulated student starts with no explicit dependency or solution prior."

    if student_level == "med_level":
        dep = str(element.get("dependency_sampled", "")).strip()
        return f"Known dependency prior:\n{dep}"

    if student_level == "high_level":
        dep = str(element.get("dependency_sampled", "")).strip()
        ref_steps_raw = str(element.get("reference_steps", "")).strip()
        first_step = ref_steps_raw.split("2.")[0].strip() if ref_steps_raw else ""
        if dep and first_step:
            body = f"{dep}\n\n{first_step}"
        else:
            body = dep or first_step
        return f"Known dependency and partial solution prior:\n{body}"

    raise ValueError(f"Unsupported student level: {student_level}")


def reference_knowledge_block(element: dict[str, Any], *, contexts_above: str) -> str:
    """Format oracle reference knowledge for the teacher."""
    dependency_paths = str(element.get("dependency_all", "")).strip().split("\n\n")
    reference_steps = _format_kc_lines(dependency_paths, str(element.get("reference_steps", "")))
    return (
        f"{REFERENCE_KNOWLEDGE_MARKER}\n"
        f"- The contexts above the {element['function_name']} function:\n"
        f"```Python\n{contexts_above}\n```\n"
        f"- The dependency paths for the {element['function_name']} function:\n"
        f"{reference_steps['dependency']}\n"
        f"- The reference key solution steps:\n"
        f"{reference_steps['steps']}"
    )


def compose_privileged_context(
    *,
    reference_block: str,
    student_level: str | None = None,
    student_prior: str | None = None,
    extra: str | None = None,
) -> str:
    """Assemble teacher-only privileged context z."""
    sections: list[str] = []

    if student_level is not None:
        label = STUDENT_LEVEL_LABELS.get(student_level, student_level)
        sections.append(f"Privileged Student Profile:\n- Simulated student level: {label}")

    if student_prior:
        sections.append(f"Privileged Student Prior / Missing Knowledge:\n{student_prior.strip()}")

    if reference_block.strip():
        sections.append(reference_block.strip())

    if extra and extra.strip():
        sections.append(extra.strip())

    if not sections:
        raise ValueError("Privileged context must contain at least one non-empty section.")
    return "\n\n".join(sections)


def split_instruction_privileged(instruction: str) -> tuple[str, str | None]:
    """Split a legacy tutor instruction into student-visible text and reference knowledge."""
    if REFERENCE_KNOWLEDGE_MARKER not in instruction:
        return instruction.strip(), None

    before, after = instruction.split(REFERENCE_KNOWLEDGE_MARKER, 1)
    if GOAL_DESCRIPTION_MARKER in after:
        reference_block, goal_and_rest = after.split(GOAL_DESCRIPTION_MARKER, 1)
        student_instruction = (before + GOAL_DESCRIPTION_MARKER + goal_and_rest).strip()
        privileged_context = (REFERENCE_KNOWLEDGE_MARKER + reference_block.strip()).strip()
    else:
        student_instruction = before.strip()
        privileged_context = (REFERENCE_KNOWLEDGE_MARKER + after.strip()).strip()

    return student_instruction, privileged_context


def build_privileged_from_record(record: dict[str, Any]) -> str:
    """Resolve teacher privileged context from explicit or legacy record fields."""
    if record.get("privileged_context"):
        return str(record["privileged_context"]).strip()

    _, reference_only = split_instruction_privileged(str(record.get("instruction") or record.get("prompt") or ""))
    if not reference_only:
        raise ValueError("Record is missing privileged context and reference knowledge.")

    return compose_privileged_context(
        reference_block=reference_only,
        student_level=record.get("student_level"),
        student_prior=record.get("student_prior_knowledge"),
        extra=record.get("privileged_extra"),
    )


def student_observation(record: dict[str, Any]) -> str:
    """Resolve the student-visible instruction portion of s_t."""
    if record.get("student_instruction"):
        return str(record["student_instruction"]).strip()

    instruction = record.get("instruction") or record.get("prompt")
    if not instruction:
        raise ValueError("Record must contain `instruction`, `prompt`, or `student_instruction`.")
    student_instruction, _ = split_instruction_privileged(str(instruction))
    return student_instruction


def _format_kc_lines(dependency_paths: list[str], reference_steps_raw: str) -> dict[str, str]:
    kc_dependency, kc_reference = [], []
    idx = 1
    for dp in dependency_paths:
        dp = dp.replace("{", "[").replace("}", "]").strip()
        if dp:
            kc_dependency.append(f"KC-{idx}: {dp}")
            idx += 1

    steps = [step.strip() for step in re.split(r"\n(?=\d+\.)", reference_steps_raw.strip()) if step.strip()]
    for step in steps:
        step = step.replace("{", "[").replace("}", "]")
        kc_reference.append(f"KC-{idx}: {step}")
        idx += 1

    return {
        "dependency": "\n".join(kc_dependency),
        "steps": "\n".join(kc_reference),
    }
