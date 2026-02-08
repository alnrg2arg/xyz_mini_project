"""Central taxonomy and schemas for FetchReach VLM pipeline."""

REACH_STAGE_ENUM = ["approach", "reach", "align", "idle", "unknown"]
REACH_FAILURE_ENUM = ["goal_mismatch", "slow_progress", "oscillation", "occlusion", "unknown"]

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "p_success": {"type": "number", "minimum": 0, "maximum": 1},
        "progress": {"type": "number", "minimum": 0, "maximum": 1},
        "uncertainty": {"type": "number", "minimum": 0, "maximum": 1},
        "judge_notes": {"type": "string"},
    },
    "required": ["p_success", "progress", "uncertainty", "judge_notes"],
    "additionalProperties": False,
}

TAG_SCHEMA = {
    "type": "object",
    "properties": {
        "stage": {"type": "string", "enum": REACH_STAGE_ENUM},
        "failure_type": {"type": "string", "enum": REACH_FAILURE_ENUM},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "notes": {"type": "string"},
    },
    "required": ["stage", "failure_type", "confidence", "notes"],
    "additionalProperties": False,
}


def make_prompt_tagging() -> str:
    return (
        "You are analyzing a robot manipulation episode frame.\n"
        "Task: FetchReach (move gripper to target position).\n\n"
        "Important:\n"
        "- Choose 'unknown' only if genuinely ambiguous.\n"
        "- Prefer a concrete failure_type if you can justify it visually.\n"
        "- Use these intuitive definitions:\n"
        "  - goal_mismatch: moving away from target / wrong direction\n"
        "  - slow_progress: little progress for this late timestep\n"
        "  - oscillation: jittery back-and-forth motion\n"
        "  - occlusion: target/gripper view is blocked\n"
        "Return ONLY valid JSON with fields:\n"
        f"- stage: one of {REACH_STAGE_ENUM}\n"
        f"- failure_type: one of {REACH_FAILURE_ENUM}\n"
        "- confidence: number 0..1\n"
        "- notes: one short sentence, only what is visually evident (no speculation)\n"
    )


def make_prompt_judge() -> str:
    return (
        "You are a success/progress judge for a robot manipulation episode frame.\n"
        "Task: move the gripper to the target position.\n\n"
        "Return ONLY valid JSON with fields:\n"
        "- p_success: number 0..1 (probability the episode will succeed)\n"
        "- progress: number 0..1 (how close to completion)\n"
        "- uncertainty: number 0..1\n"
        "- judge_notes: one short sentence (visual evidence only)\n"
    )
