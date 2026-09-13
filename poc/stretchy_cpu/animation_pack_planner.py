from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class AnimationPackTask:
    slot_key: str
    asset_type: str
    view_type: str | None
    pose: str | None
    expression: str | None
    required: bool
    prompt_suffix: str

    def to_dict(self) -> dict:
        return asdict(self)


ATOMIC_NEGATIVE_RULES = (
    "Generate exactly one subject and exactly one requested view or pose.",
    "Do not create a contact sheet, turnaround sheet, collage, split panel, storyboard, grid, or multiple poses.",
    "Do not duplicate the character.",
    "Do not add text, labels, annotations, watermarks, or logos.",
    "Keep the full body visible unless the slot explicitly requests a facial crop.",
    "Preserve the approved character identity, face, hairstyle, skin tone, clothing, proportions, and visual style.",
)


def _atomic_prompt(slot_description: str) -> str:
    rules = " ".join(ATOMIC_NEGATIVE_RULES)
    return f"{slot_description} {rules}"


def humanoid_plan() -> list[AnimationPackTask]:
    return [
        AnimationPackTask(
            "front_neutral",
            "character_animation_view",
            "front",
            "neutral_standing",
            "neutral",
            True,
            _atomic_prompt(
                "Create the approved character in a front-facing neutral full-body standing pose. Arms should be slightly separated from the torso, legs clearly separated, hands visible, and joints unobscured where possible. Use a transparent background or plain removable light background."
            ),
        ),
        AnimationPackTask(
            "three_quarter_left",
            "character_animation_view",
            "three_quarter_left",
            "neutral_standing",
            "neutral",
            True,
            _atomic_prompt(
                "Create one three-quarter-left full-body neutral standing view of the approved character. Preserve exact identity and wardrobe. Keep arms and legs readable for rigging."
            ),
        ),
        AnimationPackTask(
            "three_quarter_right",
            "character_animation_view",
            "three_quarter_right",
            "neutral_standing",
            "neutral",
            True,
            _atomic_prompt(
                "Create one three-quarter-right full-body neutral standing view of the approved character. Preserve exact identity and wardrobe. Keep arms and legs readable for rigging."
            ),
        ),
        AnimationPackTask(
            "side_left",
            "character_animation_view",
            "side_left",
            "neutral_standing",
            "neutral",
            True,
            _atomic_prompt(
                "Create one full-body left-profile neutral standing view of the approved character. Preserve exact identity and wardrobe. Do not show any second pose or alternate angle."
            ),
        ),
        AnimationPackTask(
            "side_right",
            "character_animation_view",
            "side_right",
            "neutral_standing",
            "neutral",
            True,
            _atomic_prompt(
                "Create one full-body right-profile neutral standing view of the approved character. Preserve exact identity and wardrobe. Do not show any second pose or alternate angle."
            ),
        ),
        AnimationPackTask(
            "expression_happy",
            "character_expression",
            "front",
            None,
            "happy",
            True,
            _atomic_prompt(
                "Create one front-facing facial expression asset of the approved character showing a clear happy expression. Keep framing and identity consistent with the approved character."
            ),
        ),
        AnimationPackTask(
            "expression_sad",
            "character_expression",
            "front",
            None,
            "sad",
            False,
            _atomic_prompt(
                "Create one front-facing facial expression asset of the approved character showing a clear sad expression. Keep framing and identity consistent with the approved character."
            ),
        ),
        AnimationPackTask(
            "expression_surprised",
            "character_expression",
            "front",
            None,
            "surprised",
            False,
            _atomic_prompt(
                "Create one front-facing facial expression asset of the approved character showing a clear surprised expression. Keep framing and identity consistent with the approved character."
            ),
        ),
        AnimationPackTask(
            "mouth_rest",
            "character_mouth_shape",
            "front",
            None,
            "mouth_rest",
            True,
            _atomic_prompt(
                "Create one tightly framed front-facing mouth asset for the approved character with lips naturally closed at rest. Keep skin tone and rendering style identical to the approved character."
            ),
        ),
        AnimationPackTask(
            "mouth_A",
            "character_mouth_shape",
            "front",
            None,
            "mouth_A",
            True,
            _atomic_prompt(
                "Create one tightly framed front-facing mouth asset for the approved character forming a clear A sound mouth shape."
            ),
        ),
        AnimationPackTask(
            "mouth_E",
            "character_mouth_shape",
            "front",
            None,
            "mouth_E",
            True,
            _atomic_prompt(
                "Create one tightly framed front-facing mouth asset for the approved character forming a clear E sound mouth shape."
            ),
        ),
        AnimationPackTask(
            "mouth_O",
            "character_mouth_shape",
            "front",
            None,
            "mouth_O",
            True,
            _atomic_prompt(
                "Create one tightly framed front-facing mouth asset for the approved character forming a clear O sound mouth shape."
            ),
        ),
        AnimationPackTask(
            "mouth_MBP",
            "character_mouth_shape",
            "front",
            None,
            "mouth_MBP",
            True,
            _atomic_prompt(
                "Create one tightly framed front-facing mouth asset for the approved character forming a closed-lips M/B/P sound shape."
            ),
        ),
        AnimationPackTask(
            "mouth_FV",
            "character_mouth_shape",
            "front",
            None,
            "mouth_FV",
            False,
            _atomic_prompt(
                "Create one tightly framed front-facing mouth asset for the approved character forming an F/V sound shape."
            ),
        ),
        AnimationPackTask(
            "mouth_L",
            "character_mouth_shape",
            "front",
            None,
            "mouth_L",
            False,
            _atomic_prompt(
                "Create one tightly framed front-facing mouth asset for the approved character forming an L sound shape."
            ),
        ),
    ]


def quadruped_plan() -> list[AnimationPackTask]:
    slots = [
        ("side_left", "side_left", "Create one left-profile full-body neutral standing view of the approved animal. Keep all four legs visible and separated where possible."),
        ("side_right", "side_right", "Create one right-profile full-body neutral standing view of the approved animal. Keep all four legs visible and separated where possible."),
        ("front_neutral", "front", "Create one front-facing full-body neutral standing view of the approved animal."),
        ("three_quarter_left", "three_quarter_left", "Create one three-quarter-left full-body neutral standing view of the approved animal."),
    ]
    return [AnimationPackTask(k, "character_animation_view", v, "neutral_standing", "neutral", True, _atomic_prompt(desc)) for k, v, desc in slots]


def bird_plan() -> list[AnimationPackTask]:
    slots = [
        ("side_left_perched", "side_left", "Create one left-profile full-body perched view of the approved bird, with wings folded and feet visible."),
        ("side_right_perched", "side_right", "Create one right-profile full-body perched view of the approved bird, with wings folded and feet visible."),
        ("front_perched", "front", "Create one front-facing perched view of the approved bird."),
        ("wings_open", "front", "Create one front-facing full-body view of the approved bird with both wings open symmetrically."),
    ]
    return [AnimationPackTask(k, "character_animation_view", v, "perched" if "perched" in k else "wings_open", "neutral", True, _atomic_prompt(desc)) for k, v, desc in slots]


def vehicle_plan() -> list[AnimationPackTask]:
    slots = [
        ("side_left", "side_left", "Create one clean left-side view of the approved vehicle, fully visible and centered."),
        ("side_right", "side_right", "Create one clean right-side view of the approved vehicle, fully visible and centered."),
        ("front", "front", "Create one clean front view of the approved vehicle, fully visible and centered."),
        ("three_quarter_left", "three_quarter_left", "Create one three-quarter-left view of the approved vehicle, fully visible and centered."),
    ]
    return [AnimationPackTask(k, "vehicle_animation_view", v, "neutral", None, True, _atomic_prompt(desc)) for k, v, desc in slots]


PLANNERS = {
    "humanoid": humanoid_plan,
    "quadruped": quadruped_plan,
    "bird": bird_plan,
    "vehicle": vehicle_plan,
}


def build_animation_pack_plan(animation_profile: str) -> list[dict]:
    try:
        planner = PLANNERS[animation_profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported animation profile: {animation_profile}") from exc
    return [task.to_dict() for task in planner()]


def validate_atomic_result(result: dict, required_slot: str) -> list[str]:
    """Pure metadata-level guard. Vision/model-level checks can append more failures later."""
    failures: list[str] = []
    if result.get("slot_key") != required_slot:
        failures.append("slot_key_mismatch")
    if result.get("subject_count") not in (None, 1):
        failures.append("multiple_subjects")
    if result.get("panel_count") not in (None, 1):
        failures.append("multiple_panels")
    if result.get("contains_text") is True:
        failures.append("contains_text")
    if result.get("is_contact_sheet") is True:
        failures.append("contact_sheet")
    if result.get("pose_count") not in (None, 1):
        failures.append("multiple_poses")
    return failures


if __name__ == "__main__":
    import json
    print(json.dumps(build_animation_pack_plan("humanoid"), indent=2))
