from animation_pack_planner import (
    build_animation_pack_plan,
    validate_atomic_result,
)


def test_humanoid_plan_is_atomic_and_unique():
    plan = build_animation_pack_plan("humanoid")

    assert plan
    slot_keys = [item["slot_key"] for item in plan]
    assert len(slot_keys) == len(set(slot_keys))

    expected_required = {
        "front_neutral",
        "three_quarter_left",
        "three_quarter_right",
        "side_left",
        "side_right",
        "expression_happy",
        "mouth_rest",
        "mouth_A",
        "mouth_E",
        "mouth_O",
        "mouth_MBP",
    }
    required = {item["slot_key"] for item in plan if item["required"]}
    assert expected_required.issubset(required)

    for item in plan:
        prompt = item["prompt_suffix"].lower()
        assert "exactly one subject" in prompt
        assert "exactly one requested view or pose" in prompt
        assert "contact sheet" in prompt
        assert "collage" in prompt
        assert "do not duplicate the character" in prompt
        assert "do not add text" in prompt


def test_non_humanoid_plans_have_independent_slots():
    for profile in ("quadruped", "bird", "vehicle"):
        plan = build_animation_pack_plan(profile)
        assert plan
        keys = [item["slot_key"] for item in plan]
        assert len(keys) == len(set(keys))
        assert all(item["required"] for item in plan)


def test_atomic_result_validator_accepts_single_asset():
    failures = validate_atomic_result(
        {
            "slot_key": "front_neutral",
            "subject_count": 1,
            "panel_count": 1,
            "pose_count": 1,
            "contains_text": False,
            "is_contact_sheet": False,
        },
        "front_neutral",
    )
    assert failures == []


def test_atomic_result_validator_rejects_composite_output():
    failures = validate_atomic_result(
        {
            "slot_key": "wrong_slot",
            "subject_count": 3,
            "panel_count": 4,
            "pose_count": 4,
            "contains_text": True,
            "is_contact_sheet": True,
        },
        "front_neutral",
    )

    assert set(failures) == {
        "slot_key_mismatch",
        "multiple_subjects",
        "multiple_panels",
        "contains_text",
        "contact_sheet",
        "multiple_poses",
    }


def test_unknown_profile_fails_cleanly():
    try:
        build_animation_pack_plan("unknown")
    except ValueError as exc:
        assert "Unsupported animation profile" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported profile")


if __name__ == "__main__":
    test_humanoid_plan_is_atomic_and_unique()
    test_non_humanoid_plans_have_independent_slots()
    test_atomic_result_validator_accepts_single_asset()
    test_atomic_result_validator_rejects_composite_output()
    test_unknown_profile_fails_cleanly()
    print("animation pack planner tests: OK")
