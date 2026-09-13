# Kid Studio 2D/2.5D Animation Asset Generation Specification

## Purpose

The CPU renderer proof-of-concept has shown that Kid Studio can render Stretchy-compatible animation projects without CUDA/RunPod. The remaining bottleneck is asset preparation.

Kid Studio must **not** attempt to reverse-engineer a production-quality rig from one finished flattened PNG. Instead, the image-generation workflow must create animation-ready assets intentionally.

The central rule is:

> Every required animation view or reusable image must be generated as one independent asset in one generation task.

Never ask the image model for multiple required views, expressions, poses, or panels in one image.

## Core architecture

```text
Approved canonical character
        |
        v
Animation Pack Plan
        |
        +--> front_neutral            (one generation)
        +--> three_quarter_left       (one generation)
        +--> three_quarter_right      (one generation)
        +--> side_left                (one generation)
        +--> side_right               (one generation)
        +--> expression_happy         (one generation)
        +--> expression_sad           (one generation)
        +--> mouth_rest               (one generation)
        +--> mouth_A                  (one generation)
        +--> mouth_E                  (one generation)
        +--> mouth_O                  (one generation)
        +--> etc.
        |
        v
Per-asset validation / approval / replacement
        |
        v
Animation Character Pack
        |
        v
Stretchy rig + reusable animation library
```

## Atomic generation rule

Each required animation asset is a separate production unit with its own:

- slot key
- prompt
- generation job
- file ID
- approval state
- replacement history
- source/canonical lineage
- validation result

A single generated image must never satisfy more than one required animation-pack slot.

Examples of valid slot keys:

```text
front_neutral
three_quarter_left
three_quarter_right
side_left
side_right
expression_happy
expression_sad
expression_surprised
mouth_rest
mouth_A
mouth_E
mouth_O
mouth_MBP
mouth_FV
mouth_L
```

## Prompt contract for every character view

Every prompt for an animation-ready character view should include equivalent constraints to:

```text
Generate exactly one character only.
Generate exactly one requested pose/view only.
Do not create a contact sheet.
Do not create a turnaround sheet.
Do not create multiple panels.
Do not create multiple versions of the character.
Do not include labels or text.
Do not include a collage.
Do not crop any required body part.
Preserve the exact approved character identity from the supplied reference:
- face
- hairstyle
- skin tone
- age
- proportions
- clothing identity
- color palette
- rendering style

Use a transparent background when the provider supports it.
Otherwise use a plain uniform removable background.
Keep the character centered with enough empty margin around the full body.
```

For rig-friendly neutral views add:

```text
Full body visible from head to feet.
Hands clearly visible.
Feet clearly visible.
Arms separated from the torso enough for rigging.
Legs visually separable enough for rigging.
No crossed arms.
No crossed legs.
No handheld props unless the slot explicitly requires them.
Avoid hair, clothing, or accessories obscuring major limb joints where possible.
Use a relaxed neutral pose appropriate for rigging.
```

## Identity conditioning

Every derived animation-pack generation must be conditioned from the approved canonical identity.

Later views may additionally reference already approved animation views when that improves consistency, but the canonical approved character remains the identity authority.

Derived assets must record lineage, for example:

```json
{
  "role": "animation_pack_member",
  "slot_key": "side_left",
  "canonical_asset_id": "...",
  "reference_asset_ids": ["..."],
  "identity_locked": true
}
```

Replacing an animation-pack member must replace that slot for future Director use while preserving version/history. Old rejected/replaced versions must not remain active in Director memory.

## Character Animation Pack

Suggested logical structure:

```text
character_animation_pack
  canonical_reference
  views
    front_neutral
    three_quarter_left
    three_quarter_right
    side_left
    side_right
  expressions
    neutral
    happy
    sad
    surprised
    angry
  mouth_shapes
    rest
    A
    E
    O
    MBP
    FV
    L
  rig
    rig_type
    rig_file_id
    rig_version
  actions
    idle
    walk
    run
    jump
    wave
    point
    clap
    dance_basic
    talk_idle
```

Not every character class needs every slot. The Director should choose the pack template from the asset/character class.

## Asset-class templates

### Humanoid

Required initial pack:

```text
front_neutral
three_quarter_left
three_quarter_right
side_left
side_right
neutral_face
happy_face
sad_face
surprised_face
mouth_rest
mouth_A
mouth_E
mouth_O
mouth_MBP
mouth_FV
mouth_L
```

Rig profile:

```text
humanoid_2d
head
neck
torso
shoulders
upper_arms
forearms
hands
hips
upper_legs
lower_legs
feet
face/mouth controls
```

### Quadruped

Generate atomic independent assets for the required orientations, usually:

```text
side_left_neutral
side_right_neutral
three_quarter_left
three_quarter_right
front_neutral
```

Rig profile includes:

```text
head
neck
spine
front legs
rear legs
tail
ears
mouth
```

### Bird

Possible independent slots:

```text
side_left_perched
side_right_perched
front_perched
side_left_wings_open
side_right_wings_open
```

Rig profile:

```text
body
head
beak
left_wing
right_wing
tail
feet
```

### Vehicle

Independent views:

```text
side_left
side_right
front
rear
three_quarter_front_left
three_quarter_front_right
```

Rig profile may include:

```text
body
wheels
steering
lights
doors
attachment points
```

### Plants

Usually one or more isolated structural views rather than character turnarounds. Rig structure can contain trunk/stem, branches and foliage groups for sway animation.

### Props

Generate each prop as its own isolated asset. Do not generate multiple prop states in one image. Example:

```text
basket_closed
basket_open
```

must be two independent asset slots if both are required.

## Generation planner

The Director should first create a deterministic animation-pack plan instead of directly asking the image model for "all views".

Example:

```json
{
  "pack_type": "humanoid_2d",
  "canonical_asset_id": "...",
  "units": [
    {"slot_key": "front_neutral", "asset_type": "character_animation_view"},
    {"slot_key": "three_quarter_left", "asset_type": "character_animation_view"},
    {"slot_key": "three_quarter_right", "asset_type": "character_animation_view"},
    {"slot_key": "side_left", "asset_type": "character_animation_view"},
    {"slot_key": "side_right", "asset_type": "character_animation_view"}
  ]
}
```

Execute each unit independently.

Do not batch several required views into one image-generation prompt even if the provider offers multiple output images per request. Every required slot needs an independently addressable file/result.

## Validation

An image must not be marked as satisfying its slot merely because generation returned successfully.

The validation stage should reject or flag results containing:

- more than one character instance
- multiple panels
- contact sheets
- turnaround sheets
- split screens
- duplicate poses
- labels/text
- severe cropping when full body is required
- incorrect requested orientation
- obvious identity drift
- wrong wardrobe/state
- obscured required limbs for a rig-neutral slot

Validation result example:

```json
{
  "slot_key": "side_left",
  "valid": false,
  "reasons": ["multiple_character_instances", "contact_sheet_layout"]
}
```

A failed validation must not silently consume the slot as completed.

## UI behavior

Animation Pack should be visible separately from the existing canonical character asset.

Example:

```text
Mutinta
  Canonical character       Approved

  Animation Pack
    Front neutral            Approved   View   Replace
    3/4 left                Approved   View   Replace
    3/4 right               Missing    Create
    Side left               Approved   View   Replace
    Side right              Missing    Create

    Expressions
      Happy                 Approved
      Sad                   Missing

    Mouth Shapes
      Rest                  Approved
      A                     Approved
      E                     Missing
```

Creation progress should be per-slot, for example:

```text
Creating side_right...
Created
Validation passed
```

Never show a single "Created" state for an entire pack when only one composite image was generated.

## Replacement and memory semantics

When a member is replaced:

1. generate a new independent file
2. validate it
3. approve it for that exact slot
4. mark the previous slot version inactive/superseded
5. ensure Director/runtime resolves only the active approved slot version
6. retain old version in history

Deleted/rejected/superseded images must not continue to condition later production unless explicitly selected as historical references.

## Production boundary

Kid Studio production should consume approved Animation Pack slots and an approved Stretchy rig.

It should not attempt arbitrary automatic body-part segmentation from the canonical flattened PNG during shot rendering.

The canonical image remains the visual identity authority. The Animation Pack is a derived, reusable production asset designed specifically for 2D/2.5D animation.

## Current POC status

Already proven on CPU:

```text
approved character PNG
→ transparent working image
→ valid Stretchy-compatible .stretch package
→ CPU frame rendering
→ FFmpeg H.264 MP4
```

The current flat single-part render is only a pipeline proof. It is not the target character-animation quality.

Next engineering milestone:

> Implement atomic Animation Pack planning/generation/validation, then build one proper Mutinta Stretchy rig from approved independent animation-ready assets.
