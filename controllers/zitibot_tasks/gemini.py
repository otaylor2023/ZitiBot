"""Gemini + RealSense detection helpers keyed by Object."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import math
import numpy as np

from vision import gemini_pointing as gp

from zitibot_core import arm
from zitibot_core.constants import (
    EE_ORI_TOOL_DOWN,
    EGG_CRACKER_CRADLE_CENTER_WORLD_OFFSET_M,
    OBJECT_DEFAULTS,
    PAN_STATION_GRASP_EE_ORIENTATION,
    T_FLANGE_CAMERA,
    Object,
)
from zitibot_core.context import TaskContext
from zitibot_core.runner import read_stdin_line


# Number of CONSECUTIVE Gemini timeouts that ``handle_gemini_timeout``
# auto-retries (silently / without operator input) before falling back
# to :func:`prompt_after_gemini_timeout`. Picked so transient API
# stalls don't interrupt unattended runs (Gemini Robotics-ER can
# occasionally take >8 s) but a sustained outage still surfaces to
# the operator. Set to 0 to disable auto-retry and prompt on every
# timeout (the original behaviour).
GEMINI_TIMEOUT_AUTORETRY_LIMIT: int = 3


def prompt_after_gemini_timeout(obj_name: str, kind: str = "grasp_pose") -> None:
    """Block on stdin until the operator says retry (ENTER) or quit (``q``).

    Low-level prompt used by :func:`handle_gemini_timeout` after the
    auto-retry budget is exhausted. Public so any controller that
    calls Gemini directly can fall through to it without re-implementing
    the prompt UI.

    Raises :class:`KeyboardInterrupt` if the operator types ``q`` or
    if stdin is closed.
    """
    print(
        f"[gemini:{obj_name}/{kind}] no response from Gemini within timeout — stopping.",
        flush=True,
    )
    print(
        "  Press ENTER to ask Gemini for a new frame, or type q then ENTER to abort.",
        flush=True,
    )
    line = read_stdin_line()
    if line is None:
        raise KeyboardInterrupt("stdin closed")
    token = line.strip().lower()
    if token in ("q", "quit", "exit"):
        raise KeyboardInterrupt("quit requested")


def handle_gemini_timeout(obj_name: str, kind: str, timeout_count: int) -> None:
    """Auto-retry the first few consecutive timeouts, then prompt the operator.

    Standard handler used by every Gemini-using detection loop —
    callers track a running ``timeout_count`` (1-indexed, reset on
    any non-timeout outcome) and pass it in on each call. Behaviour:

    * ``timeout_count <= GEMINI_TIMEOUT_AUTORETRY_LIMIT``: prints an
      info line and returns immediately so the caller can retry
      without operator interaction.
    * ``timeout_count >  GEMINI_TIMEOUT_AUTORETRY_LIMIT``: falls
      through to :func:`prompt_after_gemini_timeout`, blocking on
      stdin until ENTER (retry) or ``q`` (raises ``KeyboardInterrupt``).

    Distinct from the regular ``retries`` budget: timeouts never
    abort on their own — they either auto-retry or prompt.
    """
    if timeout_count <= GEMINI_TIMEOUT_AUTORETRY_LIMIT:
        print(
            f"[gemini:{obj_name}/{kind}] auto-retry "
            f"{timeout_count}/{GEMINI_TIMEOUT_AUTORETRY_LIMIT} after timeout — "
            "requesting a fresh frame.",
            flush=True,
        )
        return
    prompt_after_gemini_timeout(obj_name, kind)

# Camera optical → flange (meters) now lives in zitibot_core.constants as the
# single source of truth (imported above as T_FLANGE_CAMERA).

# Extra yaw correction (tool +Z) applied to the orientation from
# ``_apply_perpendicular_yaw``. The measured point geometry currently favors
# the +45° camera extrinsic for position; this orientation-only correction keeps
# the gripper closing direction aligned while we wait on a hand-eye calibration.
GRASP_AXIS_YAW_OFFSET_DEG = 45.0

_GEMINI_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
# Default annotated grasp images when ``ctx.gemini_response_path`` is unset.
GEMINI_GRASP_RESPONSE_PATHS: dict[Object, Path] = {
    Object.PAN: _GEMINI_LOGS_DIR / "gemini_response_pan.png",
    Object.MIXING_BOWL: _GEMINI_LOGS_DIR / "gemini_response_mixing_bowl_grasp.png",
    Object.WHISK: _GEMINI_LOGS_DIR / "gemini_response_whisk_grasp.png",
}


def resolve_grasp_response_path(
    ctx: TaskContext,
    obj: Object,
    override: str | Path | None = None,
) -> Path | None:
    """Pick where to save the annotated grasp overlay for ``obj``."""
    if override is not None:
        return Path(override).expanduser()
    if ctx.gemini_response_path is not None:
        return Path(ctx.gemini_response_path).expanduser()
    return GEMINI_GRASP_RESPONSE_PATHS.get(obj)


@dataclass
class _Detection:
    """Per-(Object, kind) Gemini detection config: prompt + point selector.

    Grasp orientation/position logic is NOT configured here — that lives in
    per-object grasp pose builders (see ``_GRASP_POSE_BUILDERS``). This
    dataclass only describes how to query Gemini and which point(s) to use
    for simple single-point detections like ``pour`` / ``center`` / ``handle``.
    """

    prompt: str | Callable[[Object], str] | None = None
    select: str = "first"  # first | midpoint | nearest_camera
    world_offset_m: np.ndarray | None = None
    # Depth-sampling policy for the grasp query. True (DEFAULT) uses Gemini's
    # exact pixel when it has valid depth, else snaps to the NEAREST shallowest
    # valid pixel and deprojects from there — right when Gemini's point IS the
    # intended grasp spot (egg, egg-cracker handles, tongs tape). False keeps
    # the legacy "shallowest pixel anywhere in the patch, deproject at Gemini's
    # XY" rule, which the bowl/cylinder RIM grasps rely on (Gemini's point sits
    # inside the rim, so we deliberately reach for the highest point nearby).
    # Those objects set this False explicitly in ``_PROMPTS``.
    prefer_gemini_pixel_depth: bool = True
    # Minimum depth (m) a pixel must report to be usable for THIS object's
    # grasp lift. 0 (default) keeps the legacy "any depth > 0" rule. >0 makes
    # the depth helpers ignore everything closer than the floor (see
    # ``gp.lift_points_to_3d``) — the egg uses this to drop spurious near-field
    # returns off the shiny shell / the gripper's own fingers.
    min_depth_m: float = 0.0
    # Per-object override of the grasp depth-patch radius (pixels). ``None``
    # (default) uses the call-wide ``GRASP_DEPTH_PATCH_RADIUS``. Compact
    # objects whose grasp point IS Gemini's point (e.g. the egg) shrink this
    # so depth is sampled tightly around Gemini's pixel instead of over the
    # wide rim-finding patch.
    depth_patch_radius: int | None = None

    def resolve_prompt(self, obj: Object) -> str:
        if callable(self.prompt):
            return self.prompt(obj)
        if self.prompt:
            return self.prompt
        return gp.build_prompt(obj.value, None)

    def offset(self, obj: Object) -> np.ndarray:
        if self.world_offset_m is not None:
            return np.asarray(self.world_offset_m, dtype=np.float64).reshape(3)
        spec = OBJECT_DEFAULTS.get(obj)
        if spec is not None:
            return spec.gemini_world_offset_m.copy()
        return np.zeros(3, dtype=np.float64)


@dataclass
class GraspPose:
    position: np.ndarray
    orientation: np.ndarray
    rim_yaw_deg: float | None = None
    rim_yaw_applied: bool = False
    # Gemini patch pixel coordinates ``(u, v)`` used for depth lift; seeds LK servo.
    source_pixels: list[tuple[int, int]] | None = None
    # Gemini bounding boxes in pixels ``(x0, y0, x1, y1)`` when the detection
    # used ``box_2d`` entries; seeds template-matching servo. ``None`` for
    # point-only detections.
    source_boxes: list[tuple[int, int, int, int]] | None = None


def _source_pixels_from_patches(
    patches: list[dict | None] | None,
) -> list[tuple[int, int]] | None:
    """Extract Gemini ``(u, v)`` seed pixels from depth-patch debug dicts."""
    if not patches:
        return None
    pixels: list[tuple[int, int]] = []
    for p in patches:
        if p is None:
            continue
        if "u" in p and "v" in p:
            pixels.append((int(p["u"]), int(p["v"])))
    return pixels if pixels else None


def _source_boxes_from_patches(
    patches: list[dict | None] | None,
) -> list[tuple[int, int, int, int]] | None:
    """Extract Gemini bounding boxes ``(x0, y0, x1, y1)`` from patch dicts."""
    if not patches:
        return None
    boxes: list[tuple[int, int, int, int]] = []
    for p in patches:
        if p is None:
            continue
        box = p.get("box")
        if box is not None and len(box) == 4:
            boxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
    return boxes if boxes else None


def _first_valid(points: list[tuple[float, float, float] | None]) -> tuple[float, float, float]:
    for p in points:
        if p is not None:
            return p
    raise ValueError("no valid 3D points from Gemini/depth")


def _midpoint_valid(points: list[tuple[float, float, float] | None]) -> tuple[float, float, float]:
    valid = [p for p in points if p is not None]
    if len(valid) < 1:
        raise ValueError("no valid 3D points from Gemini/depth")
    if len(valid) == 1:
        return valid[0]
    a = np.array(valid[0], dtype=np.float64)
    b = np.array(valid[1], dtype=np.float64)
    mid = (a + b) / 2.0
    return float(mid[0]), float(mid[1]), float(mid[2])


_SELECTORS = {
    "first": _first_valid,
    "midpoint": _midpoint_valid,
    "nearest_camera": _first_valid,
}


def _object_prompt_name(o: Object) -> str:
    if o is Object.BOTTLE:
        return "parmesan cheese bottle"
    if o is Object.PASTA_BOWL:
        return "black pasta bowl"
    if o is Object.MIXING_BOWL:
        return "blue mixing bowl"
    if o is Object.PLASTIC_BOWL_TOP:
        # Two plastic bowls are stacked / arranged in the camera frame;
        # this one is the bowl whose centroid sits **higher up** in the
        # image (smaller normalized y in [y, x] order = top of frame).
        return (
            "plastic bowl that is positioned HIGHER UP in the camera "
            "frame (the bowl whose centroid is closer to the top of "
            "the image, i.e. smaller normalized y)"
        )
    if o is Object.PLASTIC_BOWL_BOTTOM:
        # The bowl whose centroid sits **lower down** in the image
        # (larger normalized y = bottom of frame).
        return (
            "plastic bowl that is positioned LOWER DOWN in the camera "
            "frame (the bowl whose centroid is closer to the bottom of "
            "the image, i.e. larger normalized y)"
        )
    return o.value


def _bowl_left_rim_prompt(o: Object) -> str:
    name = _object_prompt_name(o)
    return (
        f"In the image, locate the **{name}**. "
        f"Pick TWO distinct points on the OUTER RIM/LIP of the bowl, BOTH "
        f"on the LEFT SIDE of the bowl as seen in the image:\n"
        f"  - Both points must lie on the visible outer rim (top edge of "
        f"the bowl wall), not inside the bowl and not on the bowl base.\n"
        f"  - Both points must be on the LEFT half of the bowl: smaller "
        f"normalized x (x grows rightward). Roughly at 9 o'clock as seen "
        f"on a clock face overlaid on the bowl.\n"
        f"  - The two points should be a few centimetres apart along the "
        f"rim arc — far enough that the line between them clearly defines "
        f"the rim's tangent direction at the left side.\n"
        f"  - Do NOT place both points at the same location, and do NOT "
        f"put one on the near/far side.\n"
        f"\n"
        f"The midpoint of the two points is the grasp target; the segment "
        f"between them defines the rim tangent, and the gripper will close "
        f"perpendicular to that tangent (one finger inside the bowl, one "
        f"outside).\n"
        f"\n"
        f"Reply with JSON only:\n"
        f'[{{"point": [y, x], "label": "{o.value}_left_rim_1"}}, '
        f'{{"point": [y, x], "label": "{o.value}_left_rim_2"}}]\n'
        f"Coordinates must be normalized 0-1000 in [y, x] order."
    )


def _mixing_bowl_grasp_prompt(o: Object) -> str:
    """Mixing bowl: upper + lower left-rim anchors; grasp is computed on-arc in code."""
    name = _object_prompt_name(o)
    return (
        f"In the image, locate the **{name}**. "
        f"Return TWO points on the OUTER RIM/LIP on the LEFT SIDE of the bowl "
        f"(smaller normalized x):\n"
        f"  1. **{o.value}_left_rim_1** — upper part of the left-side rim arc "
        f"(the narrow top edge of the bowl wall itself, not inside the bowl).\n"
        f"  2. **{o.value}_left_rim_2** — lower part of the same left-side rim "
        f"arc, a few centimetres along the rim from point 1.\n"
        f"\n"
        f"IMPORTANT — the bowl interior is shiny/reflective and may look brighter "
        f"or have strong highlights. Do NOT place points on any bright reflection, "
        f"highlight, or on the bowl floor/interior surface. Both points must sit "
        f"on the physical outer wall of the bowl — the narrow raised lip/edge "
        f"visible at the very top of the left side — NOT inside. When in doubt, "
        f"place the point slightly toward the outside (leftward in the image) "
        f"rather than inside.\n"
        f"\n"
        f"Do NOT place points on the table, inside the bowl, or on the right side. "
        f"The line between them runs along the left rim; the robot grasps on "
        f"the curved rim between them (computed separately — do NOT add a third "
        f"point).\n"
        f"\n"
        f"Reply with JSON only:\n"
        f'[{{"point": [y, x], "label": "{o.value}_left_rim_1"}}, '
        f'{{"point": [y, x], "label": "{o.value}_left_rim_2"}}]\n'
        f"Coordinates must be normalized 0-1000 in [y, x] order."
    )


def _left_rim_arc_grasp_pixel(u1: int, v1: int, u2: int, v2: int) -> tuple[int, int]:
    """Pixel on the outer rim between two left-side rim anchors.

    The straight-line midpoint between rim anchors sits inside the bowl.
    Push outward (image -u, away from the estimated bowl center) so the
    grasp lands on the curved lip — the leftmost bulge of that arc.
    """
    u_mid = (u1 + u2) * 0.5
    v_mid = (v1 + v2) * 0.5
    dv = float(abs(v2 - v1))
    du = float(abs(u2 - u1))
    rim_span = max(dv, du, 1.0)
    u_center = u_mid + 0.85 * rim_span
    v_center = v_mid
    ox = u_mid - u_center
    oy = v_mid - v_center
    o_norm = math.hypot(ox, oy)
    if o_norm < 1e-6:
        return int(round(u_mid)), int(round(v_mid))
    push = 0.40 * dv
    u_grasp = u_mid + ox / o_norm * push
    v_grasp = v_mid + oy / o_norm * push
    return int(round(u_grasp)), int(round(v_grasp))


def _lift_pixel_to_cam(
    u: int,
    v: int,
    depth_m: np.ndarray,
    intrinsics,
    depth_patch_radius: int,
    depth_quantile: float,
) -> tuple[float, float, float] | None:
    sample = gp.find_quantile_pixel(
        depth_m, u, v, depth_patch_radius, quantile=depth_quantile,
    )
    if sample is None:
        return None
    _u_s, _v_s, d = sample
    xyz = gp.deproject_pixel_to_cam(intrinsics, u, v, d)
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


def _mixing_bowl_arc_grasp_cam(
    patches: list[dict | None],
    depth_m: np.ndarray,
    intrinsics,
    depth_patch_radius: int,
    depth_quantile: float,
) -> tuple[float, float, float] | None:
    if len(patches) < 2 or patches[0] is None or patches[1] is None:
        return None
    u1, v1 = int(patches[0]["u"]), int(patches[0]["v"])
    u2, v2 = int(patches[1]["u"]), int(patches[1]["v"])
    u_grasp, v_grasp = _left_rim_arc_grasp_pixel(u1, v1, u2, v2)
    grasp_cam = _lift_pixel_to_cam(
        u_grasp, v_grasp, depth_m, intrinsics, depth_patch_radius, depth_quantile,
    )
    if grasp_cam is not None:
        print(
            f"[gemini:mixing_bowl/arc-grasp] anchors px=({u1},{v1}),({u2},{v2}) "
            f"-> rim px=({u_grasp},{v_grasp}) cam={grasp_cam}"
        )
    return grasp_cam


def _plastic_bowl_left_edge_prompt(o: Object) -> str:
    """Rectangular plastic bowl: two points along the LEFT EDGE, not corners.

    The plastic bowls are roughly rectangular trays. The "clock face / left
    rim arc" wording used for round bowls (``_bowl_left_rim_prompt``) makes
    Gemini drift onto the top-left or bottom-left corner, which puts one
    finger over the corner instead of squarely across the wall. This prompt
    forces both points onto the straight left edge, centered along it, so
    the chord runs parallel to the wall and the perpendicular gripper close
    lands one finger inside the bowl and one outside.
    """
    name = _object_prompt_name(o)
    return (
        f"In the image, locate the **{name}**. "
        f"This bowl is RECTANGULAR (not round) — like a shallow rectangular "
        f"tray. Pick TWO distinct points on its LEFT EDGE (the long straight "
        f"side of the rectangle that sits at the smallest normalized x in "
        f"the image):\n"
        f"  - Both points must lie on the OUTER TOP LIP of the LEFT WALL — "
        f"the straight top edge of the wall on the left side. Not inside "
        f"the bowl, not on the bowl floor/base, not on the front, back, "
        f"or right walls.\n"
        f"  - Both points must sit in the MIDDLE STRETCH of the left edge, "
        f"clearly AWAY FROM THE CORNERS. Do NOT pick the top-left or "
        f"bottom-left corner; aim for the central portion of the edge so "
        f"the line between the two points runs straight along the wall.\n"
        f"  - The two points should be a few centimetres apart along the "
        f"left edge — far enough that the line between them clearly "
        f"defines the edge's direction (roughly top-to-bottom in the "
        f"image), not its thickness.\n"
        f"  - Do NOT place both points at the same location, and do NOT "
        f"put one on a different wall.\n"
        f"\n"
        f"The midpoint of the two points is the grasp target; the segment "
        f"between them defines the left-edge direction, and the gripper "
        f"will close PERPENDICULAR to that edge (one finger inside the "
        f"bowl, one outside).\n"
        f"\n"
        f"Reply with JSON only:\n"
        f'[{{"point": [y, x], "label": "{o.value}_left_edge_1"}}, '
        f'{{"point": [y, x], "label": "{o.value}_left_edge_2"}}]\n'
        f"Coordinates must be normalized 0-1000 in [y, x] order."
    )


def _egg_cracker_strip_prompt(o: Object) -> str:
    """Egg cracker: one bounding box around each handle.

    The egg cracker has two handle arms. Gemini returns one bounding box
    per handle; each box CENTER becomes the grasp point (so the box must be
    centered on the blue grasp mark). The midpoint between the two centers
    is the grasp target and the line between them defines the approach angle
    (the gripper closes ALONG that line, squeezing the two handles together).
    The box crops also seed the template-matching visual servo, so each box
    should include the red tape around the blue mark for trackable texture.
    Each handle is wrapped with red tape with a blue mark at the grasp point.
    """
    return (
        f"There are TWO similar two-armed kitchen tools in the scene: a pair "
        f"of TONGS and an EGG CRACKER. Both may have red and blue tape, so do "
        f"NOT use tape colors alone to decide which tool is which. The EGG "
        f"CRACKER is the tool with a round CRADLE / ring that a single egg "
        f"sits inside to be cracked; the TONGS are the plain gripping tool "
        f"with no egg cradle. You must find the EGG CRACKER and grasp it — "
        f"IGNORE the tongs.\n"
        f"\n"
        f"The egg cracker has two handle arms, each wrapped with RED tape, "
        f"with a small BLUE mark at the grasp point.\n"
        f"For EACH handle (one on the LEFT, one on the RIGHT), return a "
        f"bounding box around the ENTIRE RED-TAPED part of that handle. The "
        f"box must span the full length and width of the red tape on the "
        f"handle, NOT just the blue mark. Include all of the red tape.\n"
        f"\n"
        f"Reply with JSON only:\n"
        f'[{{"box_2d": [ymin, xmin, ymax, xmax], "label": "{o.value}_handle_1"}}, '
        f'{{"box_2d": [ymin, xmin, ymax, xmax], "label": "{o.value}_handle_2"}}]\n'
        f"Coordinates must be normalized 0-1000."
    )


def _egg_cracker_blue_prompt(o: Object) -> str:
    """Egg cracker: one point on each handle's BLUE grasp mark.

    The egg cracker has two handle arms, each wrapped with RED tape with a
    small BLUE mark at the grasp point. This prompt targets the blue marks
    directly (one point per handle). The midpoint of the two points is the
    grasp target and the line between them defines the approach angle — the
    gripper closes ALONG that line, squeezing the two handles together
    (see ``_build_pose_egg_cracker``).
    """
    return (
        f"There are TWO similar two-armed kitchen tools in the scene: a pair "
        f"of TONGS and an EGG CRACKER. Both may have red and blue tape, so do "
        f"NOT use tape colors alone to decide which tool is which. The EGG "
        f"CRACKER is the tool with a round CRADLE / ring that a single egg "
        f"sits inside to be cracked; the TONGS are the plain gripping tool "
        f"with no egg cradle. You must find the EGG CRACKER and grasp it — "
        f"IGNORE the tongs.\n"
        f"\n"
        f"The egg cracker has two handle arms, each wrapped with RED tape, "
        f"with a small BLUE mark at the grasp point.\n"
        f"For EACH egg-cracker handle (one on the LEFT, one on the RIGHT), "
        f"return ONE point centered on that handle's BLUE mark. Both points "
        f"must land on visibly BLUE pixels — not on the surrounding red tape, "
        f"not on the bare handle, not on the tongs' blue tape, and not on the "
        f"table.\n"
        f"\n"
        f"Reply with JSON only:\n"
        f'[{{"point": [y, x], "label": "{o.value}_blue_1"}}, '
        f'{{"point": [y, x], "label": "{o.value}_blue_2"}}]\n'
        f"Coordinates must be normalized 0-1000 in [y, x] order."
    )


# Minimum depth (m) for ladle grasp: reject foreground gripper / wrist returns
# (~5–7 cm) while keeping valid counter-top ladle hits (~10–15 cm+).
LADLE_MIN_DEPTH_M = 0.08


# Minimum depth (m) for whisk grasp: reject foreground gripper returns while
# keeping valid counter-top hits (~10–15 cm+).
WHISK_MIN_DEPTH_M = 0.08


def _whisk_blue_plaster_prompt(o: Object) -> str:
    """Whisk: one point on the blue plaster/tape patch (pick-up grasp location)."""
    return (
        f"In the image, locate a **kitchen whisk** resting on the table/counter. "
        f"The whisk handle has a patch of **BLUE plaster** (or blue tape/marker) "
        f"— that blue patch is the ONLY valid grasp target for this step. "
        f"Do NOT pick the red tape, the wire loops/ball, the table, or the robot "
        f"gripper.\n"
        f"\n"
        f"Return ONE point at the **center** of the blue plaster/tape patch. "
        f"The point must sit on visibly BLUE pixels.\n"
        f"\n"
        f"Reply with JSON only:\n"
        f'[{{"point": [y, x], "label": "{o.value}_blue_plaster"}}]\n'
        f"Coordinates must be normalized 0-1000 in [y, x] order."
    )


def _whisk_red_tape_prompt(o: Object) -> str:
    """Whisk: two points on the top portion of the red tape, at opposite ends
    along the tape's long edge (handle axis). Midpoint = grasp target.
    """
    return (
        f"In the image, locate a **kitchen whisk** (it may be sitting in a bowl "
        f"on the table/counter). The whisk has a metal handle with a short strip "
        f"of **RED tape** wrapped around it — that red tape is the ONLY valid "
        f"grasp target. Do NOT pick the wire loops/ball at the far end, the "
        f"table/bowl rim, or the robot gripper.\n"
        f"\n"
        f"Return TWO distinct points on the **top portion** of the red tape strip "
        f"— the upper half / top long edge of the tape patch (the side of the "
        f"tape facing upward toward the camera), NOT the bottom edge of the tape "
        f"and NOT on bare handle metal below the tape. Place the points at "
        f"opposite ends along the tape's **longer side** (lengthwise along the "
        f"handle axis, NOT across the short width). Both points must sit on "
        f"visibly RED tape pixels on that top portion.\n"
        f"\n"
        f"The midpoint of the two points is the grasp target (it must land on the "
        f"top portion of the red tape); the line between them defines the handle "
        f"axis (the gripper closes perpendicular to this axis, wrapping around "
        f"the handle).\n"
        f"\n"
        f"Reply with JSON only:\n"
        f'[{{"point": [y, x], "label": "{o.value}_tape_1"}}, '
        f'{{"point": [y, x], "label": "{o.value}_tape_2"}}]\n'
        f"Coordinates must be normalized 0-1000 in [y, x] order."
    )


def _tongs_red_tape_prompt(o: Object) -> str:
    """Tongs: one point on each of the two RED tape marks (one per arm).

    The tongs are a two-armed kitchen tool; each arm has a piece of red tape
    near the gripping end. This prompt targets the two red marks directly (one
    point per arm). The midpoint of the two points is the grasp target and the
    line between them defines the approach angle — the gripper closes ALONG
    that line, one finger per arm (same geometry as the egg cracker; see
    ``_build_pose_egg_cracker``).
    """
    return (
        f"There are TWO similar two-armed kitchen tools in the scene: a pair "
        f"of TONGS and an EGG CRACKER. Both may have red and blue tape, so do "
        f"NOT use tape colors alone to decide which tool is which. The EGG "
        f"CRACKER is the tool with a round CRADLE / ring that a single egg "
        f"sits inside to be cracked; the TONGS are the plain two-armed "
        f"gripping tool with no egg cradle. You must find the **TONGS** and "
        f"grasp them — IGNORE the egg cracker.\n"
        f"\n"
        f"The tongs are a two-armed kitchen tool; each of the two arms has a "
        f"piece of RED tape wrapped around it.\n"
        f"For EACH tong arm (one on the LEFT, one on the RIGHT), return ONE "
        f"point centered on that arm's RED tape. Both points must land on "
        f"visibly RED tape pixels — not on the bare metal/plastic arm, not on "
        f"the hinge, not on any tape attached to the egg cracker, and not on "
        f"the table.\n"
        f"\n"
        f"Reply with JSON only:\n"
        f'[{{"point": [y, x], "label": "{o.value}_red_1"}}, '
        f'{{"point": [y, x], "label": "{o.value}_red_2"}}]\n'
        f"Coordinates must be normalized 0-1000 in [y, x] order."
    )


# Egg-specific (shrunk) depth-patch radius. The egg's grasp point IS Gemini's
# point (no rim hunting), so the wide GRASP_DEPTH_PATCH_RADIUS (50 px) only
# risks pulling depth off the table or a neighbouring object. 12 px (~2.5 cm in
# world at ~25 cm range) keeps the depth sample tight around Gemini's pixel.
EGG_DEPTH_PATCH_RADIUS = 12

# Minimum depth (m) accepted for the egg grasp lift: ignore any pixel closer
# than 20 cm so near-field noise off the shell / the gripper's own fingers
# can't pull the grasp Z toward the camera.
EGG_MIN_DEPTH_M = 0.20


def _egg_prompt(o: Object) -> str:
    return (
        f"In the image, locate the **egg(s)** (whole chicken eggs resting on the "
        f"surface). There may be ONE or TWO eggs visible. If there are two, pick "
        f"the CLOSEST one — the egg LOWEST in the image (largest y, nearest the "
        f"bottom edge). Choose ONE point at the center of the top of that egg — "
        f"the highest, most central spot where a gripper coming straight down "
        f"from above should grasp it.\n"
        f"Reply with JSON only, a single point for the chosen egg:\n"
        f'[{{"point": [y, x], "label": "{o.value}"}}]\n'
        f"Coordinates must be normalized 0-1000 in [y, x] order."
    )


def _cylinder_long_axis_prompt(o: Object) -> str:
    object_name = _object_prompt_name(o)
    return (
        f"In the image, locate the **{object_name}**. "
        f"If there are multiple cylindrical containers, choose only the "
        f"{object_name}; do not choose another bottle, jar, or cylinder. "
        f"Draw the visible LONG AXIS of the cylindrical container by choosing "
        f"TWO distinct points on the cylinder body:\n"
        f"  - Both points must lie on the visible cylindrical side surface, "
        f"not on the cap, lid, top, bottom, label edge, or background.\n"
        f"  - The line from point 1 to point 2 must run ALONG the longer axis "
        f"of the cylinder as it appears in the image, not across its width.\n"
        f"  - Put the points a few centimetres apart, roughly centered on the "
        f"body, so their midpoint is a good grasp target.\n"
        f"\n"
        f"The midpoint of the two points is the grasp target. The grasp target "
        f"must stay ON this line, and the gripper will close perpendicular to "
        f"the line between the points.\n"
        f"\n"
        f"Reply with JSON only:\n"
        f'[{{"point": [y, x], "label": "{o.value}_axis_1"}}, '
        f'{{"point": [y, x], "label": "{o.value}_axis_2"}}]\n'
        f"Coordinates must be normalized 0-1000 in [y, x] order."
    )


_PROMPTS: dict[tuple[Object, str], _Detection] = {
    (Object.PASTA_BOWL, "grasp"): _Detection(
        prompt=_bowl_left_rim_prompt,
        select="midpoint",
        # Rim grasp: keep the legacy shallowest-in-patch depth rule.
        prefer_gemini_pixel_depth=False,
    ),
    (Object.PASTA_BOWL, "center"): _Detection(
        prompt=lambda o: (
            "In the image, locate the **center** of the **black pasta bowl** "
            "interior — the dark/black bowl where egg contents should land. "
            "Do NOT pick the white plastic bowl, the table outside the bowl, "
            "or the bowl rim.\n"
            "Pick the geometric CENTER of the bowl's circular opening even if "
            "there are objects/contents inside it (eggs, shells, food, etc.) — "
            "report the center point of the BOWL itself, not of anything sitting "
            "inside it. If the exact center is occluded, still report where the "
            "bowl's center is.\n"
            "Reply with JSON only:\n"
            f'[{{"point": [y, x], "label": "{o.value}_center"}}]\n'
            "Coordinates must be normalized 0-1000 in [y, x] order."
        ),
        select="first",
        world_offset_m=np.zeros(3, dtype=np.float64),
    ),
    (Object.PLASTIC_BOWL_TOP, "grasp"): _Detection(
        prompt=_plastic_bowl_left_edge_prompt,
        select="midpoint",
        prefer_gemini_pixel_depth=False,
    ),
    (Object.PLASTIC_BOWL_TOP, "center"): _Detection(
        prompt=lambda o: (
            "In the image, locate the **center** of the **white plastic bowl** "
            "interior — the light-colored/white bowl used for shells and scraps. "
            "Do NOT pick the black pasta bowl, the table outside the bowl, "
            "or the bowl rim.\n"
            "Pick the geometric CENTER of the bowl's circular opening even if "
            "there are objects/contents inside it (shells, scraps, food, etc.) — "
            "report the center point of the BOWL itself, not of anything sitting "
            "inside it. If the exact center is occluded, still report where the "
            "bowl's center is.\n"
            "Reply with JSON only:\n"
            f'[{{"point": [y, x], "label": "{o.value}_center"}}]\n'
            "Coordinates must be normalized 0-1000 in [y, x] order."
        ),
        select="first",
        world_offset_m=np.zeros(3, dtype=np.float64),
    ),
    (Object.PLASTIC_BOWL_BOTTOM, "grasp"): _Detection(
        prompt=_plastic_bowl_left_edge_prompt,
        select="midpoint",
        prefer_gemini_pixel_depth=False,
    ),
    (Object.MIXING_BOWL, "grasp"): _Detection(
        prompt=_mixing_bowl_grasp_prompt,
        select="first",
        prefer_gemini_pixel_depth=False,
    ),
    (Object.MIXING_BOWL, "pour"): _Detection(
        prompt=lambda o: (
            f"In the image, locate the **{o.value}**. "
            f"Choose ONE point on the near rim where liquid should be poured. "
            f"Reply with JSON only:\n"
            f'[{{"point": [y, x], "label": "{o.value}_pour"}}]\n'
            f"Coordinates must be normalized 0-1000 in [y, x] order."
        ),
        select="first",
    ),
    (Object.MIXING_BOWL, "center"): _Detection(
        prompt=lambda o: (
            f"In the image, locate the **center** of the **{o.value}** interior. "
            f"Reply with JSON only:\n"
            f'[{{"point": [y, x], "label": "{o.value}_center"}}]\n'
            f"Coordinates must be normalized 0-1000 in [y, x] order."
        ),
        select="first",
    ),
    (Object.PAN, "center"): _Detection(
        prompt=lambda o: (
            f"In the image, locate the **{o.value}** (frying/cooking pan). "
            f"Choose ONE point at the **center of the pan's flat cooking surface** "
            f"(the interior floor where food sits), not on the rim, not on the "
            f"handle, and not on the stove. "
            f"Reply with JSON only:\n"
            f'[{{"point": [y, x], "label": "{o.value}_center"}}]\n'
            f"Coordinates must be normalized 0-1000 in [y, x] order."
        ),
        select="first",
        # Return the raw pan-floor center in world frame. (The PAN spec's
        # ``gemini_world_offset_m`` is tuned for the HANDLE grasp, not the
        # cooking-surface center, so don't inherit it here.) Callers add
        # their own work-height offset above this point.
        world_offset_m=np.zeros(3, dtype=np.float64),
    ),
    (Object.PAN, "center_rim"): _Detection(
        prompt=lambda o: (
            f"In the image, locate the **{o.value}** (round metal frying/cooking "
            f"pan). Find the FULL pan: it is the big circular METAL pan body — "
            f"the dark/grey metal disc with a raised circular metal rim/lip "
            f"around its outer edge. The pan usually has a handle sticking out "
            f"to one side; the round metal part is the pan.\n"
            f"CRITICAL: the pan may be PARTLY FILLED with food (a yellow/orange "
            f"egg puddle, batter, or other ingredients) sitting in the middle. "
            f"That food puddle is SMALLER than the pan and is NOT the pan. IGNORE "
            f"the food: do NOT treat the edge of the egg/food as the rim, and do "
            f"NOT put either point on or just around the food. Use the OUTER "
            f"METAL circle of the pan itself, which extends well BEYOND the food "
            f"to the actual metal lip.\n"
            f"Return EXACTLY TWO points as a JSON array, in this EXACT order. "
            f"The two points are DIFFERENT and must be FAR apart:\n"
            f"  1. CENTER — the geometric dead center of the whole round METAL "
            f"pan (the bullseye, equidistant from the OUTER METAL rim on all "
            f"sides). Estimate the center of the full metal circle even if food "
            f"covers part of it.\n"
            f"  2. RIM — a point right ON the pan's OUTER METAL rim: the raised "
            f"circular lip at the outermost boundary of the metal pan, where the "
            f"metal curves up into the side wall. Put it AS FAR FROM THE CENTER "
            f"AS POSSIBLE while still on the metal pan rim (NOT on the food, NOT "
            f"partway across the floor, NOT on the handle, NOT on the stove "
            f"outside the pan).\n"
            f"So point 1 is in the middle and point 2 is out at the far metal "
            f"edge; the distance between them is the pan's true radius and should "
            f"be clearly LARGER than the food puddle's radius.\n"
            f"Reply with JSON only:\n"
            f'[{{"point": [y, x], "label": "{o.value}_center"}}, '
            f'{{"point": [y, x], "label": "{o.value}_rim"}}]\n'
            f"Coordinates must be normalized 0-1000 in [y, x] order."
        ),
        # Raw points in world frame (radius is computed from their separation;
        # the caller adds its own work-height offset to the center).
        world_offset_m=np.zeros(3, dtype=np.float64),
    ),
    (Object.LADLE, "grasp"): _Detection(
        prompt=lambda o: (
            f"IMPORTANT — IGNORE THE ROBOT: The image may show the robot's own "
            f"white/grey Franka gripper fingers, wrist, or zip-ties in the "
            f"FOREGROUND (often along the bottom edge of the image, very close "
            f"to the camera). Do NOT pick any point on the robot, the gripper, "
            f"the camera mount, or anything attached to the robot arm.\n"
            f"\n"
            f"Find the **ladle** standing on the table/counter: a metal ladle "
            f"with its bowl resting on the surface and its handle pointing "
            f"upward. A pair of black plastic kitchen tongs is zip-tied to the "
            f"ladle handle. On those tongs (toward the free end, farthest from "
            f"the ladle bowl) there is a short strip of **BLUE tape** (blue "
            f"plaster / blue marker) — that BLUE strip on the LADLE's tongs is "
            f"the ONLY valid grasp target. Ignore any black tape, plain metal, "
            f"or other coloured marks.\n"
            f"\n"
            f"Choose TWO distinct points along the tongs' long axis, both within "
            f"that BLUE strip on the ladle (not on the pan, not on the stove, "
            f"not on the robot), so that their midpoint is the grasp target and "
            f"the line between them defines the tong axis (the gripper will "
            f"close perpendicular to this axis).\n"
            f"Reply with JSON only:\n"
            f'[{{"point": [y, x], "label": "{o.value}_blue_strip_1"}}, '
            f'{{"point": [y, x], "label": "{o.value}_blue_strip_2"}}]\n'
            f"Coordinates must be normalized 0-1000 in [y, x] order."
        ),
        select="midpoint",
        prefer_gemini_pixel_depth=True,
        min_depth_m=LADLE_MIN_DEPTH_M,
    ),
    (Object.LADLE, "handle"): _Detection(
        prompt=lambda o: (
            f"In the image, locate the **handle** of the **{o.value}**. "
            f"Reply with JSON only:\n"
            f'[{{"point": [y, x], "label": "{o.value}_handle"}}]\n'
            f"Coordinates must be normalized 0-1000 in [y, x] order."
        ),
        select="first",
    ),
    (Object.PAN, "grasp"): _Detection(
        prompt=lambda o: (
            f"In the image, locate the handle of the **{o.value}**. "
            f"Return TWO points that lie ALONG the handle's long axis "
            f"(the line running from where the handle meets the pan body "
            f"outward toward the handle tip). The two points define the handle "
            f"DIRECTION, so spread them as FAR APART as possible along the "
            f"handle's length to give an accurate axis:\n"
            f"  - Point 1 (GRASP point): on the SOLID, thick part of the "
            f"handle, just outside where it joins the pan body. Must be on "
            f"actual handle material — not on the pan body itself and not "
            f"in any gap, slot, or cutout.\n"
            f"  - Point 2 (TIP point): at the FAR END of the handle, on the "
            f"solid handle centerline as close to the very tip as possible "
            f"(NOT a few cm from point 1 — go all the way out to the end so "
            f"the line between the two points spans the full visible handle).\n"
            f"\n"
            f"Both points must sit on the handle's centerline (front-to-back, "
            f"running lengthwise). The line from point 1 to point 2 must run "
            f"ALONG the handle's length, NOT across its width, and should be as "
            f"long as the handle itself.\n"
            f"\n"
            f"Reply with JSON only:\n"
            f'[{{"point": [y, x], "label": "{o.value}_handle_grasp"}}, '
            f'{{"point": [y, x], "label": "{o.value}_handle_axis"}}]\n'
            f"Coordinates must be normalized 0-1000 in [y, x] order."
        ),
        # Position/orientation are decided by _build_pose_pan_handle below;
        # ``select`` is unused for grasp poses but kept consistent.
        select="first",
        # Use Gemini's exact handle pixel depth when it has a valid read, else
        # snap to the NEAREST valid pixel and deproject there. Gemini's point 1
        # IS the intended grasp spot on the solid handle, so we trust it rather
        # than reaching for the shallowest pixel anywhere in the patch.
        prefer_gemini_pixel_depth=True,
    ),
    (Object.PAN, "back_burner"): _Detection(
        prompt=lambda o: (
            "In the image, locate the **stove burners**. Each burner position is "
            "marked by a small **white cross ('+')** on the stove surface. There "
            "are several white crosses on the stove.\n"
            "Pick the ONE white cross for the BACK burner: among all the white "
            "crosses, choose the one that is nearest the TOP of the image "
            "(farthest from the camera) AND the RIGHTMOST. If there is a tie, "
            "prefer the upper one.\n"
            "Return ONE point at the EXACT CENTER of that white cross (where the "
            "two strokes of the '+' intersect), on the stove surface — not on the "
            "pan, not on any pot, not on food.\n"
            "Reply with JSON only:\n"
            '[{"point": [y, x], "label": "back_burner"}]\n'
            "Coordinates must be normalized 0-1000 in [y, x] order."
        ),
        select="first",
        # Flat stove surface: trust Gemini's pixel depth (else nearest valid).
        prefer_gemini_pixel_depth=True,
        world_offset_m=np.zeros(3, dtype=np.float64),
    ),
    (Object.EGG_CRACKER, "grasp"): _Detection(
        # Grab the BLUE grasp marks directly (one point per handle).
        # ``_egg_cracker_strip_prompt`` (the red-tape bounding-box version)
        # is kept above for the visual-servo path; swap it back in here if
        # the blue marks stop tracking well.
        prompt=_egg_cracker_blue_prompt,
        select="midpoint",
        # Handles are thin: trust Gemini's exact pixel when it has depth,
        # else snap to the nearest shallowest valid pixel (deproject there).
        prefer_gemini_pixel_depth=True,
    ),
    (Object.EGG_CRACKER, "center"): _Detection(
        # Single point at the cradle center -- where an egg sits in the cracker.
        # Used to aim the tong-held egg over the cracker before releasing it.
        prompt=lambda o: (
            "There are TWO similar kitchen tools in the scene: a pair of "
            "TONGS and an EGG CRACKER. Both may have red and blue tape, so do "
            "NOT use tape colors alone to decide which tool is which. The EGG "
            "CRACKER is the one with a round CRADLE / ring that a single egg "
            "sits inside to be cracked; the TONGS are the plain two-armed "
            "gripping tool with no egg cradle. You must find the EGG CRACKER "
            "(NOT the tongs).\n"
            "In the image, locate the **egg cracker**: a GRAY, one-handed "
            "kitchen egg cracker resting on the table (a small device with a "
            "round cradle / ring that a single egg sits inside to be cracked, "
            "and one handle off to the side).\n"
            "The cradle ring is OPEN in the middle, so the table is visible "
            "through it. Choose ONE point on the TABLE seen through that "
            "opening, at the CENTER of the cradle ring -- the spot where an "
            "egg would rest inside the cracker. Do NOT pick the tongs, the "
            "handle, the cradle rim/ring itself, or the surrounding table "
            "outside the ring.\n"
            "Reply with JSON only:\n"
            '[{"point": [y, x], "label": "egg_cracker_center"}]\n'
            "Coordinates must be normalized 0-1000 in [y, x] order."
        ),
        select="first",
        # The cradle CENTER is its own target, not the cracker grasp point, so
        # do NOT inherit EGG_CRACKER's grasp ``gemini_world_offset_m`` (that
        # offset is tuned for the handle grasp). Use the dedicated cradle offset.
        world_offset_m=EGG_CRACKER_CRADLE_CENTER_WORLD_OFFSET_M,
    ),
    (Object.TONGS, "grasp"): _Detection(
        # One point per red tape mark; gripper closes ALONG the line between
        # them, exactly like the egg cracker (``_build_pose_egg_cracker``).
        prompt=_tongs_red_tape_prompt,
        select="midpoint",
        # Thin tape marks: trust Gemini's exact pixel when it has depth, else
        # snap to the nearest shallowest valid pixel (same as the egg cracker).
        prefer_gemini_pixel_depth=True,
    ),
    (Object.WHISK, "grasp"): _Detection(
        prompt=_whisk_red_tape_prompt,
        select="midpoint",
        prefer_gemini_pixel_depth=True,
        min_depth_m=WHISK_MIN_DEPTH_M,
    ),
    (Object.WHISK, "grasp_blue_plaster"): _Detection(
        prompt=_whisk_blue_plaster_prompt,
        select="first",
        prefer_gemini_pixel_depth=True,
        min_depth_m=WHISK_MIN_DEPTH_M,
    ),
    # Egg: Gemini's point is the grasp spot, so use the default
    # prefer_gemini_pixel_depth=True (exact pixel if it has depth, else the
    # nearest valid pixel) rather than the rim shallowest-in-patch rule.
    # Only accept depth from pixels >= EGG_MIN_DEPTH_M away, and sample over a
    # tight EGG_DEPTH_PATCH_RADIUS patch around Gemini's pixel.
    (Object.EGG, "grasp"): _Detection(
        prompt=_egg_prompt,
        select="first",
        min_depth_m=EGG_MIN_DEPTH_M,
        depth_patch_radius=EGG_DEPTH_PATCH_RADIUS,
    ),
    (Object.BOTTLE, "grasp"): _Detection(
        prompt=_cylinder_long_axis_prompt,
        select="first",
        # Cylinder rim grasp: keep the legacy shallowest-in-patch depth rule.
        prefer_gemini_pixel_depth=False,
    ),
    (Object.JAR, "grasp"): _Detection(
        prompt=_cylinder_long_axis_prompt,
        select="first",
        prefer_gemini_pixel_depth=False,
    ),
}

_DEFAULT_DETECTION = _Detection(select="first")


def _ensure_gemini(ctx: TaskContext):
    if ctx.gemini_client is None:
        ctx.gemini_client = gp.make_genai_client(gp.resolve_api_key())
    return ctx.gemini_client


def _camera_to_world(ctx: TaskContext, point_camera: tuple[float, float, float]) -> np.ndarray:
    T_base_flange = arm.read_T_base_flange(ctx.redis, ctx.endeffector_transform_key)
    if T_base_flange is None:
        raise RuntimeError("Could not read T_end_effector from Redis")
    p_cam_h = np.ones(4, dtype=np.float64)
    p_cam_h[:3] = np.asarray(point_camera, dtype=np.float64).reshape(3)
    p_base_h = T_base_flange @ T_FLANGE_CAMERA @ p_cam_h
    return p_base_h[:3].copy()


def _pad_panels_for_pixel(
    panels: list[np.ndarray],
    pixel: tuple[int, int] | None,
    margin: int = 30,
    bg: tuple[int, int, int] = (40, 40, 40),
) -> tuple[list[np.ndarray], tuple[int, int] | None]:
    """Pad every panel by the same amount so ``pixel`` (+ ``margin``)
    sits inside the new canvas of each panel.

    Returns the padded panels and the new pixel coordinates. The
    z-offset of a computed grasp pose can push the projected pixel
    outside the original image (off the top when we lift the grasp);
    padding lets us still mark the diamond in the saved debug image.
    The same pad is applied to all panels so they can be ``np.hstack``-ed
    afterwards.
    """
    if pixel is None:
        return panels, pixel
    if not panels:
        return panels, pixel
    h, w = panels[0].shape[:2]
    u, v = pixel
    pad_top = max(0, margin - v)
    pad_bot = max(0, (v + margin + 1) - h)
    pad_left = max(0, margin - u)
    pad_right = max(0, (u + margin + 1) - w)
    if pad_top == 0 and pad_bot == 0 and pad_left == 0 and pad_right == 0:
        return panels, pixel
    padded = [
        cv2.copyMakeBorder(
            p, pad_top, pad_bot, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=bg,
        )
        for p in panels
    ]
    return padded, (u + pad_left, v + pad_top)


def _save_gemini_response(
    ctx: TaskContext,
    overlay: np.ndarray | None,
    depth_vis: np.ndarray,
    patches: list[dict | None] | None = None,
    grasp_pixel: tuple[int, int] | None = None,
    grasp_label: str = "grasp",
) -> None:
    """Write the ``[color overlay | depth panel]`` debug composite.

    When ``patches`` is provided (from :func:`gp.query_color_depth_overlay`),
    the same per-point sampling rectangle + shallow-pixel marker that's
    already on the color overlay is also drawn on the colorized depth
    panel so you can visually verify which depth values produced the
    grasp Z.

    When ``grasp_pixel`` is provided (typically from
    :func:`_project_world_to_pixel`), the computed world-frame grasp
    pose is also marked with a diamond + label on both panels. This is
    the pose *after* per-object offsets (XY shift, Z lift, etc.) — i.e.
    where the EE will actually go, not the raw Gemini-detected point.
    Both panels are padded with a dark border if the projected pixel
    falls outside the captured camera FOV (common when ``approach_dz``
    or a Z-lift offset pushes the grasp above the camera's top edge).
    """
    if overlay is None or not ctx.gemini_response_path:
        return
    try:
        save_path = Path(ctx.gemini_response_path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        depth_panel = np.ascontiguousarray(depth_vis).copy()
        if overlay.shape[:2] != depth_panel.shape[:2]:
            depth_panel = cv2.resize(depth_panel, (overlay.shape[1], overlay.shape[0]))
        if patches:
            depth_panel = gp.draw_patch_overlay(depth_panel, patches)
        color_panel = np.ascontiguousarray(overlay).copy()
        if grasp_pixel is not None:
            padded_panels, padded_pixel = _pad_panels_for_pixel(
                [color_panel, depth_panel], grasp_pixel
            )
            color_panel, depth_panel = padded_panels
            in_orig = (
                0 <= grasp_pixel[0] < overlay.shape[1]
                and 0 <= grasp_pixel[1] < overlay.shape[0]
            )
            note = "" if in_orig else " (off-FOV, image padded)"
            print(
                f"[gemini] grasp_pixel orig={grasp_pixel} "
                f"draw_at={padded_pixel}{note}"
            )
            color_panel = gp.draw_world_marker(color_panel, padded_pixel, label=grasp_label)
            depth_panel = gp.draw_world_marker(depth_panel, padded_pixel, label=grasp_label)
        composite = np.ascontiguousarray(np.hstack((color_panel, depth_panel))).copy()
        cv2.imwrite(str(save_path), composite)
        print(f"[gemini] saved response: {save_path}")
    except Exception as e:
        print(f"[gemini] failed to save response: {e}")


def _project_world_to_pixel(
    ctx: TaskContext,
    world_xyz: np.ndarray,
    intrinsics,
) -> tuple[int, int] | None:
    """World XYZ -> RealSense color-frame pixel.

    Inverse of the camera-to-world chain used by :func:`_camera_to_world`:
    apply ``inv(T_base_flange @ T_FLANGE_CAMERA)`` to land in camera
    space, then project via the RealSense intrinsics. Returns ``None``
    when the point is behind the camera, when ``T_base_flange`` is not
    available, or when the projection lands outside the image.
    """
    try:
        T_base_flange = arm.read_T_base_flange(ctx.redis, ctx.endeffector_transform_key)
        if T_base_flange is None:
            return None
        T_base_cam = T_base_flange @ T_FLANGE_CAMERA
        try:
            T_cam_base = np.linalg.inv(T_base_cam)
        except np.linalg.LinAlgError:
            return None
        p_world_h = np.ones(4, dtype=np.float64)
        p_world_h[:3] = np.asarray(world_xyz, dtype=np.float64).reshape(3)
        p_cam = (T_cam_base @ p_world_h)[:3]
        if p_cam[2] <= 0.0:
            return None
        import pyrealsense2 as rs

        px = rs.rs2_project_point_to_pixel(
            intrinsics,
            [float(p_cam[0]), float(p_cam[1]), float(p_cam[2])],
        )
        return int(round(px[0])), int(round(px[1]))
    except Exception:
        return None


def _base_grasp_orientation(
    ctx: TaskContext,
    obj: Object,
    orientation_source: str,
) -> tuple[np.ndarray, str]:
    if orientation_source == "current":
        pose = arm.read_current_ee_world(ctx.redis)
        if pose is not None:
            return pose[1].copy(), "current"
        print("[gemini] warning: no current EE orientation; using fixed object orientation")
    spec = OBJECT_DEFAULTS[obj]
    return spec.grasp_ori.copy(), "fixed"


def _apply_perpendicular_yaw(
    base_orientation: np.ndarray,
    p1_world: np.ndarray,
    p2_world: np.ndarray,
    *,
    orientation_source: str,
    parallel: bool = False,
) -> tuple[np.ndarray, float | None, bool]:
    """Rotate ``base_orientation`` about tool +Z to align body +Y to the axis.

    Projects the handle axis (``p2 - p1``) into the plane perpendicular to the
    tool axis (body +Z), then applies the smallest rotation about tool +Z that
    makes body +Y (the gripper closing direction) either PERPENDICULAR to that
    in-plane axis (default) or PARALLEL to it (``parallel=True``). Parallel is
    used by the egg cracker, where the two points sit on the two handles and
    the jaws must close ALONG the line between them (one finger per handle) —
    i.e. the gripper goes in at the angle given by the two points rather than
    perpendicular to it. This preserves the arm's live roll/pitch (critical for
    pan handle grasps where the handle is not horizontal in world XY).

    ``orientation_source`` is only used when choosing ``base_orientation``;
    the rotation is always ``base @ R_delta`` (tool-frame yaw).
    """
    _ = orientation_source
    base = np.asarray(base_orientation, dtype=np.float64).reshape(3, 3)
    p1 = np.asarray(p1_world, dtype=np.float64).reshape(3)
    p2 = np.asarray(p2_world, dtype=np.float64).reshape(3)

    axis = p2 - p1
    if np.linalg.norm(axis) < 1e-6:
        print("[gemini] warning: axis points too close; keeping base orientation")
        return base, None, False

    z = base[:, 2].copy()
    zn = np.linalg.norm(z)
    if zn < 1e-6:
        z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        z /= zn

    axis_in_plane = axis - np.dot(axis, z) * z
    apn = np.linalg.norm(axis_in_plane)
    if apn < 1e-6:
        print(
            "[gemini] warning: handle axis parallel to tool Z; "
            "keeping base orientation"
        )
        return base, None, False
    axis_in_plane /= apn
    axis_yaw_rad = float(np.arctan2(axis_in_plane[1], axis_in_plane[0]))

    close_ref = base[:, 1] - np.dot(base[:, 1], z) * z
    crn = np.linalg.norm(close_ref)
    if crn < 1e-6:
        close_ref = base[:, 0] - np.dot(base[:, 0], z) * z
        crn = np.linalg.norm(close_ref)
    if crn < 1e-6:
        print("[gemini] warning: no in-plane closing reference; keeping base")
        return base, float(np.degrees(axis_yaw_rad)), False
    close_ref /= crn

    # Perpendicular (default): closing dir ⊥ axis = z × axis. Parallel: closing
    # dir aligned WITH the in-plane axis (jaws close along the line between the
    # two points, e.g. squeezing the two egg-cracker handles together).
    if parallel:
        close_tgt_a = axis_in_plane.copy()
    else:
        close_tgt_a = np.cross(z, axis_in_plane)
    close_tgt_a /= np.linalg.norm(close_tgt_a)
    close_tgt_b = -close_tgt_a
    close_tgt = (
        close_tgt_a
        if float(np.dot(close_tgt_a, close_ref)) >= float(np.dot(close_tgt_b, close_ref))
        else close_tgt_b
    )

    sin_a = float(np.dot(np.cross(close_ref, close_tgt), z))
    cos_a = float(np.dot(close_ref, close_tgt))
    delta = float(np.arctan2(sin_a, cos_a))

    c, s = np.cos(delta), np.sin(delta)
    r_delta = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    out = base @ r_delta

    # TEMPORARY: cancel the ~45° yaw error injected by the +45° Z rotation in
    # T_FLANGE_CAMERA (see GRASP_AXIS_YAW_OFFSET_DEG). Applied as a tool-frame
    # (body +Z) rotation so it composes with the alignment above.
    if GRASP_AXIS_YAW_OFFSET_DEG:
        o = np.radians(GRASP_AXIS_YAW_OFFSET_DEG)
        co, so = np.cos(o), np.sin(o)
        r_offset = np.array(
            [[co, -so, 0.0], [so, co, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        out = out @ r_offset

    closing = out[:, 1]
    print(
        f"[gemini] handle axis yaw: {np.degrees(axis_yaw_rad):+.2f} deg  "
        f"(tool-Z {'parallel' if parallel else 'perp'}, "
        f"delta={np.degrees(delta):+.2f} deg, "
        f"yaw_offset={GRASP_AXIS_YAW_OFFSET_DEG:+.1f} deg, "
        f"closing=[{closing[0]:+.3f}, {closing[1]:+.3f}, {closing[2]:+.3f}])"
    )
    return out, float(np.degrees(axis_yaw_rad)), True


def _pan_grasp_orientation_tool_down(
    p1_world: np.ndarray,
    p2_world: np.ndarray,
) -> tuple[np.ndarray, float | None, bool]:
    """Pan handle: tool-down (world −Z) with jaws closing ⊥ handle in the horizontal plane.

    Do **not** use the live ARM_HOME orientation as a rotation base — at the
    pan station the flange is tilted (~45°+ from vertical tool-down), so
    ``_apply_perpendicular_yaw`` about tool +Z inherits that tilt and the
    commanded closing axis picks up a large Z component (see logs:
    ``delta≈−70°``, ``closing=[..., +0.16]`` in Z).

    Instead: project the handle axis into world XY, set tool ``z = [0,0,−1]``,
    and set body ``+Y`` to ``z × handle`` (perpendicular in the counter plane).
    """
    p1 = np.asarray(p1_world, dtype=np.float64).reshape(3)
    p2 = np.asarray(p2_world, dtype=np.float64).reshape(3)
    axis = p2 - p1
    axis_xy = axis.copy()
    axis_xy[2] = 0.0
    apn = np.linalg.norm(axis_xy)
    if apn < 1e-6:
        print("[gemini:pan] handle axis vertical in XY; using EE_ORI_TOOL_DOWN")
        return EE_ORI_TOOL_DOWN.copy(), None, False

    handle = axis_xy / apn
    axis_yaw_rad = float(np.arctan2(handle[1], handle[0]))
    z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    close_a = np.cross(z_axis, handle)
    close_a /= np.linalg.norm(close_a)
    close_b = -close_a
    ref = EE_ORI_TOOL_DOWN[:, 1]
    close_dir = (
        close_a
        if float(np.dot(close_a, ref)) >= float(np.dot(close_b, ref))
        else close_b
    )

    y_axis = close_dir
    x_axis = np.cross(y_axis, z_axis)
    xn = np.linalg.norm(x_axis)
    if xn < 1e-6:
        return EE_ORI_TOOL_DOWN.copy(), float(np.degrees(axis_yaw_rad)), False
    x_axis /= xn
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    out = np.column_stack([x_axis, y_axis, z_axis])

    # ``out``'s closing direction (body +Y) is built as ``z × handle`` from the
    # WORLD-frame handle axis, i.e. perpendicular to the DETECTED axis. But the
    # +45° Z rotation baked into T_FLANGE_CAMERA (see GRASP_AXIS_YAW_OFFSET_DEG)
    # rotates the measured handle direction itself by ~45°, so perpendicular-to-
    # detected lands ~45° off perpendicular-to-true. Apply the same +45° yaw
    # correction the axis-based ``_apply_perpendicular_yaw`` path uses, about the
    # tool's straight-down Z, to recover the true perpendicular grasp.
    if GRASP_AXIS_YAW_OFFSET_DEG:
        o = np.radians(GRASP_AXIS_YAW_OFFSET_DEG)
        co, so = np.cos(o), np.sin(o)
        r_offset = np.array(
            [[co, -so, 0.0], [so, co, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        out = out @ r_offset

    closing = out[:, 1]
    closing_yaw = float(np.degrees(np.arctan2(closing[1], closing[0])))
    print(
        f"[gemini:pan] handle axis yaw: {np.degrees(axis_yaw_rad):+.2f} deg  "
        f"closing yaw: {closing_yaw:+.2f} deg "
        f"(tool-down perp, yaw_offset={GRASP_AXIS_YAW_OFFSET_DEG:+.1f} deg, "
        f"closing=[{closing[0]:+.3f}, {closing[1]:+.3f}, {closing[2]:+.3f}])"
    )
    return out, float(np.degrees(axis_yaw_rad)), True


# ---------------------------------------------------------------------------
# Per-object grasp pose builders.
#
# Each builder converts the raw Gemini camera-frame points into a world-frame
# GraspPose with its own position-selection rule and orientation rule. Add a
# new builder + register it in ``_GRASP_POSE_BUILDERS`` when an object needs
# different grasp geometry. The shared frame capture / Gemini call lives in
# ``find_grasp_pose``.
# ---------------------------------------------------------------------------

GraspPoseBuilder = Callable[
    [
        TaskContext,
        Object,
        "_Detection",
        list[tuple[float, float, float] | None],
        str,
    ],
    "GraspPose",
]


def _xy_mid_with_p1_z(p1: np.ndarray, p2: np.ndarray | None) -> np.ndarray:
    """Average XY between two Gemini-derived world points, keep Z from p1.

    Depth from RealSense averaged over two pixels tends to drift when one
    point sits on a steep surface or on a thin lip — taking Z directly
    from p1 keeps the grasp height anchored to a single, deterministic
    depth sample. XY is still the midpoint so the gripper centers on
    the chord between the two points.
    """
    if p2 is None:
        return p1.copy()
    return np.array([(p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5, p1[2]], dtype=np.float64)


def _build_pose_bowl(
    ctx: TaskContext,
    obj: Object,
    detection: "_Detection",
    points_camera: list[tuple[float, float, float] | None],
    orientation_source: str,
) -> "GraspPose":
    """Bowl: XY = midpoint of two rim points, Z = p1's depth; gripper closes across the chord."""
    valid = [p for p in points_camera if p is not None]
    if not valid:
        raise ValueError("bowl grasp: no valid points")
    p1 = _camera_to_world(ctx, valid[0])
    p2 = _camera_to_world(ctx, valid[1]) if len(valid) >= 2 else None
    center = _xy_mid_with_p1_z(p1, p2)
    position = center + detection.offset(obj)
    base_ori, source = _base_grasp_orientation(ctx, obj, orientation_source)
    if p2 is not None and OBJECT_DEFAULTS[obj].grasp_align_jaws_to_detected_axis:
        grasp_ori, rim_yaw_deg, rim_applied = _apply_perpendicular_yaw(
            base_ori, p1, p2, orientation_source=source,
        )
    else:
        grasp_ori, rim_yaw_deg, rim_applied = base_ori, None, False
    print(f"[gemini:{obj.value}/bowl-pose] world={position.tolist()}")
    return GraspPose(position, grasp_ori, rim_yaw_deg, rim_applied)


def _build_pose_mixing_bowl(
    ctx: TaskContext,
    obj: Object,
    detection: "_Detection",
    points_camera: list[tuple[float, float, float] | None],
    orientation_source: str,
) -> "GraspPose":
    """Mixing bowl: position = arc-on-rim point (index 2); yaw from anchors 1–2."""
    valid = [p for p in points_camera if p is not None]
    if len(valid) < 2:
        raise ValueError("mixing bowl grasp: need at least 2 rim anchors")
    p1 = _camera_to_world(ctx, valid[0])
    p2 = _camera_to_world(ctx, valid[1])
    if len(valid) < 3:
        raise ValueError(
            "mixing bowl grasp: missing arc-on-rim grasp point "
            "(expected injection from find_grasp_pose)"
        )
    position = _camera_to_world(ctx, valid[2]) + detection.offset(obj)
    base_ori, source = _base_grasp_orientation(ctx, obj, orientation_source)
    if OBJECT_DEFAULTS[obj].grasp_align_jaws_to_detected_axis:
        grasp_ori, rim_yaw_deg, rim_applied = _apply_perpendicular_yaw(
            base_ori, p1, p2, orientation_source=source,
        )
    else:
        grasp_ori, rim_yaw_deg, rim_applied = base_ori, None, False
    print(f"[gemini:{obj.value}/mixing-bowl-pose] world={position.tolist()}")
    return GraspPose(position, grasp_ori, rim_yaw_deg, rim_applied)


def _build_pose_cylinder(
    ctx: TaskContext,
    obj: Object,
    detection: "_Detection",
    points_camera: list[tuple[float, float, float] | None],
    orientation_source: str,
) -> "GraspPose":
    """Cylinder: target is on the long-axis midpoint; gripper closes perpendicular to axis, and normal to the cylinder surface."""
    valid = [p for p in points_camera if p is not None]
    if len(valid) < 2:
        raise ValueError(f"cylinder grasp: need 2 long-axis points, got {len(valid)}")

    p1 = _camera_to_world(ctx, valid[0])
    p2 = _camera_to_world(ctx, valid[1])
    position = _xy_mid_with_p1_z(p1, p2) + detection.offset(obj)

    base_ori, source = _base_grasp_orientation(ctx, obj, orientation_source)
    if OBJECT_DEFAULTS[obj].grasp_align_jaws_to_detected_axis:
        grasp_ori, axis_yaw_deg, axis_applied = _apply_perpendicular_yaw(
            base_ori, p1, p2, orientation_source=source,
        )
    else:
        grasp_ori, axis_yaw_deg, axis_applied = base_ori, None, False

    print(f"[gemini:{obj.value}/cylinder-pose] world={position.tolist()}")
    return GraspPose(position, grasp_ori, axis_yaw_deg, axis_applied)


def _build_pose_whisk(
    ctx: TaskContext,
    obj: Object,
    detection: "_Detection",
    points_camera: list[tuple[float, float, float] | None],
    orientation_source: str,
) -> "GraspPose":
    """Whisk: fixed taught grasp orientation. Accepts a single tape point (used
    directly) or two tape-axis points (midpoint, p1 Z)."""
    valid = [p for p in points_camera if p is not None]
    if len(valid) < 1:
        raise ValueError(f"whisk grasp: need at least 1 tape point, got {len(valid)}")

    if len(valid) == 1:
        position = _camera_to_world(ctx, valid[0]) + detection.offset(obj)
    else:
        p1 = _camera_to_world(ctx, valid[0])
        p2 = _camera_to_world(ctx, valid[1])
        position = _xy_mid_with_p1_z(p1, p2) + detection.offset(obj)

    grasp_ori, _ = _base_grasp_orientation(ctx, obj, orientation_source)
    fwd = float(OBJECT_DEFAULTS[obj].grasp_forward_tool_z_m)
    if fwd != 0.0:
        position = position + grasp_ori[:, 2] * fwd
    up_z = float(OBJECT_DEFAULTS[obj].grasp_world_z_offset_m)
    if up_z != 0.0:
        position[2] += up_z
    print(
        f"[gemini:{obj.value}/whisk-pose] world={position.tolist()} "
        f"(fixed taught grasp orientation"
        f"{f', +{fwd * 100:.1f} cm along tool +Z' if fwd else ''}"
        f"{f', +{up_z * 100:.1f} cm world +Z' if up_z else ''})"
    )
    return GraspPose(position, grasp_ori, None, False)


def _flatten_to_tool_down(R: np.ndarray) -> np.ndarray:
    """Return a tool-straight-down orientation that keeps only ``R``'s yaw.

    Forces body +Z to world -Z (no roll/pitch tilt) and sets body +Y to ``R``'s
    body +Y projected into the world XY plane (the gripper open/close axis), so
    the jaw-alignment yaw is preserved but the wrist no longer inherits any tilt
    from the base orientation it was built on.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    y = R[:, 1].copy()
    y[2] = 0.0
    n = np.linalg.norm(y)
    if n < 1e-6:
        return EE_ORI_TOOL_DOWN.copy()
    y /= n
    z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, z])


def _build_pose_egg_cracker(
    ctx: TaskContext,
    obj: Object,
    detection: "_Detection",
    points_camera: list[tuple[float, float, float] | None],
    orientation_source: str,
) -> "GraspPose":
    """Egg cracker: one point per handle; gripper goes in at the angle of the
    two points and closes ALONG the line between them (one finger per handle,
    squeezing the handles together) rather than perpendicular to it."""
    valid = [p for p in points_camera if p is not None]
    if len(valid) < 2:
        raise ValueError(f"egg cracker grasp: need 2 handle points, got {len(valid)}")

    p1 = _camera_to_world(ctx, valid[0])
    p2 = _camera_to_world(ctx, valid[1])
    position = _xy_mid_with_p1_z(p1, p2) + detection.offset(obj)

    # --- extrinsic-diagnostic logging --------------------------------------
    # Dump the LIVE flange pose actually used by _camera_to_world plus the raw
    # camera coords and per-handle world points, so the full transform chain
    # (T_base_flange @ T_FLANGE_CAMERA @ p_cam) can be reproduced offline and
    # compared against ground truth. Remove once the extrinsic is trusted.
    T_bf = arm.read_T_base_flange(ctx.redis, ctx.endeffector_transform_key)
    if T_bf is not None:
        R_bf = T_bf[:3, :3]
        t_bf = T_bf[:3, 3]
        np.set_printoptions(precision=4, suppress=True)
        print(f"[gemini:diag] live T_base_flange t={t_bf.tolist()}")
        print(f"[gemini:diag] live T_base_flange R rows:")
        for row in R_bf:
            print(f"[gemini:diag]   {row.tolist()}")
        print(f"[gemini:diag] tool_z(world)={R_bf[:, 2].tolist()}  (optical axis ~ depth dir)")
    print(f"[gemini:diag] cam p0={list(valid[0])}  cam p1={list(valid[1])}")
    print(f"[gemini:diag] world p1={p1.tolist()}")
    print(f"[gemini:diag] world p2={p2.tolist()}")
    print(f"[gemini:diag] world midZ-from-p1={position.tolist()} (offset={detection.offset(obj).tolist()})")
    # -----------------------------------------------------------------------

    base_ori, source = _base_grasp_orientation(ctx, obj, orientation_source)
    grasp_ori, axis_yaw_deg, axis_applied = _apply_perpendicular_yaw(
        base_ori, p1, p2, orientation_source=source, parallel=True,
    )
    # Egg cracker is grasped from directly above a flat counter: force the wrist
    # tool-straight-down (no inherited roll/pitch tilt), keeping only the yaw.
    grasp_ori = _flatten_to_tool_down(grasp_ori)

    print(f"[gemini:{obj.value}/egg-cracker-pose] world={position.tolist()}")
    return GraspPose(position, grasp_ori, axis_yaw_deg, axis_applied)


def _build_pose_pan_handle(
    ctx: TaskContext,
    obj: Object,
    detection: "_Detection",
    points_camera: list[tuple[float, float, float] | None],
    orientation_source: str,
) -> "GraspPose":
    """Pan: position = handle grasp point (point 1); orientation = tool pointing
    straight down with the jaws closing PERPENDICULAR to the handle axis.

    Gemini returns two points along the handle's long axis. Point 1 is the grasp
    spot; the (point2 - point1) vector is the handle axis. ``_pan_grasp_orientation_tool_down``
    projects that axis into world XY, points the tool ``z`` straight down, and
    sets the closing direction perpendicular to the handle in the horizontal
    plane, so the gripper closes across the handle width.
    """
    valid = [p for p in points_camera if p is not None]
    if not valid:
        raise ValueError("pan grasp: no valid points")
    p1 = _camera_to_world(ctx, valid[0])
    position = p1 + detection.offset(obj)
    if len(valid) >= 2:
        p2 = _camera_to_world(ctx, valid[1])
        grasp_ori, axis_yaw_deg, axis_applied = _pan_grasp_orientation_tool_down(
            p1, p2
        )
    else:
        print(
            f"[gemini:{obj.value}/pan-pose] only one handle point; "
            f"falling back to tool-down (no handle-axis yaw)"
        )
        grasp_ori, axis_yaw_deg, axis_applied = EE_ORI_TOOL_DOWN.copy(), None, False
    print(
        f"[gemini:{obj.value}/pan-pose] world={position.tolist()} "
        f"(tool-down, jaws ⊥ handle axis)"
    )
    return GraspPose(position, grasp_ori, axis_yaw_deg, axis_applied)


def _build_pose_egg(
    ctx: TaskContext,
    obj: Object,
    detection: "_Detection",
    points_camera: list[tuple[float, float, float] | None],
    orientation_source: str,
) -> "GraspPose":
    """Egg: position = the single detected point; orientation = home wrist flattened
    to tool-straight-down with NO detected-axis yaw (the egg is symmetric)."""
    valid = [p for p in points_camera if p is not None]
    if not valid:
        raise ValueError("egg grasp: no valid points")
    p1 = _camera_to_world(ctx, valid[0])
    position = p1 + detection.offset(obj)
    base_ori, _ = _base_grasp_orientation(ctx, obj, orientation_source)
    grasp_ori = _flatten_to_tool_down(base_ori)
    print(f"[gemini:{obj.value}/egg-pose] world={position.tolist()}")
    return GraspPose(position, grasp_ori)


def _build_pose_first_fixed(
    ctx: TaskContext,
    obj: Object,
    detection: "_Detection",
    points_camera: list[tuple[float, float, float] | None],
    orientation_source: str,
) -> "GraspPose":
    """Default: position = first valid point; orientation = object's fixed grasp_ori."""
    valid = [p for p in points_camera if p is not None]
    if not valid:
        raise ValueError(f"{obj.value} grasp: no valid points")
    p1 = _camera_to_world(ctx, valid[0])
    position = p1 + detection.offset(obj)
    base_ori, _ = _base_grasp_orientation(ctx, obj, orientation_source)
    print(f"[gemini:{obj.value}/fixed-pose] world={position.tolist()}")
    return GraspPose(position, base_ori)


_GRASP_POSE_BUILDERS: dict[Object, GraspPoseBuilder] = {
    Object.PASTA_BOWL: _build_pose_bowl,
    Object.PLASTIC_BOWL_TOP: _build_pose_bowl,
    Object.PLASTIC_BOWL_BOTTOM: _build_pose_bowl,
    Object.MIXING_BOWL: _build_pose_mixing_bowl,
    Object.PAN: _build_pose_pan_handle,
    Object.LADLE: _build_pose_cylinder,   # handle axis → midpoint + perp yaw
    Object.BOTTLE: _build_pose_cylinder,
    Object.JAR: _build_pose_cylinder,
    # Egg cracker: XY midpoint of the two handle points + p1 Z, but the gripper
    # goes in AT the angle of the two points and closes ALONG the line between
    # them (one finger per handle) instead of perpendicular to it.
    Object.EGG_CRACKER: _build_pose_egg_cracker,
    # Tongs: two red-tape points → grasp ALONG the line between them (one
    # finger per arm), exactly like the egg cracker.
    Object.TONGS: _build_pose_egg_cracker,
    Object.WHISK: _build_pose_whisk,
    # Egg: single detected point → world position; tool-down home wrist, no yaw.
    Object.EGG: _build_pose_egg,
}


def _grasp_pose_builder(obj: Object) -> GraspPoseBuilder:
    return _GRASP_POSE_BUILDERS.get(obj, _build_pose_first_fixed)


def _detect(ctx: TaskContext, obj: Object, kind: str, *, retries: int = 1) -> np.ndarray:
    detection = _PROMPTS.get((obj, kind), _DEFAULT_DETECTION)
    selector = _SELECTORS.get(detection.select, _first_valid)
    client = _ensure_gemini(ctx)
    _, _, _, intrinsics = ctx.realsense()

    prompt = detection.resolve_prompt(obj)
    last_err: Exception | None = None
    miss_counter = [0]
    # ``GeminiTimeoutError`` doesn't count against ``retries`` — see
    # ``handle_gemini_timeout`` (first few consecutive timeouts are
    # silently auto-retried; only after that do we prompt the operator).
    # Only non-timeout failures (no RGBD frame, no points, parse
    # error, ...) consume the ``retries`` budget. ``timeout_count``
    # tracks consecutive timeouts and resets on any non-timeout
    # outcome so the auto-retry budget refreshes after a real failure.
    error_attempts = 0
    timeout_count = 0

    while True:
        try:
            triple = _grab_rgbd_frame(
                ctx, miss_counter, label=f"[gemini:{obj.value}/{kind}]"
            )
            if triple is None:
                raise RuntimeError("no RGBD frame")
            color_bgr, depth_m, depth_vis = triple
            overlay, points_camera, patches = gp.query_color_depth_overlay(
                client,
                ctx.gemini_model,
                prompt,
                ctx.gemini_temperature,
                color_bgr,
                depth_m,
                intrinsics,
                ctx.depth_patch_radius,
            )
            _save_gemini_response(ctx, overlay, depth_vis, patches)
            if points_camera is None:
                raise ValueError("Gemini returned no points")
            hit = selector(points_camera)
            xyz_world = _camera_to_world(ctx, hit) + detection.offset(obj)
            print(f"[gemini:{obj.value}/{kind}] world={xyz_world.tolist()}")
            return xyz_world
        except gp.GeminiTimeoutError as e:
            last_err = e
            timeout_count += 1
            print(f"[gemini:{obj.value}/{kind}] timed out: {e}")
            handle_gemini_timeout(obj.value, kind, timeout_count)
            continue
        except Exception as e:
            last_err = e
            error_attempts += 1
            timeout_count = 0
            print(f"[gemini:{obj.value}/{kind}] attempt {error_attempts} failed: {e}")
            if error_attempts > retries:
                raise RuntimeError(
                    f"gemini {kind} for {obj.value} failed: {last_err}"
                ) from last_err


def find_grasp_point(ctx: TaskContext, obj: Object, *, retries: int = 1) -> np.ndarray:
    return _detect(ctx, obj, "grasp", retries=retries)


# Depth quantile used for grasp queries. 0 = true min depth in the patch
# = highest world-Z point under the camera-down-looking case. Combined
# with the larger GRASP_DEPTH_PATCH_RADIUS below, this lets us latch onto
# the rim/top of an object even when Gemini's chosen pixel sat several cm
# inside the rim on the deeper interior surface. Bump this back up to ~5
# or 10 if a single noisy shallow pixel ever shows up in the patch and
# yanks the grasp Z too high.
GRASP_DEPTH_QUANTILE = 0.0

# Radius (in image pixels) of the depth patch sampled around each
# Gemini pixel when computing a grasp pose. Much larger than the default
# ``ctx.depth_patch_radius`` (5) because Gemini points routinely land
# 20-50 px away from the actual rim of the object — a tiny patch never
# sees a rim pixel and ends up reporting the interior surface depth.
# 50 px at ~25 cm camera range covers ~10 cm in world: enough to reach
# the rim of a pasta bowl from a pixel placed deep inside it. Combined
# with ``find_quantile_pixel`` (which deprojects from the rim pixel's
# own (u, v), not the Gemini pixel's), the resulting 3D point lands
# right on the rim instead of carrying the interior's XY plus the rim's
# Z. Bigger patches risk spilling onto neighboring objects — bump down
# if multiple bowls/pans sit within a few cm of each other in-frame.
GRASP_DEPTH_PATCH_RADIUS = 50


@dataclass
class _GraspAttempt:
    """Result of one Gemini grasp detection frame (see :func:`_grasp_pose_attempt`).

    On success ``pose`` is set and ``error`` is ``None``. On failure ``error``
    holds the exception and ``pose`` is ``None`` — but ``pixels`` / ``image_wh``
    may still carry WHERE Gemini saw the object (even when the 3D lift failed),
    which the recentering path uses to nudge the camera and retry.
    """

    pose: GraspPose | None = None
    points_camera: list[tuple[float, float, float] | None] | None = None
    patches: list[dict | None] | None = None
    overlay: np.ndarray | None = None
    depth_vis: np.ndarray | None = None
    pixels: list[tuple[int, int]] | None = None
    image_wh: tuple[int, int] | None = None
    error: Exception | None = None


class _InvalidEggPoseError(ValueError):
    """Egg point lifted to an impossible 3D pose; recenter and retry."""


def _validate_grasp_pose(
    obj: Object,
    pose: GraspPose,
    points_camera: list[tuple[float, float, float] | None] | None,
) -> None:
    if obj is not Object.EGG:
        return
    # Depth/world-z RANGE bounds intentionally removed: legitimate eggs were
    # being rejected for being a couple cm outside the window (e.g. world z
    # 0.458 m just over the old 0.45 m cap). We now only reject genuinely
    # garbage depth reads (NaN / non-positive), which would otherwise lift the
    # grasp to an impossible pose, and let everything else through.
    valid = [p for p in (points_camera or []) if p is not None]
    depth_m = float(valid[0][2]) if valid else float("nan")
    if not math.isfinite(depth_m) or depth_m <= 0.0:
        raise _InvalidEggPoseError(
            f"egg pose rejected: invalid depth read ({depth_m:.3f} m)"
        )


# Frames to discard before keeping one, so the kept frame is captured at the
# CURRENT arm pose rather than a stale buffered one from while the arm moved.
# (try_wait_for_frames returns the oldest queued frame.) ~5 frames at 30 fps
# clears ≈0.17 s of backlog.
_FRAME_FLUSH_COUNT = 5


def _grab_rgbd_frame(
    ctx: TaskContext,
    miss_counter: list[int],
    *,
    label: str,
    max_misses: int = 10,
    flush_frames: int = _FRAME_FLUSH_COUNT,
):
    """One aligned RGBD frame, restarting the RealSense pipeline if it wedges.

    Always pulls the *current* pipeline handles from ``ctx.realsense()`` (so it
    keeps working after a restart) and grabs a frame. ``flush_frames`` stale
    buffered frames are discarded first so the returned frame reflects the
    current camera pose (not one captured while the arm was still moving). If
    the stream returns nothing within the timeout, the pipeline is hard-restarted
    once and the grab is retried — the librealsense stream has been seen to wedge
    on a later detection in a multi-detection routine, which previously aborted
    the whole run with ``no RGBD frame``. Returns the
    ``(color_bgr, depth_m, depth_vis)`` triple, or ``None`` if a frame still
    can't be obtained after the restart.
    """
    from vision import realsense_rgbd as rs_cam

    pipeline, align, depth_scale, _ = ctx.realsense()
    triple = rs_cam.next_rgbd_frame(
        pipeline, align, depth_scale, ctx.cam_timeout_ms, miss_counter,
        max_misses=max_misses, flush_frames=flush_frames,
    )
    if triple is not None:
        return triple
    print(
        f"{label} RealSense stream wedged (no frame in {ctx.cam_timeout_ms} ms); "
        "restarting pipeline and retrying once..."
    )
    try:
        pipeline, align, depth_scale, _ = ctx.restart_realsense()
    except Exception as e:  # noqa: BLE001 - restart is best-effort recovery
        print(f"{label} RealSense restart failed: {e}")
        return None
    miss_counter[0] = 0
    return rs_cam.next_rgbd_frame(
        pipeline, align, depth_scale, ctx.cam_timeout_ms, miss_counter,
        max_misses=max_misses, flush_frames=flush_frames,
    )


def _grasp_pose_attempt(
    ctx: TaskContext,
    obj: Object,
    detection: "_Detection",
    builder: GraspPoseBuilder,
    *,
    client,
    pipeline,
    align,
    depth_scale,
    intrinsics,
    prompt: str,
    depth_quantile: float,
    depth_patch_radius: int,
    min_depth_m: float,
    orientation_source: str,
    miss_counter: list[int],
) -> _GraspAttempt:
    """One grab-frame + Gemini + build cycle. Never retries.

    Re-raises :class:`gp.GeminiTimeoutError` so the caller's loop can apply the
    auto-retry/operator-prompt policy. Every OTHER failure (no frame, no
    points, bad depth, build error) is captured into the returned
    :class:`_GraspAttempt` (``.error``) instead of raised, so the caller can
    decide whether to recenter-and-retry or count it against the retry budget.
    """
    pixels_out: list[tuple[int, int]] = []
    overlay = None
    depth_vis = None
    patches: list[dict | None] | None = None
    points_camera: list[tuple[float, float, float] | None] | None = None
    image_wh: tuple[int, int] | None = None
    try:
        triple = _grab_rgbd_frame(
            ctx, miss_counter, label=f"[gemini:{obj.value}/grasp_pose]"
        )
        if triple is None:
            raise RuntimeError("no RGBD frame")
        color_bgr, depth_m, depth_vis = triple
        image_wh = (int(color_bgr.shape[1]), int(color_bgr.shape[0]))
        overlay, points_camera, patches = gp.query_color_depth_overlay(
            client,
            ctx.gemini_model,
            prompt,
            ctx.gemini_temperature,
            color_bgr,
            depth_m,
            intrinsics,
            depth_patch_radius,
            depth_quantile=depth_quantile,
            prefer_gemini_pixel=detection.prefer_gemini_pixel_depth,
            min_depth_m=min_depth_m,
            pixels_out=pixels_out,
        )
        if points_camera is None:
            raise ValueError("Gemini returned no points")
        if obj is Object.MIXING_BOWL:
            grasp_cam = _mixing_bowl_arc_grasp_cam(
                patches,
                depth_m,
                intrinsics,
                depth_patch_radius,
                depth_quantile,
            )
            if grasp_cam is None:
                raise ValueError(
                    "mixing bowl grasp: could not lift arc-on-rim point"
                )
            pc = list(points_camera)
            if len(pc) < 3:
                pc.extend([None] * (3 - len(pc)))
            pc[2] = grasp_cam
            points_camera = pc
        pose = builder(ctx, obj, detection, points_camera, orientation_source)
        _validate_grasp_pose(obj, pose, points_camera)
        return _GraspAttempt(
            pose=pose,
            points_camera=points_camera,
            patches=patches,
            overlay=overlay,
            depth_vis=depth_vis,
            pixels=pixels_out or None,
            image_wh=image_wh,
        )
    except gp.GeminiTimeoutError:
        raise
    except Exception as e:  # noqa: BLE001 - surfaced to the caller via .error
        return _GraspAttempt(
            points_camera=points_camera,
            patches=patches,
            overlay=overlay,
            depth_vis=depth_vis,
            pixels=pixels_out or None,
            image_wh=image_wh,
            error=e,
        )


def _save_success_overlay(
    ctx: TaskContext, attempt: _GraspAttempt, obj: Object, intrinsics
) -> GraspPose:
    """Save the annotated overlay for a successful attempt and finalize the pose."""
    pose = attempt.pose
    assert pose is not None
    grasp_pixel = _project_world_to_pixel(ctx, pose.position, intrinsics)
    _save_gemini_response(
        ctx,
        attempt.overlay,
        attempt.depth_vis,
        attempt.patches,
        grasp_pixel=grasp_pixel,
        grasp_label=f"{obj.value} grasp",
    )
    pose.source_pixels = _source_pixels_from_patches(attempt.patches)
    pose.source_boxes = _source_boxes_from_patches(attempt.patches)
    return pose


def find_grasp_pose(
    ctx: TaskContext,
    obj: Object,
    *,
    kind: str = "grasp",
    retries: int = 1,
    orientation_source: str = "fixed",
    depth_quantile: float = GRASP_DEPTH_QUANTILE,
    depth_patch_radius: int = GRASP_DEPTH_PATCH_RADIUS,
) -> GraspPose:
    """Run the Gemini prompt for ``(obj, kind)`` and dispatch to its pose builder.

    Per-object position/orientation rules live in ``_GRASP_POSE_BUILDERS``;
    this function only handles the shared work of grabbing a frame, calling
    Gemini, saving the annotated overlay, and retrying on failure.

    ``depth_quantile`` controls how the patch around each Gemini pixel is
    reduced to a single depth value. Default :data:`GRASP_DEPTH_QUANTILE`
    (0) picks the patch's minimum depth so the grasp Z follows the
    highest world-Z point in the patch (= the rim/top). Override per
    call when an object's grasp surface is *not* the highest point.

    ``depth_patch_radius`` overrides ``ctx.depth_patch_radius`` for the
    grasp query. Default :data:`GRASP_DEPTH_PATCH_RADIUS` is large enough
    for the patch to actually contain the rim, even when Gemini's chosen
    pixel landed well inside it. A per-object ``_Detection.depth_patch_radius``
    (e.g. the egg) overrides this default, and ``_Detection.min_depth_m``
    imposes a near-field depth floor.
    """
    detection = _PROMPTS.get((obj, kind), _DEFAULT_DETECTION)
    builder = _grasp_pose_builder(obj)
    client = _ensure_gemini(ctx)
    pipeline, align, depth_scale, intrinsics = ctx.realsense()

    radius = (
        detection.depth_patch_radius
        if detection.depth_patch_radius is not None
        else depth_patch_radius
    )
    min_depth_m = detection.min_depth_m

    prior_response_path = ctx.gemini_response_path
    resolved_path = resolve_grasp_response_path(ctx, obj)
    if resolved_path is not None:
        ctx.gemini_response_path = str(resolved_path)

    prompt = detection.resolve_prompt(obj)
    log_kind = f"{kind}_pose" if kind != "grasp" else "grasp_pose"
    last_err: Exception | None = None
    miss_counter = [0]
    # See note in ``_detect``: timeouts are auto-retried up to
    # ``GEMINI_TIMEOUT_AUTORETRY_LIMIT`` consecutive times, then
    # prompt the operator. Non-timeout failures consume the
    # ``retries`` budget and abort the call once exhausted.
    error_attempts = 0
    timeout_count = 0

    try:
        while True:
            try:
                attempt = _grasp_pose_attempt(
                    ctx,
                    obj,
                    detection,
                    builder,
                    client=client,
                    pipeline=pipeline,
                    align=align,
                    depth_scale=depth_scale,
                    intrinsics=intrinsics,
                    prompt=prompt,
                    depth_quantile=depth_quantile,
                    depth_patch_radius=radius,
                    min_depth_m=min_depth_m,
                    orientation_source=orientation_source,
                    miss_counter=miss_counter,
                )
            except gp.GeminiTimeoutError as e:
                last_err = e
                timeout_count += 1
                print(f"[gemini:{obj.value}/{log_kind}] timed out: {e}")
                handle_gemini_timeout(obj.value, log_kind, timeout_count)
                continue

            if attempt.error is not None:
                # Match the legacy behaviour: save the plain overlay only when
                # Gemini returned NO points (so the operator can see the empty
                # frame); build/depth failures with points present don't save.
                if attempt.points_camera is None and attempt.overlay is not None:
                    _save_gemini_response(
                        ctx, attempt.overlay, attempt.depth_vis, attempt.patches
                    )
                last_err = attempt.error
                error_attempts += 1
                timeout_count = 0
                print(
                    f"[gemini:{obj.value}/{log_kind}] attempt {error_attempts} failed: "
                    f"{attempt.error}"
                )
                if error_attempts > retries:
                    raise RuntimeError(
                        f"gemini {kind} pose for {obj.value} failed: {last_err}"
                    ) from last_err
                continue

            return _save_success_overlay(ctx, attempt, obj, intrinsics)
    finally:
        ctx.gemini_response_path = prior_response_path


# --- Egg recenter-on-failure tunables --------------------------------------
# When an egg detection can't produce a valid grasp but Gemini still SAW the
# egg, nudge the flange so the detected pixel slides toward the frame center
# and try again. The move is a pure translation (orientation held), so the
# camera — rigidly bolted to the flange — translates with it.
EGG_RECENTER_MAX_MOVES = 3
# Pixel error (as a fraction of the frame's half-extent diagonal) below which
# the egg is considered "centered enough" and we stop nudging.
EGG_RECENTER_TOL_FRAC = 0.18
# Proportional gain applied to the computed camera-plane correction.
EGG_RECENTER_GAIN = 0.6
# Hard cap on a single recenter step (m) so a wild pixel never commands a big
# arm move.
EGG_RECENTER_MAX_STEP_M = 0.06
# Assumed camera->egg distance (m) used to convert pixel error into a metric
# camera-plane shift. Exact value only scales the step (which is capped); the
# camera looks ~straight down at the counter from the home pose.
EGG_RECENTER_NOMINAL_DEPTH_M = 0.45
# The recenter nudge only needs to roughly re-frame the egg, so use a loose
# convergence tolerance (it kept timing out at the old 0.03 m even when within
# ~0.035 m) and the default arm.move_to timeout.
EGG_RECENTER_MOVE_TOL_M = 0.05
EGG_RECENTER_MOVE_TIMEOUT_S = 3.0


def _recenter_ee_toward_pixel(
    ctx: TaskContext,
    pixel: tuple[int, int] | None,
    image_wh: tuple[int, int] | None,
    intrinsics,
    *,
    hold_orientation: np.ndarray | None = None,
    gain: float = EGG_RECENTER_GAIN,
    max_step_m: float = EGG_RECENTER_MAX_STEP_M,
    nominal_depth_m: float = EGG_RECENTER_NOMINAL_DEPTH_M,
    center_tol_frac: float = EGG_RECENTER_TOL_FRAC,
    move_timeout_s: float = EGG_RECENTER_MOVE_TIMEOUT_S,
    label: str = "[gemini/recenter]",
) -> bool:
    """Nudge the flange so ``pixel`` moves toward the image center.

    Returns ``True`` if a recenter move was issued, ``False`` if the pixel was
    already within ``center_tol_frac`` of center or the move couldn't be
    computed (missing intrinsics / transforms / current pose).
    """
    if pixel is None or image_wh is None:
        return False
    u, v = float(pixel[0]), float(pixel[1])
    w, h = float(image_wh[0]), float(image_wh[1])
    if w <= 0.0 or h <= 0.0:
        return False
    # Desired pixel shift: bring the egg to the geometric frame center.
    du = (w * 0.5) - u
    dv = (h * 0.5) - v
    err_frac = math.hypot(du / w, dv / h)
    if err_frac <= center_tol_frac:
        print(
            f"{label} egg pixel ({u:.0f},{v:.0f}) already centered "
            f"(err={err_frac:.2f} <= {center_tol_frac:.2f}); not moving."
        )
        return False
    try:
        fx = float(intrinsics.fx)
        fy = float(intrinsics.fy)
    except AttributeError:
        return False
    if fx <= 1e-6 or fy <= 1e-6:
        return False
    Z = float(nominal_depth_m)
    # RealSense optical frame: +X is image +u, +Y is image +v, +Z forward.
    # Translating the camera by +X shifts a static point's u DOWN, so to move
    # the egg toward center by (du, dv) we translate by the negated, range-
    # scaled error in the camera plane.
    dx_cam = -du * Z / fx
    dy_cam = -dv * Z / fy
    delta_cam = np.array([dx_cam, dy_cam, 0.0], dtype=np.float64) * float(gain)
    n = float(np.linalg.norm(delta_cam))
    if n > max_step_m and n > 1e-9:
        delta_cam *= max_step_m / n
    T_bf = arm.read_T_base_flange(ctx.redis, ctx.endeffector_transform_key)
    if T_bf is None:
        print(f"{label} no flange transform on Redis; skipping recenter.")
        return False
    R_base_cam = np.asarray(T_bf, dtype=np.float64)[:3, :3] @ T_FLANGE_CAMERA[:3, :3]
    delta_world = R_base_cam @ delta_cam
    cur = arm.read_current_ee_world(ctx.redis)
    if cur is None:
        print(f"{label} no current EE pose on Redis; skipping recenter.")
        return False
    cur_pos, cur_ori = cur
    ori = hold_orientation if hold_orientation is not None else cur_ori
    new_pos = cur_pos + delta_world
    print(
        f"{label} egg off-center (err={err_frac:.2f}); nudging EE by "
        f"world=[{delta_world[0]:+.3f}, {delta_world[1]:+.3f}, "
        f"{delta_world[2]:+.3f}] m to recenter."
    )
    arm.move_to(
        ctx,
        new_pos,
        ori,
        label=f"{label} nudge to center egg",
        tol_m=EGG_RECENTER_MOVE_TOL_M,
        timeout_s=move_timeout_s,
        gated=False,
    )
    return True


def find_grasp_pose_recentering(
    ctx: TaskContext,
    obj: Object,
    *,
    retries: int = 1,
    orientation_source: str = "fixed",
    depth_quantile: float = GRASP_DEPTH_QUANTILE,
    depth_patch_radius: int = GRASP_DEPTH_PATCH_RADIUS,
    hold_orientation: np.ndarray | None = None,
    max_recenter_moves: int = EGG_RECENTER_MAX_MOVES,
    center_tol_frac: float = EGG_RECENTER_TOL_FRAC,
    recenter_gain: float = EGG_RECENTER_GAIN,
    max_recenter_step_m: float = EGG_RECENTER_MAX_STEP_M,
    nominal_depth_m: float = EGG_RECENTER_NOMINAL_DEPTH_M,
) -> GraspPose:
    """:func:`find_grasp_pose` plus a "recenter the object, then retry" fallback.

    When an attempt fails to produce a valid grasp but Gemini still reported a
    pixel for the object, the flange is nudged so that pixel slides toward the
    frame center (orientation held at ``hold_orientation`` if given, else the
    live orientation) and the detection is retried. Recenter moves are capped at
    ``max_recenter_moves`` and DON'T consume the ``retries`` budget. A genuinely
    empty Gemini response, or a detection that keeps failing while the egg is
    already centered, falls back to the normal retry/abort path.
    """
    detection = _PROMPTS.get((obj, "grasp"), _DEFAULT_DETECTION)
    builder = _grasp_pose_builder(obj)
    client = _ensure_gemini(ctx)
    pipeline, align, depth_scale, intrinsics = ctx.realsense()

    radius = (
        detection.depth_patch_radius
        if detection.depth_patch_radius is not None
        else depth_patch_radius
    )
    min_depth_m = detection.min_depth_m

    prior_response_path = ctx.gemini_response_path
    resolved_path = resolve_grasp_response_path(ctx, obj)
    if resolved_path is not None:
        ctx.gemini_response_path = str(resolved_path)

    prompt = detection.resolve_prompt(obj)
    last_err: Exception | None = None
    miss_counter = [0]
    error_attempts = 0
    timeout_count = 0
    recenter_moves = 0

    try:
        while True:
            try:
                attempt = _grasp_pose_attempt(
                    ctx,
                    obj,
                    detection,
                    builder,
                    client=client,
                    pipeline=pipeline,
                    align=align,
                    depth_scale=depth_scale,
                    intrinsics=intrinsics,
                    prompt=prompt,
                    depth_quantile=depth_quantile,
                    depth_patch_radius=radius,
                    min_depth_m=min_depth_m,
                    orientation_source=orientation_source,
                    miss_counter=miss_counter,
                )
            except gp.GeminiTimeoutError as e:
                last_err = e
                timeout_count += 1
                print(f"[gemini:{obj.value}/grasp_pose] timed out: {e}")
                handle_gemini_timeout(obj.value, "grasp_pose", timeout_count)
                continue

            if attempt.error is None:
                return _save_success_overlay(ctx, attempt, obj, intrinsics)

            # Failed attempt. Save whatever overlay we have so the miss is
            # visible, then decide between a recenter move and a real retry.
            last_err = attempt.error
            if attempt.overlay is not None:
                _save_gemini_response(
                    ctx, attempt.overlay, attempt.depth_vis, attempt.patches
                )

            pixel = attempt.pixels[0] if attempt.pixels else None
            # ONLY recenter when the failure is a bad DEPTH / world-Z reading
            # (``_InvalidEggPoseError``). A fresh viewpoint can give a clean
            # depth, and a bogus depth can come from a near-centered pixel (e.g.
            # Gemini lands inside the held tongs), so force the nudge regardless
            # of how centered the pixel is. Every OTHER miss (no points, build
            # error) — even with the egg off-center — is NOT a reason to nudge;
            # it falls straight through to the normal retry budget.
            depth_off = isinstance(attempt.error, _InvalidEggPoseError)
            if depth_off and pixel is not None and recenter_moves < max_recenter_moves:
                moved = _recenter_ee_toward_pixel(
                    ctx,
                    pixel,
                    attempt.image_wh,
                    intrinsics,
                    hold_orientation=hold_orientation,
                    gain=recenter_gain,
                    max_step_m=max_recenter_step_m,
                    nominal_depth_m=nominal_depth_m,
                    center_tol_frac=-1.0,
                    label=f"[gemini:{obj.value}/recenter]",
                )
                if moved:
                    recenter_moves += 1
                    print(
                        f"[gemini:{obj.value}/grasp_pose] recenter move "
                        f"{recenter_moves}/{max_recenter_moves} after bad-depth "
                        f"miss: {attempt.error}"
                    )
                    continue

            error_attempts += 1
            timeout_count = 0
            print(
                f"[gemini:{obj.value}/grasp_pose] attempt {error_attempts} failed: "
                f"{attempt.error}"
            )
            if error_attempts > retries:
                raise RuntimeError(
                    f"gemini grasp pose for {obj.value} failed: {last_err}"
                ) from last_err
    finally:
        ctx.gemini_response_path = prior_response_path


def find_pour_target(ctx: TaskContext, obj: Object, *, retries: int = 1) -> np.ndarray:
    return _detect(ctx, obj, "pour", retries=retries)


def find_center(ctx: TaskContext, obj: Object, *, retries: int = 1) -> np.ndarray:
    return _detect(ctx, obj, "center", retries=retries)


def find_handle(ctx: TaskContext, obj: Object, *, retries: int = 1) -> np.ndarray:
    return _detect(ctx, obj, "handle", retries=retries)


def find_drop_target(ctx: TaskContext, obj: Object, *, retries: int = 1) -> np.ndarray:
    return _detect(ctx, obj, "drop", retries=retries)


def find_back_burner(ctx: TaskContext, *, retries: int = 1) -> np.ndarray:
    """Detect the back-burner white cross center (rightmost + uppermost) in world.

    Returns the world XYZ of the chosen burner's cross center. Callers that only
    need the burner's world X (e.g. to align the pan with it) can ignore Y/Z.
    """
    return _detect(ctx, Object.PAN, "back_burner", retries=retries)


# --- Bowl drop-center detection (crack / pour targets) ----------------------
# Radius (px) searched for a valid-depth pixel near Gemini's center point when
# the exact center pixel has no reading (dark interior). ~40 px at ~25-45 cm
# range covers several cm of world.
BOWL_DROP_DEPTH_PATCH_RADIUS = 40
# Reject near-field depth (gripper fingers, held cracker, glints) closer than
# this floor so they can't supply the center depth.
BOWL_DROP_MIN_DEPTH_M = 0.05


def find_bowl_drop_center(
    ctx: TaskContext,
    obj: Object,
    *,
    kind: str = "center",
    retries: int = 1,
) -> np.ndarray:
    """Detect a bowl's center and return the world drop point.

    Uses the single Gemini center point, deprojected at the pixel as close to
    the center as possible that has a valid depth (``prefer_gemini_pixel=True``):
    if the exact center pixel has a depth reading it's used directly, otherwise
    the NEAREST pixel to it with a valid depth (>= ``BOWL_DROP_MIN_DEPTH_M``,
    within ``BOWL_DROP_DEPTH_PATCH_RADIUS``) supplies the point. The caller adds
    its own +Z offset (the crack/pour point sits above this center).
    """
    detection = _PROMPTS.get((obj, kind), _DEFAULT_DETECTION)
    client = _ensure_gemini(ctx)
    _, _, _, intrinsics = ctx.realsense()

    prompt = detection.resolve_prompt(obj)
    last_err: Exception | None = None
    miss_counter = [0]
    error_attempts = 0
    timeout_count = 0

    while True:
        try:
            triple = _grab_rgbd_frame(
                ctx, miss_counter, label=f"[gemini:{obj.value}/{kind}]"
            )
            if triple is None:
                raise RuntimeError("no RGBD frame")
            color_bgr, depth_m, depth_vis = triple
            overlay, points_camera, patches = gp.query_color_depth_overlay(
                client,
                ctx.gemini_model,
                prompt,
                ctx.gemini_temperature,
                color_bgr,
                depth_m,
                intrinsics,
                BOWL_DROP_DEPTH_PATCH_RADIUS,
                prefer_gemini_pixel=True,
                min_depth_m=BOWL_DROP_MIN_DEPTH_M,
            )
            _save_gemini_response(ctx, overlay, depth_vis, patches)
            if not points_camera or points_camera[0] is None:
                raise ValueError("center point had no usable depth in range")
            center_world = _camera_to_world(ctx, points_camera[0])
            position = center_world + detection.offset(obj)
            print(f"[gemini:{obj.value}/{kind}] center_world={position.tolist()}")
            return position
        except gp.GeminiTimeoutError as e:
            last_err = e
            timeout_count += 1
            print(f"[gemini:{obj.value}/{kind}] timed out: {e}")
            handle_gemini_timeout(obj.value, kind, timeout_count)
            continue
        except Exception as e:  # noqa: BLE001 - surfaced via the retry budget
            last_err = e
            error_attempts += 1
            timeout_count = 0
            print(
                f"[gemini:{obj.value}/{kind}] attempt {error_attempts} failed: {e}"
            )
            if error_attempts > retries:
                raise RuntimeError(
                    f"gemini {kind} for {obj.value} failed: {last_err}"
                ) from last_err


def find_pan_center_radius(
    ctx: TaskContext,
    obj: Object = Object.PAN,
    *,
    kind: str = "center_rim",
    retries: int = 1,
) -> tuple[np.ndarray, float | None]:
    """Detect a pan's center AND a rim point; return ``(center_world, radius_m)``.

    Gemini returns two points (center first, then a point on the rim). Both are
    deprojected with the same valid-depth policy as :func:`find_bowl_drop_center`
    (``prefer_gemini_pixel=True``): the exact pixel when it has a depth reading,
    else the NEAREST pixel with a valid depth in range. The radius is the
    HORIZONTAL (world XY) distance between the center and rim points, which is
    robust to the rim sitting higher than the floor. ``radius_m`` is ``None``
    when the rim point could not be deprojected (center is still returned).
    """
    detection = _PROMPTS.get((obj, kind), _DEFAULT_DETECTION)
    client = _ensure_gemini(ctx)
    _, _, _, intrinsics = ctx.realsense()

    prompt = detection.resolve_prompt(obj)
    last_err: Exception | None = None
    miss_counter = [0]
    error_attempts = 0
    timeout_count = 0

    while True:
        try:
            triple = _grab_rgbd_frame(
                ctx, miss_counter, label=f"[gemini:{obj.value}/{kind}]"
            )
            if triple is None:
                raise RuntimeError("no RGBD frame")
            color_bgr, depth_m, depth_vis = triple
            overlay, points_camera, patches = gp.query_color_depth_overlay(
                client,
                ctx.gemini_model,
                prompt,
                ctx.gemini_temperature,
                color_bgr,
                depth_m,
                intrinsics,
                BOWL_DROP_DEPTH_PATCH_RADIUS,
                prefer_gemini_pixel=True,
                min_depth_m=BOWL_DROP_MIN_DEPTH_M,
            )
            _save_gemini_response(ctx, overlay, depth_vis, patches)
            if not points_camera or points_camera[0] is None:
                raise ValueError("center point had no usable depth in range")
            center_world = _camera_to_world(ctx, points_camera[0])
            position = center_world + detection.offset(obj)

            radius_m: float | None = None
            if len(points_camera) > 1 and points_camera[1] is not None:
                rim_world = _camera_to_world(ctx, points_camera[1])
                radius_m = float(
                    np.linalg.norm(rim_world[:2] - center_world[:2])
                )
                print(
                    f"[gemini:{obj.value}/{kind}] rim_world={rim_world.tolist()} "
                    f"radius={radius_m:.4f} m"
                )
            else:
                print(
                    f"[gemini:{obj.value}/{kind}] rim point had no usable depth "
                    f"in range — radius unavailable."
                )
            print(
                f"[gemini:{obj.value}/{kind}] center_world={position.tolist()}"
            )
            return position, radius_m
        except gp.GeminiTimeoutError as e:
            last_err = e
            timeout_count += 1
            print(f"[gemini:{obj.value}/{kind}] timed out: {e}")
            handle_gemini_timeout(obj.value, kind, timeout_count)
            continue
        except Exception as e:  # noqa: BLE001 - surfaced via the retry budget
            last_err = e
            error_attempts += 1
            timeout_count = 0
            print(
                f"[gemini:{obj.value}/{kind}] attempt {error_attempts} failed: {e}"
            )
            if error_attempts > retries:
                raise RuntimeError(
                    f"gemini {kind} for {obj.value} failed: {last_err}"
                ) from last_err
