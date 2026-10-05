#!/usr/bin/env python3
"""Interactive design tool: sliders for L/L1/L2/L3/h + camera params + the
commanded tilt (theta/phi) -> live view of the resulting bearing (servo)
angles and how much of the plate the camera actually sees.

This exists because tuning the rig by re-flashing main.py and watching the
real servos hunt for their mechanical stops is slow and a little risky.
Everything here reuses the actual inverse-kinematics functions from
src/control/inverse_kinematics.py (not a re-derivation), so a config that
looks good here matches what incline_board() will really do.

Run from the repo root:
    .venv/bin/python -m src.tools.geometry_explorer   (Linux/macOS)
    .venv\\Scripts\\python -m src.tools.geometry_explorer   (Windows)

What each panel shows
----------------------
Top-left    one arm's 2-link linkage (motor -> elbow -> plate joint), seen
            edge-on in its own vertical plane, flat vs. currently tilted.
Top-right   top-down layout: plate disc, joint circle (radius L), motor
            circle (radius L3), and where the 3 plate joints actually land
            (they drift slightly inward as tilt increases - the model does
            not hold their horizontal position exactly fixed).
Bottom-left camera side view: the FOV cone from the camera (mounted at
            camera_z, looking up through the base) up to the plate, with
            the visible footprint compared against the physical plate.
Bottom-right bearing angle vs. tilt magnitude, swept for all 3 arms, with
            the servo's mechanical stops (0 deg / 180 deg raw) shaded out -
            this is the plot that tells you if a geometry choice runs out
            of servo travel before it runs out of desired tilt.

theta/phi vs. "the bearing angle"
----------------------------------
theta is the plate tilt magnitude you want to command (like max_tilt_rad in
Controller), phi is which way it leans. "The angle of the bearing" is the
resulting motor-shaft angle IK produces for that tilt - not something you
set directly, since it's a consequence of geometry. What IS directly
adjustable is bearing_offset: where that servo's mechanical neutral
(offset_rad in calibration_servo.py) sits inside its raw travel, which is
what actually determines how much headroom to the stops a given tilt range
has. Move it to preview calibration choices, the same object main.py's
OFFSET_ARM_* constants control.

Comparing two configurations
------------------------------
Click "Snapshot A" / "Snapshot B" to freeze the current sliders; both get
overlaid on every plot (solid = current, dashed = A, dotted = B) plus a
numeric diff in the text panel. "Clear A/B" drops them.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# control/arm.py needs src/ importable as the top-level package root (it
# does `from Actuator.Servo import Servo`, not `from src.Actuator...`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _stub_missing_hardware_modules() -> None:
    """Actuator/Servo.py imports Blinka's board/busio/adafruit_pca9685 to
    talk to the real PCA9685 over I2C. This tool never touches a servo -
    arms are built with servo=None - but importing control/arm.py (needed
    to reuse the real IK code) drags Servo.py in unconditionally. On a dev
    machine without the Pi's I2C stack those imports fail outright, so stub
    them here; skipped entirely wherever the real libraries ARE installed
    (i.e. running this directly on the Pi still uses the real ones).
    """
    stubs = {
        "busio": {"I2C": lambda *a, **k: None},
        "board": {"SCL": None, "SDA": None},
        "adafruit_pca9685": {"PCA9685": type("PCA9685", (), {})},
    }
    for name, attrs in stubs.items():
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            fake = types.ModuleType(name)
            for attr, val in attrs.items():
                setattr(fake, attr, val)
            sys.modules[name] = fake


_stub_missing_hardware_modules()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
import matplotlib.patches as patches

from control.arm import Arm
from control.inverse_kinematics import (
    solve_end_effector_positions,
    solve_bearing_positions,
    solve_servo_angles,
)
from utils import PlateOrientation

# --------------------------------------------------------------------------
# Defaults - taken from main.py's current geometry, so the tool opens on the
# rig's actual configuration rather than an arbitrary guess.
# --------------------------------------------------------------------------

# Regulation table-tennis ball (ITTF spec): 40mm diameter. Coverage is
# checked against plate_diameter + this, not the bare plate - the camera
# needs to see the whole ball (not just its center) even when it's rolled
# all the way to the plate's edge, and a bit of slack around that for lens
# distortion near the frame border.
BALL_DIAMETER_MM = 40.0

AZIMUTHS_DEG = (0.0, 120.0, 240.0)  # arm layout; matches THETAS in tests

# Pi Camera Module 3 published lens FOV (Raspberry Pi datasheet). "Windowed"
# capture modes (e.g. main.py's 640x480 preview) can crop the sensor and see
# LESS than this full-lens FOV - these are an upper bound, not a guarantee
# for every capture mode.
CAMERA_PRESETS = {
    "Standard (Cam3)": (66.0, 41.0, 75.0),   # fov_h, fov_v, fov_diag deg
    "Wide (Cam3 Wide)": (102.0, 67.0, 120.0),
}
CAMERA_RES = (640, 480)  # matches camera_width/height in main.py
DEFAULT_CAMERA_PRESET = "Wide (Cam3 Wide)"  # Standard's FOV cannot cover a
# 240mm plate from anywhere near the rig without an impractically deep
# enclosure - Wide is the realistic starting point.

DEFAULTS = dict(
    L=140.0, L1=100.0, L2=130.0, L3=90.0, h=150.0,
    theta_deg=8.0, phi_deg=0.0, bearing_offset_deg=45.0,
    plate_diameter=280.0,
)

# The camera is embedded right at the base/motor plane (origin of the rig's
# frame), looking straight up - so camera_z = 0 and h (plate height) IS the
# camera-to-plate distance directly, no separate offset to account for.
#
# NOTE: at this h/plate_diameter, coverage is NOT sufficient (~198.6mm
# visible vertically vs. 320mm needed for the full plate+ball) - confirmed
# and accepted deliberately in favor of the larger 280mm plate. Reduce L
# toward ~65-70mm (diameter ~130-140mm) if full-plate camera coverage is
# ever needed instead.
DEFAULTS["camera_z"] = 0.0

SLIDER_SPECS = [
    # (key, label, min, max)
    ("L", "L  plate joint radius (mm)", 40.0, 200.0),
    ("L1", "L1  distal link (mm)", 20.0, 200.0),
    ("L2", "L2  proximal link (mm)", 20.0, 200.0),
    ("L3", "L3  base/motor radius (mm)", 20.0, 200.0),
    ("h", "h  plate neutral height (mm)", 10.0, 200.0),
    ("theta_deg", "theta  tilt magnitude (deg)", -30.0, 30.0),
    ("phi_deg", "phi  tilt direction (deg)", 0.0, 360.0),
    ("bearing_offset_deg", "bearing neutral offset (deg)", -90.0, 90.0),
    ("plate_diameter", "physical plate diameter (mm)", 60.0, 500.0),
    ("camera_z", "camera mount height (mm, below plate = negative)", -300.0, 150.0),
]


def build_arms(L1: float, L2: float, L3: float) -> np.ndarray:
    return np.array([
        Arm(servo=None, L1=L1, L2=L2, L3=L3, mount_angle_rad=np.radians(az))
        for az in AZIMUTHS_DEG
    ])


class State:
    """Everything derived from one slider configuration."""

    def __init__(self, p: dict):
        self.p = dict(p)
        L, L1, L2, L3, h = p["L"], p["L1"], p["L2"], p["L3"], p["h"]
        theta = np.radians(p["theta_deg"])
        phi = np.radians(p["phi_deg"])

        arms = build_arms(L1, L2, L3)
        self.arms = arms

        n_flat = (0.0, 0.0, 1.0)
        n_tilt = (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))

        with np.errstate(all="ignore"):
            self.ee_flat = solve_end_effector_positions(PlateOrientation(n=n_flat, h=h), arms, L)
            self.bp_flat = solve_bearing_positions(PlateOrientation(n=n_flat, h=h), arms, self.ee_flat, L)
            self.q_flat = solve_servo_angles(arms, self.bp_flat)  # rad, same for all 3 by symmetry

            self.ee_tilt = solve_end_effector_positions(PlateOrientation(n=n_tilt, h=h), arms, L)
            self.bp_tilt = solve_bearing_positions(PlateOrientation(n=n_tilt, h=h), arms, self.ee_tilt, L)
            self.q_tilt = solve_servo_angles(arms, self.bp_tilt)

        bearing_offset = np.radians(p["bearing_offset_deg"])
        delta = self.q_tilt - self.q_flat
        raw_command = bearing_offset + delta  # rad; mirrors Actuation's differential frame
        self.raw_command = raw_command
        self.raw_clipped = np.clip(raw_command, 0.0, np.pi)
        self.saturated = ~np.isclose(raw_command, self.raw_clipped)
        self.reachable = np.isfinite(self.q_flat) & np.isfinite(self.q_tilt)
        self.margin_deg = np.degrees(np.minimum(self.raw_clipped, np.pi - self.raw_clipped))

        # --- camera coverage ---
        fov_h, fov_v, fov_diag = p["fov"]
        dist = h - p["camera_z"]
        self.cam_dist = dist
        if dist > 0:
            self.footprint_h = 2 * dist * np.tan(np.radians(fov_h) / 2)
            self.footprint_v = 2 * dist * np.tan(np.radians(fov_v) / 2)
            self.footprint_diag = 2 * dist * np.tan(np.radians(fov_diag) / 2)
        else:
            self.footprint_h = self.footprint_v = self.footprint_diag = float("nan")
        self.mm_per_px_h = self.footprint_h / CAMERA_RES[0]
        self.mm_per_px_v = self.footprint_v / CAMERA_RES[1]
        binding = min(self.footprint_h, self.footprint_v)
        # Require the frame to fit the plate PLUS a full ball-diameter of
        # slack, so a ball rolled to the plate's edge is still entirely
        # visible (and not just clipped in by its center point).
        self.required_span = p["plate_diameter"] + BALL_DIAMETER_MM
        self.coverage_margin = binding - self.required_span
        self.coverage_ok = self.coverage_margin > 0

        # --- single go/no-go verdict for this configuration ---
        reasons = []
        if not np.all(self.reachable):
            bad = [i + 1 for i in range(3) if not self.reachable[i]]
            reasons.append(f"arm{bad} unreachable (IK has no solution)")
        if np.any(self.saturated):
            bad = [i + 1 for i in range(3) if self.saturated[i]]
            reasons.append(f"arm{bad} past the servo's mechanical stop")
        if not self.coverage_ok:
            reasons.append(f"camera does not cover plate+ball ({self.coverage_margin:+.0f} mm, "
                            f"need {self.required_span:.0f} mm, mount lower/wider lens)")
        self.reasons = reasons
        self.ok = len(reasons) == 0

    def bearing_sweep(self, phi_deg: float, n: int = 61):
        """raw_command (deg) for all 3 arms across theta in [0, 30 deg] at a
        fixed tilt direction - the "run out of servo before run out of
        tilt?" curve."""
        L, h = self.p["L"], self.p["h"]
        arms = self.arms
        phi = np.radians(phi_deg)
        bearing_offset = np.radians(self.p["bearing_offset_deg"])
        thetas_deg = np.linspace(0.0, 30.0, n)
        out = np.empty((n, len(arms)))
        with np.errstate(all="ignore"):
            for i, td in enumerate(thetas_deg):
                theta = np.radians(td)
                n_tilt = (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
                orient = PlateOrientation(n=n_tilt, h=h)
                ee = solve_end_effector_positions(orient, arms, L)
                bp = solve_bearing_positions(orient, arms, ee, L)
                q = solve_servo_angles(arms, bp)
                out[i] = np.degrees(bearing_offset + (q - self.q_flat))
        return thetas_deg, out


def main() -> None:
    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title("Ball-balancing robot: geometry explorer")

    ax_arm = fig.add_axes([0.36, 0.55, 0.28, 0.40])
    ax_top = fig.add_axes([0.68, 0.55, 0.28, 0.40])
    ax_cam = fig.add_axes([0.36, 0.08, 0.28, 0.40])
    ax_sweep = fig.add_axes([0.68, 0.08, 0.28, 0.40])

    # Pushed flush to the left edge, and given a big bold go/no-go marker
    # instead of a wall of numbers - this is the panel people said they
    # could not read.
    text_ax = fig.add_axes([0.0, 0.02, 0.34, 0.30])
    text_ax.axis("off")
    marker_text = text_ax.text(0.5, 0.90, "", ha="center", va="center",
                                fontsize=24, fontweight="bold",
                                transform=text_ax.transAxes)
    detail_text = text_ax.text(0.03, 0.68, "", va="top", ha="left",
                                family="monospace", fontsize=9.5,
                                transform=text_ax.transAxes)

    params = dict(DEFAULTS)
    params["fov"] = CAMERA_PRESETS[DEFAULT_CAMERA_PRESET]
    snapshots: dict[str, State | None] = {"A": None, "B": None}

    # Slider axes start well clear of the figure's left edge - the default
    # Slider label is drawn just outside the axes, and with the axes
    # starting near x=0 the longer labels (e.g. "bearing neutral offset
    # (deg)") had nowhere to render and were clipped off-canvas entirely.
    sliders: dict[str, Slider] = {}
    top = 0.97
    step = 0.031
    for i, (key, label, lo, hi) in enumerate(SLIDER_SPECS):
        ax = fig.add_axes([0.15, top - i * step, 0.18, 0.018])
        sliders[key] = Slider(ax, label, lo, hi, valinit=DEFAULTS[key], valstep=0.5)
        sliders[key].label.set_fontsize(8.5)
        sliders[key].valtext.set_fontsize(8.5)

    radio_ax = fig.add_axes([0.04, 0.40, 0.26, 0.09])
    radio = RadioButtons(radio_ax, list(CAMERA_PRESETS.keys()),
                          active=list(CAMERA_PRESETS.keys()).index(DEFAULT_CAMERA_PRESET))

    btn_axA = fig.add_axes([0.04, 0.355, 0.075, 0.035])
    btn_axB = fig.add_axes([0.125, 0.355, 0.075, 0.035])
    btn_axClearA = fig.add_axes([0.04, 0.315, 0.075, 0.035])
    btn_axClearB = fig.add_axes([0.125, 0.315, 0.075, 0.035])
    btn_axReset = fig.add_axes([0.205, 0.355, 0.08, 0.035])
    btn_snap_a = Button(btn_axA, "Snapshot A")
    btn_snap_b = Button(btn_axB, "Snapshot B")
    btn_clear_a = Button(btn_axClearA, "Clear A")
    btn_clear_b = Button(btn_axClearB, "Clear B")
    btn_reset = Button(btn_axReset, "Reset")

    STYLE = {"cur": dict(color="tab:blue", ls="-"), "A": dict(color="tab:orange", ls="--"),
             "B": dict(color="tab:green", ls=":")}

    def current_params() -> dict:
        p = {key: sliders[key].val for key, *_ in SLIDER_SPECS}
        p["fov"] = params["fov"]
        return p

    def draw_arm_panel(states: dict[str, State]):
        ax_arm.clear()
        ax_arm.set_title("Arm 1 (azimuth 0 deg): motor -> elbow -> joint")
        ax_arm.set_xlabel("radial distance from center (mm)")
        ax_arm.set_ylabel("height (mm)")
        ax_arm.set_aspect("equal")
        any_warning = False
        for tag, st in states.items():
            if st is None:
                continue
            if not st.reachable[0]:
                any_warning = True
                continue  # NaN geometry - nothing to draw for this arm
            style = dict(STYLE[tag])
            L3 = st.p["L3"]
            motor = (L3, 0.0)
            elbow = (st.bp_tilt[0].x, st.bp_tilt[0].z)
            az = np.radians(AZIMUTHS_DEG[0])
            joint = (st.ee_tilt[0].x * np.cos(az) + st.ee_tilt[0].y * np.sin(az), st.ee_tilt[0].z)
            xs = [motor[0], elbow[0], joint[0]]
            ys = [motor[1], elbow[1], joint[1]]
            label = "current" if tag == "cur" else tag
            if st.saturated[0]:
                # Raw IK doesn't know about the servo's mechanical stop - this
                # is the *unclamped* target pose; the real arm would stop at
                # elbow height 0 (q=0 deg) or L2 (q=180 deg) instead.
                style["color"] = "tab:red"
                label += " (unclamped - past stop)"
                any_warning = True
            ax_arm.plot(xs, ys, marker="o", label=label, **style)
        ax_arm.plot(0, 0, "k+", markersize=10)  # plate center reference
        ax_arm.axhline(0.0, color="gray", lw=0.6, ls=":")  # z=0: elbow can never go below this
        if any_warning:
            ax_arm.text(0.02, 0.02, "red/skipped = past this servo's mechanical travel",
                        transform=ax_arm.transAxes, fontsize=7, color="tab:red")
        ax_arm.legend(loc="upper right", fontsize=7)
        ax_arm.grid(alpha=0.3)

    def draw_top_panel(states: dict[str, State]):
        ax_top.clear()
        ax_top.set_title("Top-down layout")
        ax_top.set_aspect("equal")
        st = states["cur"]
        L, L3, plate_d = st.p["L"], st.p["L3"], st.p["plate_diameter"]
        ax_top.add_patch(patches.Circle((0, 0), plate_d / 2, fill=False, color="gray", lw=2, label="physical plate"))
        ax_top.add_patch(patches.Circle((0, 0), L, fill=False, color="tab:blue", ls="--", lw=1, label="joint circle (L)"))
        ax_top.add_patch(patches.Circle((0, 0), L3, fill=False, color="tab:red", ls="--", lw=1, label="motor circle (L3)"))
        for tag, s in states.items():
            if s is None:
                continue
            style = STYLE[tag]
            for i, az in enumerate(AZIMUTHS_DEG):
                azr = np.radians(az)
                ax_top.plot(s.p["L3"] * np.cos(azr), s.p["L3"] * np.sin(azr), "s", color=style["color"], markersize=5)
                ax_top.plot(s.ee_tilt[i].x, s.ee_tilt[i].y, "o", color=style["color"], markersize=6)
        phi = np.radians(st.p["phi_deg"])
        r = plate_d / 2
        ax_top.annotate("", xy=(r * np.cos(phi), r * np.sin(phi)), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        lim = plate_d / 2 * 1.25
        ax_top.set_xlim(-lim, lim)
        ax_top.set_ylim(-lim, lim)
        ax_top.legend(loc="upper right", fontsize=7)
        ax_top.grid(alpha=0.3)

    def draw_camera_panel(states: dict[str, State]):
        ax_cam.clear()
        ax_cam.set_title("Camera FOV (vertical cross-section)")
        ax_cam.set_xlabel("horizontal (mm)")
        ax_cam.set_ylabel("height (mm)")
        st = states["cur"]
        cam_z = st.p["camera_z"]
        h = st.p["h"]
        plate_d = st.p["plate_diameter"]
        fov_v = params["fov"][1]
        ax_cam.plot(0, cam_z, "^", color="black", markersize=10, label="camera")
        if st.cam_dist > 0:
            half = np.tan(np.radians(fov_v) / 2) * st.cam_dist
            ax_cam.plot([0, -half], [cam_z, h], color="tab:blue", lw=1)
            ax_cam.plot([0, half], [cam_z, h], color="tab:blue", lw=1)
        ax_cam.plot([-plate_d / 2, plate_d / 2], [h, h], color="gray", lw=4, solid_capstyle="butt", label="plate (physical)")
        req = st.required_span / 2
        ax_cam.plot([-req, req], [h - 4, h - 4], color="tab:orange", lw=2, ls=":",
                    label=f"needed (plate+ball, {st.required_span:.0f} mm)")
        if np.isfinite(st.footprint_v):
            fv = st.footprint_v / 2
            color = "tab:green" if st.coverage_ok else "tab:red"
            ax_cam.plot([-fv, fv], [h + 3, h + 3], color=color, lw=3, label="visible footprint (V)")
        span = max(plate_d, st.required_span, st.footprint_v if np.isfinite(st.footprint_v) else plate_d) * 0.7
        ax_cam.set_xlim(-span, span)
        ax_cam.set_ylim(min(cam_z, 0) - 10, h + 20)
        ax_cam.legend(loc="upper right", fontsize=7)
        ax_cam.grid(alpha=0.3)

    def draw_sweep_panel(states: dict[str, State]):
        ax_sweep.clear()
        ax_sweep.set_title("Bearing angle vs. tilt (sweep at current phi)")
        ax_sweep.set_xlabel("theta - tilt magnitude (deg)")
        ax_sweep.set_ylabel("commanded bearing angle, raw frame (deg)")
        ax_sweep.axhspan(-20, 0, color="red", alpha=0.15)
        ax_sweep.axhspan(180, 200, color="red", alpha=0.15)
        for tag, st in states.items():
            if st is None:
                continue
            style = STYLE[tag]
            thetas_deg, sweep_deg = st.bearing_sweep(st.p["phi_deg"])
            for arm_i in range(sweep_deg.shape[1]):
                ax_sweep.plot(thetas_deg, sweep_deg[:, arm_i], color=style["color"],
                              ls=style["ls"], lw=1.5 if arm_i == 0 else 0.8,
                              label=(f"{tag} arm{arm_i+1}" if arm_i == 0 else None))
        ax_sweep.axvline(st_cur_theta[0], color="black", lw=0.8, ls=":")
        ax_sweep.set_ylim(-10, 190)
        ax_sweep.legend(loc="upper right", fontsize=7)
        ax_sweep.grid(alpha=0.3)

    def format_verdict(st: State) -> tuple[str, str]:
        """One word, one color: is the CURRENT geometry usable or not."""
        if st.ok:
            return "GEOMETRY OK", "tab:green"
        return "GEOMETRY NOT OK", "tab:red"

    def format_details(st: State) -> str:
        lines = [
            f"min bearing margin to stop:      {np.min(st.margin_deg):5.1f} deg",
            f"camera coverage margin:          {st.coverage_margin:+6.1f} mm",
            f"  (needs plate {st.p['plate_diameter']:.0f} + ball {BALL_DIAMETER_MM:.0f} "
            f"= {st.required_span:.0f} mm)",
        ]
        if st.reasons:
            lines.append("")
            lines.append("why not:")
            for reason in st.reasons:
                lines.append(f"  - {reason}")
        return "\n".join(lines)

    st_cur_theta = [DEFAULTS["theta_deg"]]  # mutable box so draw_sweep_panel sees the live value

    def redraw(_=None):
        cur_p = current_params()
        st_cur_theta[0] = cur_p["theta_deg"]
        states = {"cur": State(cur_p), "A": snapshots["A"], "B": snapshots["B"]}
        draw_arm_panel(states)
        draw_top_panel(states)
        draw_camera_panel(states)
        draw_sweep_panel(states)
        verdict, color = format_verdict(states["cur"])
        marker_text.set_text(verdict)
        marker_text.set_color(color)
        detail_text.set_text(format_details(states["cur"]))
        fig.canvas.draw_idle()

    def on_radio(label):
        params["fov"] = CAMERA_PRESETS[label]
        redraw()

    def on_snap_a(_):
        snapshots["A"] = State(current_params())
        redraw()

    def on_snap_b(_):
        snapshots["B"] = State(current_params())
        redraw()

    def on_clear_a(_):
        snapshots["A"] = None
        redraw()

    def on_clear_b(_):
        snapshots["B"] = None
        redraw()

    def on_reset(_):
        for key, *_ in SLIDER_SPECS:
            sliders[key].set_val(DEFAULTS[key])
        redraw()

    for s in sliders.values():
        s.on_changed(redraw)
    radio.on_clicked(on_radio)
    btn_snap_a.on_clicked(on_snap_a)
    btn_snap_b.on_clicked(on_snap_b)
    btn_clear_a.on_clicked(on_clear_a)
    btn_clear_b.on_clicked(on_clear_b)
    btn_reset.on_clicked(on_reset)

    redraw()
    plt.show()


if __name__ == "__main__":
    main()
