"""Matplotlib-based live renderer for the plate sim."""

from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Renderer:
    def __init__(self, plate_radius: float = 100.0):
        self.r = plate_radius
        plt.ion()
        self.fig, (self.ax, self.ax_t) = plt.subplots(1, 2, figsize=(10, 5))

        self.ax.set_xlim(-1.2 * self.r, 1.2 * self.r)
        self.ax.set_ylim(-1.2 * self.r, 1.2 * self.r)
        self.ax.set_aspect("equal")
        self.ax.add_patch(patches.Circle((0, 0), self.r, fill=False, lw=2))
        self.ax.axhline(0, color="gray", lw=0.5)
        self.ax.axvline(0, color="gray", lw=0.5)
        (self.ball,) = self.ax.plot([], [], "o", color="orange", markersize=12)
        (self.target,) = self.ax.plot([0], [0], "x", color="green", markersize=10)
        self.ax.set_title("Plate (top view)")

        self.ax_t.set_title("Position vs time")
        self.ax_t.set_xlabel("t (s)")
        self.ax_t.set_ylabel("mm")
        (self.line_x,) = self.ax_t.plot([], [], label="x")
        (self.line_y,) = self.ax_t.plot([], [], label="y")
        self.ax_t.legend()
        self.ax_t.set_ylim(-self.r, self.r)
        self.ts: list[float] = []
        self.xs: list[float] = []
        self.ys: list[float] = []

    def update(self, t: float, x: float, y: float) -> None:
        self.ball.set_data([x], [y])
        self.ts.append(t)
        self.xs.append(x)
        self.ys.append(y)
        # keep last ~10 s
        while self.ts and self.ts[-1] - self.ts[0] > 10:
            self.ts.pop(0)
            self.xs.pop(0)
            self.ys.pop(0)
        self.line_x.set_data(self.ts, self.xs)
        self.line_y.set_data(self.ts, self.ys)
        if self.ts:
            self.ax_t.set_xlim(self.ts[0], max(self.ts[-1], self.ts[0] + 1))
        self.fig.canvas.draw_idle()
        plt.pause(0.001)
