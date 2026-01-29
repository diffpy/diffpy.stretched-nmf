import matplotlib.pyplot as plt
import numpy as np


class SNMFPlotter:
    def __init__(self, figsize=(12, 4)):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=figsize)
        titles = [
            "Components",
            "Weights (rows as series)",
            "Stretch (rows as series)",
        ]
        for ax, t in zip(self.axes, titles):
            ax.set_title(t)
        self.lines = {"components": [], "weights": [], "stretch": []}
        self._layout_done = False
        plt.show()

    def _ensure_lines(self, ax, key, n_series):
        cur = self.lines[key]
        if len(cur) != n_series:
            ax.cla()
            ax.set_title(ax.get_title())
            self.lines[key] = [ax.plot([], [])[0] for _ in range(n_series)]
        return self.lines[key]

    def _update_series(self, ax, key, data_2d):
        # Expect rows = separate series for components
        data_2d = np.atleast_2d(data_2d)
        n_series, n_pts = data_2d.shape
        lines = self._ensure_lines(ax, key, n_series)
        x = np.arange(n_pts)
        for ln, y in zip(lines, data_2d):
            ln.set_data(x, y)
        ax.relim()
        ax.autoscale_view()

    def update(self, components, weights, stretch, update_tag=None):
        # Components: transpose before plotting
        c = np.asarray(components).T
        self._update_series(self.axes[0], "components", c)

        w = np.asarray(weights)
        self._update_series(self.axes[1], "weights", w)

        s = np.asarray(stretch)
        self._update_series(self.axes[2], "stretch", s)

        if update_tag is not None:
            self.fig.suptitle(f"Updated: {update_tag}", fontsize=14)

        if not self._layout_done:
            self.fig.tight_layout()
            self._layout_done = True

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
