
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def set_axes(axes: Axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator:
    def __init__(self,
                 xlabel=None,
                 ylabel=None,
                 legend=None,
                 xlim=None,
                 ylim=None,
                 xscale='linear',
                 yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'),
                 nrows=1,
                 ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [
                self.axes,
            ]
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        plt.ion()
        self.fig.show()

    def add(self, x, y):
        # 向圖表中新增多個資料點
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

        self.axes[0].relim()  # recompute the data limits
        self.axes[0].autoscale_view()  # automatic axis scaling
        self.fig.canvas.flush_events()