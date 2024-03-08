#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
import numpy as np

import pyglet
from matplotlib import pylab as plt
import matplotlib.backends.backend_agg as agg

DPI = 96
WIDTH = 800
HEIGHT = 600
RESOLUTION = (WIDTH, HEIGHT)

RENDERING_MODES = {
    'human': lambda: HumanRenderer,
    'rgb_array': lambda: RgbRenderer,
}


class RgbRenderer(object):
    def __init__(self, resolution=RESOLUTION, dpi=DPI):
        self.resolution = resolution
        self.dpi = dpi

    def render(self, state):
        width = self.resolution[0] / self.dpi
        height = self.resolution[1] / self.dpi
        fig = plt.figure(0, figsize=(width, height), dpi=self.dpi)

        fig.tight_layout()
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        plt.close(fig)

        # return np.frombuffer(raw_data, dtype=np.uint8).reshape(
        #     (size[0], size[1], 3)
        # )


class HumanRenderer(RgbRenderer, pyglet.window.Window):
    def __init__(self, resolution=RESOLUTION, dpi=DPI):
        super().__init__(resolution, dpi)

        self.rendering = None
        width, height = resolution
        self.window = pyglet.window.Window(width, height, visible=False)
        self.window.set_caption('Offloading State')
        self.window.set_visible()
        self.window.on_draw = self.on_draw

    def on_draw(self):
        self.window.clear()
        if self.rendering is not None:
            height, width, _ = self.rendering.shape
            img = pyglet.image.ImageData(
                height,
                width,
                'RGB',
                self.rendering.data.tobytes(),
                -3 * height,
            )

            img.blit(0, 0)

    def render(self, state):
        self.rendering = super().render(state)

        pyglet.clock.tick()
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.dispatch_event('on_draw')
        self.window.flip()

        return self.rendering


class OffloadingRenderer(object):
    def __init__(self, mode, *args, **kwargs):
        if mode not in RENDERING_MODES:
            raise RuntimeError(f'Unsupported requested mode {mode}')
        self.renderer = RENDERING_MODES[mode]()(*args, **kwargs)

    def render(self, state):
        return self.renderer.render(state)
