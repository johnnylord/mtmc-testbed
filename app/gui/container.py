import threading
from functools import wraps
from abc import ABC, abstractmethod

import cv2
import numpy as np


__all__ = [ "Container", "check_ready" ]


def check_ready(method):
    """Decorator for Container's methods"""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self._ready:
            raise RuntimeError("Container is not ready yet, call container.ready() first")
        return method(self, *args, **kwargs)
    return wrapper


class Container:
    """Display wrapper for panels

    Arguments:
        nrows (int, optional): number of rows in the container, default 2
        ncols (int, optional): number of columns in the container, default 2
        size (tuple, optiona): 2D tuple specified container size, default (1920, 1080)
    """
    def __init__(self, nrows=2, ncols=2, size=(1920, 1080), **kwargs):
        # Container metadata
        # =================================================
        self.nrows = nrows
        self.ncols = ncols
        self.size = size

        # Panel information
        # =================================================
        self.panels = []
        self.panel_offsets = []
        for i in range(nrows):
            for j in range(ncols):
                y_offset = (size[1]//nrows)*i
                x_offset = (size[0]//ncols)*j
                self.panel_offsets.append((x_offset, y_offset))

        # Dynamic information (assign value later)
        # =================================================
        self.focus_panel = None
        self.container_cache = None
        # Following properties will be set after ready get called
        self.stype = None
        self.fps = None
        self.total_frames = None
        self._ready = False

    # Proxy methods for internel panels
    # =====================================================
    @check_ready
    def start(self):
        _ = [ p.start() for p in self.panels ]
        return self

    @check_ready
    def pause(self):
        _ = [ p.pause() for p in self.panels ]
        return self

    @check_ready
    def stop(self):
        _ = [ p.stop() for p in self.panels ]
        return self

    @check_ready
    def jump(self, index):
        """Spawn multiple threads to jump to target frame with specified index"""
        if self.is_pause():
            def jump_task(panel):
                panel.jump(index)

            jump_threads = [ threading.Thread(target=jump_task, args=(p,), daemon=True)
                            for p in self.panels ]
            _ = [ thread.start() for thread in jump_threads ]
            _ = [ thread.join() for thread in jump_threads ]

        return self

    @check_ready
    def is_start(self):
        return self.panels[0].is_start()

    @check_ready
    def is_pause(self):
        return self.panels[0].is_pause()

    @check_ready
    def is_stop(self):
        return self.panels[0].is_stop()

    # Container methods
    # ======================================================
    @check_ready
    def find_panel(self, pos):
        """Return the target panel under the specified 'pos' position

        Arguments:
            pos (list or tuple): a sequence of length 2 specified (x, y) coordinate
        """
        target_panel = None
        for panel, panel_offset in zip(self.panels, self.panel_offsets):
            if (1
                and panel_offset[0] <= pos[0] <= panel_offset[0] + panel.size[0]
                and panel_offset[1] <= pos[1] <= panel_offset[1] + panel.size[1]
            ):
                target_panel = panel
                break

        return target_panel

    @check_ready
    def rerender(self, panel_contents):
        # Render panels
        # ==============================================================
        for panel_content, panel_offset in zip(panel_contents, self.panel_offsets):
            # Copy panel to its slot in the container
            x_start = panel_offset[0]
            y_start = panel_offset[1]
            x_end = x_start + self.panels[0].size[0]
            y_end = y_start + self.panels[0].size[1]
            self.container_cache[y_start:y_end, x_start:x_end, :] = panel_content['panel_frame']

        # Cover container with the panel being focus
        # ====================================================================
        if self.focus_panel is not None:
            index = self.panels.index(self.focus_panel)
            self.container_cache = cv2.resize(panel_contents[index]['panel_frame'], self.size)

        content = { 'fid': panel_content['fid'],
                    'container_frame': self.container_cache,
                    'panel_contents': panel_contents }
        return content

    @check_ready
    def render(self):
        """Return container content of type dict"""
        # Container canvas
        container_width = self.size[0]
        container_height = self.size[1]
        container_frame = np.zeros((container_height, container_width, 3), dtype=np.uint8)

        # Render panels
        panel_contents = []
        for panel, panel_offset in zip(self.panels, self.panel_offsets):
            panel_content = panel.render()
            panel_contents.append(panel_content)

            # Copy panel to its slot in the container
            x_start = panel_offset[0]
            y_start = panel_offset[1]
            x_end = x_start + panel.size[0]
            y_end = y_start + panel.size[1]
            container_frame[y_start:y_end, x_start:x_end, :] = panel_content['panel_frame']

        # Cover container with the panel being focus
        if self.focus_panel is not None:
            index = self.panels.index(self.focus_panel)
            container_frame = cv2.resize(panel_contents[index]['panel_frame'], self.size)

        self.container_cache = container_frame.copy()

        content = { 'fid': panel_content['fid'],
                    'container_frame': container_frame,
                    'panel_contents': panel_contents }
        return content

    def add_panel(self, panel):
        """Add panel to the container"""
        if len(self.panels) < self.nrows*self.ncols:
            # Reset panel size
            panel_width = self.size[0]//self.ncols
            panel_height = self.size[1]//self.nrows
            panel.responsive((panel_width, panel_height))
            self.panels.append(panel)

        else:
            panel_limit = self.nrows*self.ncols
            raise RuntimeError("You can only register '{}' panels".format(panel_limit))

    def ready(self):
        """Make container ready for postprocessing"""
        if len(self.panels) == 0:
            raise RuntimeError("Container needs at least one panel to get ready")

        # Check unified source type
        for p in self.panels:
            if self.stype is None:
                self.stype = p.stype
                continue
            if self.stype != p.stype:
                raise RuntimeError("Media sources should all be same type")

        # Align total frames between panaels
        align_frames = np.array([ p.total_frames for p in self.panels ]).min()
        for p in self.panels:
            p.total_frames = align_frames

        # Align fps between panels
        align_fps = np.array([ p.fps for p in self.panels ]).min()
        for p in self.panels:
            p.fps = align_fps

        self._ready = True
        self.fps = align_fps
        self.total_frames = align_frames
