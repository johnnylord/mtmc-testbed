import math
from uuid import uuid1

import cv2
import numpy as np

from .media import MediaPlayer, MediaType


__all__ = [ "Panel" ]


class Panel:
    """Display wrapper for media player

    Arguments:
        media (MediaPlayer): media media
        margin (int): margin size
        border (int): border size

    A single panel is defined as following:

        +---------------------------------------+
        |                Margin                 |
        |  +---------------------------------+  |
        |  |             Border              |  |
        |  |  +---------------------------+  |  |
        |  |  |       Media content       |  |  |
        |  |  +---------------------------+  |  |
        |  |                                 |  |
        |  +---------------------------------+  |
        |                                       |
        +---------------------------------------+

    Note:
        - Every panel must resides in a Container object
        - Border & Margin are predefined/static values
        - Size of video content is adjusted dynamically with respect to the
            host container
    """
    TIMER_LIMIT = 90

    def __init__(self, media, margin=10, border=3):
        # Panel metadata
        # =========================================
        self.pid = uuid1().hex
        self.media = media
        self.margin = margin
        self.border = border

        # Proxy media metadata
        # =========================================
        self.src = self.media.src
        self.stype = self.media.stype

        # Changed after aligned with other panels
        self.fps = self.media.fps
        self.total_frames = self.media.total_frames

        # Random but static information
        # ========================================
        self.base_color = tuple(np.array(np.random.rand(3)*255, dtype=int).tolist())
        self.dynamic_dim = np.random.randint(len(self.base_color))

        # Dynamic information (assign value later)
        # =========================================
        self.fid = -1
        self.size = (-1, -1)
        self.focus = False
        self.timer = 0
        self.panel_cache = None
        self.media_cache = None

    @property
    def border_color(self):
        times = np.linspace(0, 2*np.pi, Panel.TIMER_LIMIT)
        values = (np.sin(times)+1)*100
        color = list(self.base_color)
        color[self.dynamic_dim] = int(values[self.timer])

        # Update timer
        self.timer = self.timer+1 if self.timer < Panel.TIMER_LIMIT-1 else 0

        return color

    # Proxy methods for internal media
    # ================================
    def start(self):
        self.media.start()
        return self

    def pause(self):
        self.media.pause()
        return self

    def stop(self):
        self.media.stop()
        return self

    def jump(self, index):
        if self.is_pause():
            self.media.jump(index)
        return self

    def is_start(self):
        return self.media.state == MediaPlayer.STATE_START

    def is_pause(self):
        return self.media.state == MediaPlayer.STATE_PAUSE

    def is_stop(self):
        return self.media.state == MediaPlayer.STATE_STOP

    # Panel methods
    # ========================================
    def reset_timer(self):
        self.timer = 0

    def responsive(self, size):
        self.size = size

    def rerender(self, media_frame):
        """Rerender media content portion in the panel"""
        # Determine panel size & Define panal canvas
        panel_width = (self.size[0]
                        if self.size[0] > 0 else
                        self.media.width + 2*self.border + 2*self.margin)
        panel_height = (self.size[1]
                        if self.size[1] > 0 else
                        self.media.height + 2*self.border + 2*self.margin)

        # Render video frame
        media_width = panel_width - 2*self.border - 2*self.margin
        media_height = panel_height - 2*self.border - 2*self.margin
        resized_frame = cv2.resize(media_frame, dsize=(media_width, media_height))

        # Copy video frame to panel canvas
        offset = self.border + self.margin
        x_end = offset + media_width
        y_end = offset + media_height
        self.panel_cache[offset:y_end, offset:x_end, :] = resized_frame

        content = { 'pid': self.pid,
                    'src': self.src,
                    'fid': self.fid,
                    'panel_frame': self.panel_cache,
                    'media_frame': self.media_cache }
        return content

    def render(self):
        """Return panel content of type dict"""
        fid, media_frame = self.media.read()

        self.fid = fid
        self.media_cache = media_frame.copy()

        # Display content is synchronized with other panels
        if fid >= self.total_frames and self.total_frames > 0:
            content = { 'pid': self.pid,
                        'src': self.src,
                        'fid': self.total_frames,
                        'panel_frame': self.panel_cache,
                        'media_frame': self.media_cache, }
            return content

        # Determine panel size & Define panal canvas
        panel_width = (self.size[0]
                        if self.size[0] > 0 else
                        self.media.width + 2*self.border + 2*self.margin)
        panel_height = (self.size[1]
                        if self.size[1] > 0 else
                        self.media.height + 2*self.border + 2*self.margin)
        panel_frame = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

        # Render video frame
        media_width = panel_width - 2*self.border - 2*self.margin
        media_height = panel_height - 2*self.border - 2*self.margin
        resized_frame = cv2.resize(media_frame, dsize=(media_width, media_height))

        # Copy video frame to panel canvas
        offset = self.border + self.margin
        x_end = offset + media_width
        y_end = offset + media_height
        panel_frame[offset:y_end, offset:x_end, :] = resized_frame

        # Render border
        panel_frame = cv2.rectangle(panel_frame,
                                    pt1=(self.margin+self.border,
                                        self.margin+self.border),
                                    pt2=(panel_width-self.margin-self.border,
                                        panel_height-self.margin-self.border),
                                    color=self.border_color,
                                    thickness=self.border)
        self.panel_cache = panel_frame.copy()

        content = { 'pid': self.pid,
                    'src': self.src,
                    'fid': self.fid,
                    'panel_frame': panel_frame,
                    'media_frame': media_frame }
        return content
