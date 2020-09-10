import logging

import cv2
import numpy as np

from ..base import App
from ..base import Keyboard as kb
from ..gui.container import check_ready
from ..gui.media import MediaType


logger = logging.getLogger(__name__)

__all__ = [ "EchoApp" ]

class EchoApp(App):

    MATCHED_WORKER = "EchoWorker"

    def __init__(self, **kwargs):
        raise RuntimeError("Cannot directly instantiate object from EchoApp")

    def boot(self):
        """Prepare runtime environment for worker"""
        pass

    def export(self, output_dir):
        """Export tracking result to output directory"""
        pass

    @check_ready
    def run(self):
        """App loop for running app"""
        while not self.is_stop():
            content = self.render()
            fid, frame = content['fid'], content['container_frame']

            # Send request
            request = { 'action': 'echo' }
            self.send(request)

            # Send raw frames to workers
            video_frames = []
            for panel in self.panel_to_channel.keys():
                media_frame = panel.media_cache
                media_frame = cv2.resize(media_frame, self.trans_resolution)
                frame_bytes = cv2.imencode('.jpg', media_frame)[1]
                video_frames.append({ 'panel': panel, 'frame_bytes': frame_bytes })
            self.parallel_send_videos(video_frames)

            # Catch response from remote worker
            response = self.recv()
            if response is None:
                break

            # Show applications
            cv2.imshow(self.winname, frame)

            if self.stype == MediaType.VIDEO:
                cv2.setTrackbarPos(self.barname, self.winname, fid)

            # Handling keyboard events
            key = cv2.waitKey(1) & 0xff
            self.keyboaord_handler(key)

        cv2.destroyAllWindows()

    def keyboaord_handler(self, key):
        # When certain panel is in focused
        # ====================================
        if self.mode == App.OPERATION_MODE:
            if key == kb.ESC:
                self.focus_panel.focus = False
                self.focus_panel = None
                self.mode = App.SELECT_MODE
                return

        # Common key handler
        # =====================================
        super().keyboaord_handler(key)

    def mouse_callback(self, event, x, y, flags, param):
        # Wait for selecting panel to focus on
        # ==================================================
        if self.mode == App.SELECT_MODE:
            super().mouse_callback(event, x, y, flags, param)

        elif self.mode == App.OPERATION_MODE:
            pass

    def trackbar_callback(self, value):
        super().trackbar_callback(value)
