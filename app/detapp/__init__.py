import logging

import cv2
import numpy as np

from ..base import App
from ..base import Keyboard as kb
from ..gui.container import check_ready
from ..gui.media import MediaType
from ..utils.transform import convert_bbox_coordinate
from ..utils.visualize import draw_bbox


logger = logging.getLogger(__name__)


__all__ = [ "DetApp" ]

class DetApp(App):

    MATCHED_WORKER = "DetWorker"

    def __init__(self, **kwargs):
        raise RuntimeError("Cannot directly instantiate object from MOTApp")

    def boot(self):
        """Prepare runtime environment for worker"""
        pass

    @check_ready
    def run(self):
        """App loop for running app"""
        while not self.is_stop():
            content = self.render()
            fid, frame = content['fid'], content['container_frame']

            # Send request
            request = { 'action': 'detect' }
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

            # Rerender panels (add bboxes)
            panel_contents = []
            for panel in response['content']:
                pid = panel['pid']
                confs = [ bbox['conf'] for bbox in panel['bboxes']]
                bboxes = [ bbox['bbox'] for bbox in panel['bboxes']]

                # Select target panel to manipulate
                target_panel = [ panel
                                for panel in self.panels
                                if panel.pid == pid ][0]

                # Draw bboxes on target panel
                target_media_frame = target_panel.media_cache
                new_resolution = target_media_frame.shape[:2][::-1]
                old_resolution = self.trans_resolution
                bboxes = convert_bbox_coordinate(bboxes, old_resolution, new_resolution)
                for bbox in bboxes:
                    draw_bbox(target_media_frame, bbox, thickness=3)

                # Rerender
                target_panel_content = target_panel.rerender(target_media_frame)
                panel_contents.append(target_panel_content)

            # Align/Sort rerendered panel_contents
            panel_contents = [ [ panel_content
                                for panel_content in panel_contents
                                if panel_content['pid'] == panel.pid ][0]
                            for panel in self.panels ]

            # Rerender container
            content = self.rerender(panel_contents)
            fid, frame = content['fid'], content['container_frame']

            # Show applications
            cv2.imshow(self.winname, frame)
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
