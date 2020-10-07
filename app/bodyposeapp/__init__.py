import os
import os.path as osp
import logging

import cv2
import numpy as np

from ..base import App
from ..base import Keyboard as kb
from ..gui.container import check_ready
from ..gui.media import MediaType
from ..utils.transform import convert_bbox_coordinate
from ..utils.visualize import draw_bbox, draw_bodypose


logger = logging.getLogger(__name__)


__all__ = [ "BodyPoseApp" ]

class BodyPoseApp(App):

    MATCHED_WORKER = "BodyPoseWorker"

    def __init__(self, **kwargs):
        raise RuntimeError("Cannot directly instantiate object from BodyPoseApp")

    def boot(self):
        """Prepare runtime environment for worker"""
        self.video_results = {}
        self.event_handler = { 'detect': self._detect_handler }

    def export(self, output_dir):
        """Export tracking result to output directory"""
        # Check output directory exists
        output_dir = osp.join(output_dir, self.__class__.__name__)
        if not osp.exists(output_dir):
            os.makedirs(output_dir)

        # Export video result panel-by-panel
        for panel, result in self.video_results.items():
            fname = "{}.txt".format(osp.basename(panel.src))
            fname = osp.join(output_dir, fname)
            with open(fname, "w") as f:
                fids = sorted(result.keys())
                for fid in fids:
                    people = result[fid]
                    for person in people:
                        bbox = person['bbox'] # (xmin, ymin, xmax, ymax)

                        # keypoints (18, 3)
                        #   - 18 types of keypoint
                        #   - 3 means (x, y, score)
                        # The order of keypoint is conformed to openpose project
                        keypoints = person['keypoints']
                        keypoints = keypoints.reshape(-1)
                        line = f"{fid},0"
                        bline = ",".join([ str(bbox[i]) for i in range(4) ])
                        kline = ",".join([ str(v) for v in keypoints ])
                        line = ",".join([line, bline, kline]) + '\n'
                        f.write(line)

        logger.info(f"Export result to '{output_dir}'")

    @check_ready
    def run(self):
        """App loop for running app"""
        while not self.is_stop():
            content = self.render()
            fid, frame = content['fid'], content['container_frame']

            if not self.is_pause():
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

                # Handle server response
                handler = self.event_handler[response['action']]
                new_content = handler(response)
                fid, frame = new_content['fid'], new_content['container_frame']
                last_frame = frame

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

    def _detect_handler(self, response):
        # Rerender panels (add bboxes)
        panel_contents = []
        for panel in response['content']:
            pid = panel['pid']
            people = panel['people']
            bboxes = np.array([ person['bbox'] for person in people ])
            keypointss = np.array([ person['keypoints'] for person in people ])

            # Select target panel to manipulate
            target_panel = [ panel
                            for panel in self.panels
                            if panel.pid == pid ][0]

            # Convert coordinate system
            target_media_frame = target_panel.media_cache
            new_resolution = np.array(target_media_frame.shape[:2][::-1])
            old_resolution = np.array(self.trans_resolution)
            bboxes = convert_bbox_coordinate(bboxes, old_resolution, new_resolution)

            # Convert keypoint system
            keypointss[:, :, :2] = keypointss[:, :, :2] * (new_resolution / old_resolution)

            # Save result in mot tracking format
            for bbox, keypoints in zip(bboxes, keypointss):
                # Check data structure format
                if target_panel not in self.video_results:
                    self.video_results[target_panel] = {}
                if target_panel.fid not in self.video_results[target_panel]:
                    self.video_results[target_panel][target_panel.fid] = []

                record = { 'bbox': bbox, 'keypoints': keypoints }
                self.video_results[target_panel][target_panel.fid].append(record)

            # Draw bboxes on target panel
            for bbox in bboxes:
                draw_bbox(target_media_frame, bbox, thickness=self.line_thickness)

            for keypoints in keypointss:
                draw_bodypose(target_media_frame, keypoints, thickness=self.line_thickness)

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
        return content
