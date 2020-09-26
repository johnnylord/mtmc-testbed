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
from ..utils.visualize import draw_bbox, draw_text, draw_velocity, draw_gaussian, get_unique_color


logger = logging.getLogger(__name__)


__all__ = [ "MTMCApp" ]

class MTMCApp(App):

    MATCHED_WORKER = "MTMCWorker"

    def __init__(self, **kwargs):
        raise RuntimeError("Cannot directly instantiate object from MOTApp")

    def boot(self):
        """Prepare runtime environment for worker"""
        self.video_results = {}
        self.event_handler = { 'track': self._track_handler }

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
                    tracks = result[fid]
                    for t in tracks:
                        f.write(f"{fid},{t[0]},{t[1]},{t[2]},{t[3]},{t[4]}\n")

        logger.info(f"Export result to '{output_dir}'")

    @check_ready
    def run(self):
        """App loop for running app"""
        while not self.is_stop():
            content = self.render()
            fid, frame = content['fid'], content['container_frame']

            if not self.is_pause():
                # Send request
                request = { 'action': 'track' }
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
            if not self.is_pause():
                cv2.imshow(self.winname, frame)
            else:
                cv2.imshow(self.winname, last_frame)
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

    def _track_handler(self, response):
        # Rerender panels (add tracks)
        panel_contents = []
        for panel in response['content']:
            pid = panel['pid']
            tids = [ track['tid'] for track in panel['tracks'] if track['state'] == "tracked" ]
            bboxes = [ track['bbox'] for track in panel['tracks'] if track['state'] == "tracked" ]
            velocities = [ track['velocity'] for track in panel['tracks'] if track['state'] == 'tracked' ]
            covars = [ track['covar'] for track in panel['tracks'] if track['state'] == "tracked" ]

            # Select target panel to manipulate
            target_panel = [ panel
                            for panel in self.panels
                            if panel.pid == pid ][0]

            # Convert coordinate system
            target_media_frame = target_panel.media_cache
            new_resolution = target_media_frame.shape[:2][::-1]
            old_resolution = self.trans_resolution

            # Convert coordinate system of bbox
            bboxes = convert_bbox_coordinate(bboxes, old_resolution, new_resolution)
            means = np.array([ ((b[0]+b[2])//2, (b[1]+b[3])//2) for b in bboxes ])

            # Convert coordinate system of velocity of targets
            if len(velocities) > 0:
                scale_vec = np.array(new_resolution) / np.array(old_resolution)
                velocities = np.array(velocities)*scale_vec

            # Convert coordinate system of covariacne matrix
            if len(covars) > 0:
                scale_vec = np.array(new_resolution) / np.array(old_resolution)
                covars = np.array(covars)*scale_vec

            # Save result in mot tracking format
            for tid, bbox in zip(tids, bboxes):
                # Check data structure format
                if target_panel not in self.video_results:
                    self.video_results[target_panel] = {}
                if target_panel.fid not in self.video_results[target_panel]:
                    self.video_results[target_panel][target_panel.fid] = []

                record = (tid, bbox[0], bbox[1], bbox[2], bbox[3])
                self.video_results[target_panel][target_panel.fid].append(record)

            # Draw tracks on target panel
            for tid, bbox, velocity, mean, covar in zip(tids, bboxes, velocities, means, covars):
                bbox_color = get_unique_color(tid)
                draw_bbox(target_media_frame,
                            bbox=bbox,
                            color=(bbox_color),
                            thickness=self.line_thickness)
                draw_text(target_media_frame,
                            text=str(tid),
                            position=bbox[:2],
                            fontScale=self.font_scale,
                            fgcolor=(255, 255, 255),
                            bgcolor=bbox_color)
                draw_velocity(target_media_frame,
                            position=mean,
                            vector=velocity,
                            thickness=self.line_thickness)
                draw_gaussian(target_media_frame,
                            mean=mean,
                            covariance=covar,
                            color=bbox_color,
                            thickness=self.line_thickness)

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
