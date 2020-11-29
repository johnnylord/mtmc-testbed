import os
import os.path as osp
import logging

import cv2
import numpy as np
from easydict import EasyDict

from ..base import App
from ..base import Keyboard as kb
from ..gui.container import check_ready
from ..gui.media import MediaType
from ..utils.transform import convert_bbox_coordinate
from ..utils.visualize import draw_bbox, draw_text, draw_gaussian, get_unique_color


logger = logging.getLogger(__name__)


__all__ = [ "SOTApp" ]

class SOTApp(App):

    MATCHED_WORKER = "SOTWorker"

    def __init__(self, **kwargs):
        raise RuntimeError("Cannot directly instantiate object from SOTApp")

    def boot(self):
        """Prepare runtime environment for worker"""
        self.video_results = {}
        self.event_handler = {
            'nop': self._nop_handler,
            'reset': self._reset_handler,
            'track': self._track_handler,
            }
        self.state = EasyDict({
            'debug': True,
            'reset': False,
            'tracked': False,
            'results': {},
            # Opencv app state
            'app': {
                'click': False,
                'clicked': False,
                'tlbr': [],
                },

            # Synchronized with remote tracker on server
            'remote': {
                "fid": -1,
                "pid": None,
                "tlbr": [],
                },
            })

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
            # Render new frame
            content = self.render()
            fid, frame = content['fid'], content['container_frame']
            action = self._determine_action()

            # Target object is being tracked
            if action == 'track':
                old_resolution = frame.shape[:2][::-1]
                new_resolution = self.trans_resolution

                # Prepare current tracked object position to remote server
                self.state.remote['fid'] = fid
                if (
                    self.state.tracked
                    and len(self.state.app.tlbr) == 4
                    and self.mode == App.OPERATION_MODE
                ):
                    tlbr = self.state.app.tlbr
                    self.state.remote.tlbr = convert_bbox_coordinate([tlbr],
                                                                old_resolution,
                                                                new_resolution)[0]
                    self.state.remote.pid = self.focus_panel.pid
                    self.state.app.tlbr = []
                else:
                    self.state.remote.pid = None
                    self.state.remote.tlbr = []

                # Send request
                request = { 'action': action, 'remote': self.state.remote }
                self.send(request)

                # Send raw frames to workers
                video_frames = []
                for panel in self.panel_to_channel.keys():
                    media_frame = panel.media_cache
                    media_frame = cv2.resize(media_frame, self.trans_resolution)
                    frame_bytes = cv2.imencode('.jpg', media_frame)[1]
                    video_frames.append({ 'panel': panel, 'frame_bytes': frame_bytes })
                self.parallel_send_videos(video_frames)

            # No object is being tracked
            else:
                # Send request
                request = { 'action': action }
                self.send(request)

            # Catch response from remote worker
            response = self.recv()
            if response is None:
                break

            # Handle server response
            handler = self.event_handler[response['action']]
            new_content = handler(response)
            if response['action'] == 'track':
                fid, frame = new_content['fid'], new_content['container_frame']

            # Draw the selected bbox
            if (
                self.mode == App.OPERATION_MODE
                and self.state.app.clicked
                and len(self.state.app.tlbr) == 4
            ):
                frame = self.container_cache.copy()
                draw_bbox(frame, self.state.app.tlbr)

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

            if key == ord('r') or key == ord('R'):
                self.state.reset = True

        # Common key handler
        # =====================================
        super().keyboaord_handler(key)

    def mouse_callback(self, event, x, y, flags, param):
        # Wait for selecting panel to focus on
        # ==================================================
        if self.mode == App.SELECT_MODE:
            super().mouse_callback(event, x, y, flags, param)

        elif self.mode == App.OPERATION_MODE:
            # Save the top left coordinate (x, y) of the tracking bounding box
            if event == cv2.EVENT_LBUTTONDOWN:
                self.state.app.click = True
                self.state.app.clicked = True
                self.state.tracked = False
                self.state.app.tlbr = [x, y]
                if self.is_start():
                    self.pause()

            # Temporarily save the bottom right coordinate (x, y) of the tracking box
            elif event == cv2.EVENT_MOUSEMOVE and self.state.app.clicked:
                self.state.app.click = False
                if len(self.state.app.tlbr) == 4:
                    self.state.app.tlbr[2] = x
                    self.state.app.tlbr[3] = y
                elif len(self.state.app.tlbr) == 2:
                    self.state.app.tlbr += [x, y]

            # Save the final bottom right coordinate (x, y) of the tracking box
            elif event == cv2.EVENT_LBUTTONUP and self.state.app.clicked:
                self.state.tracked = True
                self.state.app.clicked = False
                # Prevent rectangle with zero area
                if len(self.state.app.tlbr) == 2:
                    self.state.app.tlbr += [x+10, y+10]
                elif len(self.state.app.tlbr) == 4:
                    self.state.app.tlbr[2] = x
                    self.state.app.tlbr[3] = y
                if self.is_pause():
                    self.start()

    def trackbar_callback(self, value):
        super().trackbar_callback(value)

    def _determine_action(self):
        """Given current app state determine the action and sent arguments

        There are three poosible actions for single object tracking application.
            - 'nop':    send dummy package to the server
            - 'reset':  send reset signal to the server
            - 'track':  send tracking signal and position of tracked object

        Returns:
            action(str)
        """
        if self.state.app.clicked and not self.state.tracked:
            return 'reset'

        return 'track'

    def _nop_handler(self, response):
        self.state.reset = False

    def _reset_handler(self, response):
        if self.is_start():
            self.pause()
        self.state.reset = False
        self.state.app.click = False
        logger.info("Reset")

    def _track_handler(self, response):
        # Rerender panels (add tracks)
        panel_contents = []
        for panel in response['content']:
            # Extract information of tracked object
            pid = panel['pid']
            tids = [ track['tid']
                    for track in panel['tracks']
                    if track['state'] == "tracked" ]
            bboxes = [ track['bbox']
                    for track in panel['tracks']
                    if track['state'] == "tracked" ]
            covars = [ track['covar']
                    for track in panel['tracks']
                    if track['state'] == "tracked" ]

            assert len(tids) <= 1
            assert len(bboxes) <= 1
            assert len(covars) <= 1

            # Select target panel to manipulate
            target_panel = [ panel
                            for panel in self.panels
                            if panel.pid == pid ][0]
            target_media_frame = target_panel.media_cache

            # Nothing is being tracked
            if len(bboxes) == 0:
                target_panel_content = target_panel.rerender(target_media_frame)
                panel_contents.append(target_panel_content)
                if target_panel not in self.video_results:
                    self.video_results[target_panel] = {}
                if target_panel.fid not in self.video_results[target_panel]:
                    self.video_results[target_panel][target_panel.fid] = []
                continue

            # Convert coordinate system
            old_resolution = self.trans_resolution
            new_resolution = target_media_frame.shape[:2][::-1]
            bboxes = convert_bbox_coordinate(bboxes, old_resolution, new_resolution)
            means = np.array([ ((b[0]+b[2])//2, (b[1]+b[3])//2) for b in bboxes ])

            scale_vec = np.array(new_resolution) / np.array(old_resolution)
            covars = np.array(covars)*scale_vec

            # Draw tracks on target panel
            for tid, bbox, mean, covar in zip(tids, bboxes, means, covars):
                bbox_color = get_unique_color(tid)
                draw_bbox(target_media_frame, bbox=bbox, color=(bbox_color), thickness=3)
                draw_text(target_media_frame, text=str(tid), position=bbox[:2],
                        fontScale=3, fgcolor=(255, 255, 255), bgcolor=bbox_color)

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
