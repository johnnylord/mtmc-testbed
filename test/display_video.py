import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2

from app.gui.media import MediaPlayer
from app.gui.panel import Panel
from app.gui.container import Container

# Construct media player
media1 = MediaPlayer(src="../resource/aist/group/locking_ch01/gLO_sGR_c01_d13_d14_d15_mLO0_ch01.mp4").start()
media2 = MediaPlayer(src="../resource/aist/group/locking_ch01/gLO_sGR_c02_d13_d14_d15_mLO0_ch01.mp4").start()
media3 = MediaPlayer(src="../resource/aist/group/locking_ch01/gLO_sGR_c03_d13_d14_d15_mLO0_ch01.mp4").start()
media4 = MediaPlayer(src="../resource/aist/group/locking_ch01/gLO_sGR_c04_d13_d14_d15_mLO0_ch01.mp4").start()

# Wrap media as panel
panel1 = Panel(media1)
panel2 = Panel(media2)
panel3 = Panel(media3)
panel4 = Panel(media4)

# Wrap all panels into a container
container = Container(nrows=2, ncols=2)
container.add_panel(panel1)
container.add_panel(panel2)
container.add_panel(panel3)
container.add_panel(panel4)
container.ready()

cv2.namedWindow("test", cv2.WINDOW_GUI_EXPANDED)

while True:
    content= container.render()

    cv2.imshow("test", content['container_frame'])

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
