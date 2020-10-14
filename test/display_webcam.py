import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2

from app.gui.media import MediaPlayer
from app.gui.panel import Panel
from app.gui.container import Container

# Construct media player
media1 = MediaPlayer(src="0").start()

# Wrap media as panel
panel1 = Panel(media1)

# Wrap all panels into a container
container = Container(nrows=1, ncols=1)
container.add_panel(panel1)
container.ready()

cv2.namedWindow("test", cv2.WINDOW_GUI_EXPANDED)

while True:
    content= container.render()

    cv2.imshow("test", content['container_frame'])

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
