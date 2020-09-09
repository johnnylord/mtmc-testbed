import os
import os.path as osp
import sys
sys.path = [ osp.dirname(osp.dirname(osp.abspath(__file__))) ] + sys.path

import yaml
import signal
import argparse
import traceback

from app.gui.media import MediaPlayer
from app.gui.panel import Panel
from app import LazyApp


parser = argparse.ArgumentParser()
parser.add_argument("--ip", required=True, type=str, help="server ip")
parser.add_argument("--port", required=True, type=int, help="server port")
parser.add_argument("--src", required=True, type=str, help="video source")
parser.add_argument("--tw", required=True, type=int, help="transmit width")
parser.add_argument("--th", required=True, type=int, help="transmit height")
parser.add_argument("--name", required=True, type=str, help="app name")


def signal_handler(signum, frame):
    raise Exception()

signal.signal(signal.SIGINT, signal_handler)

def main(args):
    # Extract user arguments
    remote_ip = args['ip']
    remote_port = args['port']
    src = args['src']
    trans_width = args['tw']
    trans_height = args['th']
    app_name = args['name']

    # Construct app
    media = MediaPlayer(src=src).start()
    panel = Panel(media=media)
    app = LazyApp(ip=remote_ip, port=remote_port,
                trans_resolution=(trans_width, trans_height),
                nrows=1, ncols=1)
    app.add_panel(panel)

    try:
        # Configure app & remote worker
        app.config_as(app_name)

        # Run app
        app.ready()
        app.boot()
        app.run()
        app.close()

    except Exception:
        app.close()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args)
