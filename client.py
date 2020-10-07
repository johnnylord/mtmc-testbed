import os
import yaml
import logging
import signal
import argparse
import multiprocessing

from app.gui.media import MediaPlayer
from app.gui.panel import Panel
from app import LazyApp


class LogFilter(object):

    def __init__(self, level):
        self._level = level

    def filter(self, logRecord):
        return logRecord.levelno == self._level

# Logging System
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Stream Handler
s_handler = logging.StreamHandler()
s_handler.setLevel(logging.INFO)
s_format = logging.Formatter('%(asctime)s, %(levelname)s, PID[%(process)d] %(name)s, %(message)s')
s_handler.setFormatter(s_format)

# Error file handler
f1_handler = logging.FileHandler("log-client-error.txt")
f1_handler.setLevel(logging.ERROR)
f1_handler.addFilter(LogFilter(logging.ERROR))
f1_format = logging.Formatter('%(asctime)s, PID[%(process)d], %(name)s, %(message)s')
f1_handler.setFormatter(f1_format)

# Info file handler
f2_handler = logging.FileHandler("log-client-info.txt")
f2_handler.setLevel(logging.INFO)
f2_handler.addFilter(LogFilter(logging.INFO))
f2_format = logging.Formatter('%(asctime)s, PID[%(process)d], %(name)s, %(message)s')
f2_handler.setFormatter(f2_format)

# Register handler on root logger
logger.addHandler(s_handler)
logger.addHandler(f1_handler)
logger.addHandler(f2_handler)

# Commandline parser
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", required=True, help="app configuration file")

# Ctrl+C handler
def signal_handler(signum, frame):
    raise Exception("Ctrl+C is triggerd")

signal.signal(signal.SIGINT, signal_handler)

def main(args):
    # Load app configuraiton file
    with open(args['config']) as f:
        config = yaml.full_load(f)

    # Construct video medias
    medias = [ MediaPlayer(src=src).start()
                for src in config['app']['sources']  ]

    # Wrap medias into panels
    panels = [  Panel(media=media)
                for media in medias   ]

    # Group panels into application
    app = LazyApp(ip=config['app']['remote_ip'],
                port=config['app']['remote_port'],
                trans_resolution=tuple(config['app']['transmit_resolution']),
                size=tuple(config['app']['resolution']),
                nrows=config['app']['nrows'],
                ncols=config['app']['ncols'],
                font_scale=config['app']['font_scale'],
                line_thickness=config['app']['line_thickness'])

    for panel in panels:
        app.add_panel(panel)

    try:
        # Let user select what type of app he/she wants to use
        app.config_prompt()

        # Run the application
        app.ready()
        app.boot()
        app.run()
        app.close()

    except Exception as e:
        app.close()
        logger.warning(f"{app} shutdown abnormally", exc_info=True)

    # Export result
    app.export(config['app']['output_dir'])

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args)
