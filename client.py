import os
import yaml
import logging
import signal
import argparse
import multiprocessing

from app.gui.media import MediaPlayer
from app.gui.panel import Panel
from app import LazyApp

# Logging system
logger = logging.getLogger()
logger.setLevel(logging.INFO)

s_handler = logging.StreamHandler()
s_handler.setLevel(logging.INFO)
s_format = logging.Formatter('%(levelname)s - %(message)s')
s_handler.setFormatter(s_format)

f_handler = logging.FileHandler("log-client.txt")
f_handler.setLevel(logging.ERROR)
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)

logger.addHandler(s_handler)
logger.addHandler(f_handler)

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
                ncols=config['app']['ncols'])

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
