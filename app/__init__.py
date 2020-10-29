import logging

from .base import App
from .echoapp import EchoApp
from .detapp import DetApp
from .sotapp import SOTApp
from .mtmcapp import MTMCApp
from .bodyposeapp import BodyPoseApp


logger = logging.getLogger(__name__)

class LazyApp(App):
    """An unknown app in lazy state

    Every app will be first in this state before casting itself into true app.
    User will interactive tell the app to become which type of app, and communicate
    with remote worker properly to serve the app.
    """
    AVAILABLE_APPS = [ EchoApp, DetApp, MTMCApp, BodyPoseApp, SOTApp ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def config_as(self, app_name):
        """Use default configuration to configure app & remote worker with user interaction"""
        # Cast to target app
        # ====================================================================
        app_names = [ app_cls.__name__ for app_cls in LazyApp.AVAILABLE_APPS ]
        target_app_idx = app_names.index(app_name)
        target_app = LazyApp.AVAILABLE_APPS[target_app_idx]
        self.__class__ = target_app

        # Configure remote worker
        # ====================================================================
        # Tell remote worker to become target worker
        message = { 'action': 'worker', 'content': target_app.MATCHED_WORKER }
        self.send(message)

        # Use default worker config
        # ====================================================================
        response = self.recv()
        if response is None:
            raise Exception("Remote worker has been closed")

        # Send changed worker config to server
        worker_config = response['content']
        message = { 'action': 'config', 'content': worker_config }
        self.send(message)

        # You're ready to go
        logger.info(f"{self.__class__.__name__} has been configured properly")

    def config_prompt(self, default_app=None):
        # Select type of apps
        # ====================================================
        while True:
            print("\nChoose one of following apps to use:")
            for idx, app in enumerate(LazyApp.AVAILABLE_APPS):
                print(f"\t[{idx}]: {app.__name__}")

            target_idx = input("Applicaiton ID: ")
            if len(target_idx) == 0:
                continue
            else:
                target_idx = int(target_idx)
                break

        target_app = LazyApp.AVAILABLE_APPS[target_idx]
        self.__class__ = target_app

        # Configure remote worker
        # ====================================================
        # Tell remote worker to become target worker
        message = { 'action': 'worker', 'content': target_app.MATCHED_WORKER }
        self.send(message)

        # Recieve default worker config
        response = self.recv()
        if response is None:
            raise Exception("Remote worker has been closed")

        # Let user change worker config
        worker_config = response['content']
        while True:
            print("\nCurrent worker configuraiton:")
            for idx, (key, value) in enumerate(worker_config.items()):
                print(f"\t[{idx}]: '{key}' = {value}")

            option_idx = input("Option ID ('q' to exit): ")
            if len(option_idx) == 0:
                continue
            elif option_idx == 'q':
                break
            else:
                option_idx = int(option_idx)
                target_key = list(worker_config.keys())[option_idx]
                worker_config[target_key] = input(f"\t'{target_key}' = ")

        # Send changed worker config to server
        message = { 'action': 'config', 'content': worker_config }
        self.send(message)

        # You're ready to go
        logger.info(f"{self.__class__.__name__} has been configured properly")

    def boot(self):
        raise RuntimeError("You're still in lazy state")

    def export(self):
        raise RuntimeError("You're still in lazy state")

    def run(self):
        raise RuntimeError("You're still in lazy state")

    def keyboaord_handler(self, key):
        raise RuntimeError("You're still in lazy state")

    def mouse_callback(self, event, x, y, flags, param):
        raise RuntimeError("You're still in lazy state")

    def trackbar_callback(self, value):
        raise RuntimeError("You're still in lazy state")
