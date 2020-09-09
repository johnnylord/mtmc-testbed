import logging

from ..base import Worker


logger = logging.getLogger(__name__)


class EchoWorker(Worker):
    """Echo what remote client sent

    This worker is just a simple example showing how to communicate with remote
    client. You can based on this worker to develop your own worker.
    """
    DEFAULT_CONFIG = {}

    def __init__(self):
        raise RuntimeError("You cannot directly instantiate EchoWorker")

    def boot(self, config):
        """Prepare environment for worker to run"""
        self.config = config

    def run(self):
        """Worker job"""
        try:
            while True:
                request = self.recv()
                videos = self.parallel_recv_videos()

                # Remote client socket is closed
                if request is None:
                    break

                # Send back message
                self.send(request)

        except Exception as e:
            logger.warning(f"Shutdown echoworker", exc_info=True)

        # Cleanup process
        self.close()
        logger.info(f"Shutdown {self}")
