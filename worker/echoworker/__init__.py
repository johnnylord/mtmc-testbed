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
        pass

    def run(self):
        """Worker job"""
        try:
            while not self.shutdown_event.is_set():
                request = self.recv()
                videos = self.parallel_recv_videos()

                # Remote client socket is closed
                if request is None:
                    break

                # Send back message
                self.send(request)

        except Exception as e:
            logger.warning(f"Error occur in echoworker", exc_info=True)

        # Cleanup process
        self.close()

    def close(self):
        pass
