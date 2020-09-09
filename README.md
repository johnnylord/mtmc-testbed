# mtmc-testbed

## Observation
```
# Speed Boost method

## Network performance
nmlab507_5G - 55 Mbps - 30 fps
nmlab507    - 11 Mbps - 1 fps
NTU         - 10 Mbps - 1 fps

## Process priority (-20 ~ 19)
Give different process different process prioirty
```


## Introduction
The design pattern of MTMC testbed is client-server architecture.

Client is responsible for fetching video sources from multiple cameras, sending video frames to server for processing, and rendering result sent back from server.
Server is responsible for processing video frame sent from client and sent back result to client.

Following is the system architecture:
![system architecture](imgs/system.png)


## Download AIST dataset
- Download AIST video metadata list
```bash
$ cd resource/aist
$ wget https://aistdancedb.ongaaccel.jp/data/video_refined/10M/refined_10M_all_video_url.csv
```
- Download Popping videos (You can specifiy filter to download anything you want)
```bash
$ mkdir popping_ch01

# First use following command to check videos you want to downloads
# $ cat refined_10M_all_video_url.csv | grep gPO | grep sGR | grep ch01
$ cat refined_10M_all_video_url.csv | grep gPO | grep sGR | grep ch01 | xargs -I{} -P 4 wget {}
$ mv *.mp4 popping_ch01
```

## Usage Guide
- Launch server first
```bash
$ python server --ip 0.0.0.0 --port 6666
```
- Launch client app
```bash
$ python client --config config/client/video.yml
```

## Design your own app & worker
In the following, I will use echoapp and echoworker for our example. As you can see 'echo' term in app and worker, the client send whatever it get from camera and server just echoing recieved message from client and send back nothing.

All apps must inherited from base class `App`, and override what `App` specifies need to be defined in derived class. As each app associated with specific type of worker, you need to define class attribute `MATCHED_WORKER`.
```python
# File: app/echoapp/__init__.py

class EchoApp(App):

    MATCHED_WORKER = "EchoWorker"

    def __init__(self, **kwargs):
        raise RuntimeError("Cannot directly instantiate object from EchoApp")

    @check_ready
    def run(self):
        """App loop for running app"""
        pass

    def keyboaord_handler(self, key):
        pass

    def mouse_callback(self, event, x, y, flags, param):
        pass

    def trackbar_callback(self, value):
        pass
```

To make `EchoApp` available to the system, you need to register it into the system. Add `EchoApp` class type into the `AVAILABLE_APPS` in `LazyApp`.
```python
# File: app/__init__.py

from .base import App
from .echoapp import EchoApp

class LazyApp(App):

    AVAILABLE_APPS = [ EchoApp ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ...
```

Before using `EchoApp`, you also need to define its worker `EchoWorker` and register it into system.

All workers must inherited from base class `Worker`, and override what `Worker` specifies need to be defined in derived class. Each worker has its own default configuration, so you need to define class attribute `DEFAULT_CONFIG` in each worker class.
```python
# File: worker/echoworker/__init__.py
class EchoWorker(Worker):

    DEFAULT_CONFIG = {}

    def __init__(self):
        raise RuntimeError("You cannot directly instantiate EchoWorker")

    def boot(self, config):
        """Prepare environment for worker to run"""
        pass

    def run(self):
        """Worker job"""
        pass
```

After that, register your defined worker into system
```python
# File: worker/__init__.py

import os
import sys
from .base import Worker

# Add following lines
from .echoworker import EchoWorker

# ...
```

Then you are ready to experiment on your `Echoapp` & `Echoworker`
