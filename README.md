# Multi-Camera Multi-Target Tracking Testbed

![demo](https://github.com/johnnylord/mtmc-testbed/blob/master/imgs/demo.gif?raw=true)

A testbed to help you develop your tracking algorithm with online visualization.

## Motivation
Tracking algorithm consists of multiple phases. For a simple tracking-by-detection paradigm, the overall processing pipeline can be represented as follows:

![pipeline](https://github.com/johnnylord/mtmc-testbed/blob/master/imgs/pipeline.png?raw=true)

In the above pipeline, `detection`, and `recognition` components need powerful GPU resource to get their jobs done in reasonable time. Therefore, this testbed decouples the tracking application into two parts. One for visualization purpose, and the other one is for GPU-intensive tracking algorithm.

## Testbed Diagram

![testbed](https://github.com/johnnylord/mtmc-testbed/raw/master/imgs/system.png)

## Package Dependencies

Use the package manager `pip` to install all the required packages

```bash
$ pip install -r requirements.txt
```

## Run Testbed

Run `server.py` on computer with power GPU(nvidia) resource
```bash
$ python server.py --ip 0.0.0.0 --port 6666
```

Modify config file (`config/client/video.yml`) of app before running `client.py`
```yaml
---
app:
  remote_ip: "140.112.18.217"     # server ip
  remote_port: 6666               # server port

  resolution: [1920, 1080]        # display video resolution
  transmit_resolution: [512, 512] # transmission video resolution

  output_dir: "result"            # Output directory for tracking result

  nrows: 2                        # number of rows in app
  ncols: 2                        # number of cols in app

  # videos to process 
  sources:
    - "resource/aist/locking_ch01/gLO_sGR_c01_d13_d14_d15_mLO0_ch01.mp4"
    - "resource/aist/locking_ch01/gLO_sGR_c02_d13_d14_d15_mLO0_ch01.mp4"
    - "resource/aist/locking_ch01/gLO_sGR_c08_d13_d14_d15_mLO0_ch01.mp4"
```

Run `client.py` on your PC/laptop
```bash
$ python client.py --config config/client/video.yml
```

NOTE:
> - To make your `client.py` run smoothly, make sure you are connected to a fast network
> - Reduce the `transmission_resolution` can also make your program run faster

## Develop Your Own App and Worker

You can refer to `app/echoapp/__init__.py` and `worker/echoworker/__init__.py`, and extend their  abilities respectively to develop your own algorithm.

All the client apps are inherited from `App` defined in `app/base.py`. Check the source code to know which methods you need to override in the subclass to get the client app work properly.

All the server workers are inherited from `Worker` defined in `worker/base.py`. Check the source code to know which methods you need to override in the subclass to get the server worker work properly.

Last but not least, register your own app and worker by modifying the files `app/__init__.py` and `worker/__init__.py`:

```python
# app/__init__.py

from .base import App
from .echoapp import EchoApp
from .detapp import DetApp
from .motapp import MOTApp

class LazyApp(App):

    AVAILABLE_APPS = [ EchoApp, DetApp, MOTApp ] # Add your developed app class

    # ...
```

```python
# app/__init__.py

# You only need to import your worker class here
from .base import Worker
from .echoworker import EchoWorker
from .detworker import DetWorker
from .motworker import MOTWorker

class LazyWorker(Worker):

    # ...
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

