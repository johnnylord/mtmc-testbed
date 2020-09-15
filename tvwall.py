import os
import yaml
import signal
import argparse
from subprocess import Popen


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", required=True, type=str, help="tvall configuraiton file")

def signal_handler(signum, frame):
    raise Exception()

signal.signal(signal.SIGINT, signal_handler)

def main(args):
    # Load tvwall configuraiton file
    with open(args['config']) as f:
        config = yaml.full_load(f)
        tw = config['common']['transmit_resolution'][0]
        th = config['common']['transmit_resolution'][1]

    # Construct child process for each apps
    procs = []
    for app_setting in config['apps']:
        cmd = ("python script/single.py"
                " --ip {ip} --port {port}"
                " --tw {tw} --th {th}"
                " --src {src} --name {name}"
                " --output {output}").format(
                ip=config['common']['remote_ip'],
                port=config['common']['remote_port'],
                tw=tw, th=th,
                src=app_setting['src'],
                name=app_setting['name'],
                output=config['common']['output_dir'])
        p = Popen(cmd.split())
        procs.append(p)

    # Wait for all child process to complete
    try:
        while len(procs):
            status = [ p.poll() for p in procs ]
            for idx, s in enumerate(status):
                if s is not None:
                    procs.remove(procs[idx])

    except Exception:
        for p in procs:
            p.kill()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args)
