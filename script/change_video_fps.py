import os
import os.path as osp
import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, type=str, help="input directory containing mp4 videos")
parser.add_argument("--output_dir", default="output", type=str, help="output directory containing transformed videos")
parser.add_argument("--fps", default=30, type=int, help="target fps")
parser.add_argument("--core", default=4, type=int, help="number of parallel processes")


def main(args):
    # Input vide files to be transformed
    videos = [ osp.join(args['input_dir'], fname)
            for fname in os.listdir(args['input_dir']) ]

    # Check output directory
    if not osp.exists(args['output_dir']):
        os.makedirs(args['output_dir'])

    # Spawn ffmpeg process to transform input videos
    procs = []

    try:
        video_count = 0
        while video_count < len(videos):
            ifname = videos[video_count]
            ofname = osp.join(args['output_dir'], osp.basename(ifname))
            cmdline = "ffmpeg -i {} -r 30 -y {}".format(ifname, ofname)

            if len(procs) < args['core']:
                p = subprocess.Popen(cmdline.split())
                procs.append(p)
                video_count += 1

            else:
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
