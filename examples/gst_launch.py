import time
import argparse

from gstreamer import GstPipeline, GstContext

DEFAULT_PIPELINE = "videotestsrc num-buffers=100 ! fakesink sync=false"

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pipeline", required=True,
                default=DEFAULT_PIPELINE, help="Gstreamer pipeline without gst-launch")

args = vars(ap.parse_args())

if __name__ == '__main__':
    with GstContext(), GstPipeline(args['pipeline']) as pipeline:
        while not pipeline.is_done:
            time.sleep(.1)
