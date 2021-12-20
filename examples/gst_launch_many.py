import time
from random import randint

from gstreamer import GstPipeline, GstContext


if __name__ == '__main__':
    with GstContext():
        pipelines = [GstPipeline(
            "videotestsrc num-buffers={} ! gtksink".format(randint(50, 300))) for _ in range(5)]

        for p in pipelines:
            p.startup()

        while any(p.is_active for p in pipelines):
            time.sleep(.5)

        for p in pipelines:
            p.shutdown()
