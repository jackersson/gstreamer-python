import time
from random import randint

from gstreamer import GstPipeline, GstContext


if __name__ == '__main__':
    ctx = GstContext()
    ctx.startup()

    pipelines = [GstPipeline(f"videotestsrc num-buffers={randint(50, 300)} ! gtksink") for _ in range(5)]

    for p in pipelines:
        p.startup()

    while any(p.is_active for p in pipelines):
        time.sleep(.5)

    for p in pipelines:
        p.shutdown()

    ctx.shutdown()
