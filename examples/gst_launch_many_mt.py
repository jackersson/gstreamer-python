"""
    Launching multiple pipelines in different threads. Graceful shutdown
"""

import time
import threading

from gstreamer import GstPipeline, GstContext

DEFAULT_PIPELINE = "videotestsrc num-buffers=100 ! gtksink"


def launch(stop_event: threading.Event):
    with GstContext(), GstPipeline(DEFAULT_PIPELINE) as pipeline:
        while not pipeline.is_done and not stop_event.is_set():
            time.sleep(.1)


def launch_context(event: threading.Event):
    with GstContext():
        while not event.is_set():
            time.sleep(1)


def launch_pipeline(event: threading.Event):
    with GstPipeline(DEFAULT_PIPELINE) as pipeline:
        while not pipeline.is_done and not event.is_set():
            time.sleep(.1)


if __name__ == '__main__':

    num_pipeline = 3
    num_threads = num_pipeline + 1  # thread for context
    events = [threading.Event() for _ in range(num_threads)]
    for e in events:
        e.clear()

    context = threading.Thread(target=launch_context, args=(events[0],))
    pipelines = [threading.Thread(target=launch_pipeline, args=(e,))
                 for e in events[1:]]

    threads = [context] + pipelines
    for t in threads:
        t.start()

    try:
        # check if any thread (except context) is alive
        while any([t.isAlive() for t in threads[1:]]):
            time.sleep(.1)
    except KeyboardInterrupt as e:
        print("Pressed Ctrl-C")
    finally:
        # reverse, so the context will be stopped the last one
        for e, t in zip(reversed(events), reversed(threads)):
            e.set()
            try:
                t.join(timeout=1)
            except Exception as e:
                pass
