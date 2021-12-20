import numpy as np

from gstreamer import GstVideoSink, GstVideo, GstContext

WIDTH, HEIGHT, CHANNELS = 640, 480, 3
NUM_BUFFERS = 1000
VIDEO_FORMAT = GstVideo.VideoFormat.RGB
command = "appsrc emit-signals=True is-live=True ! videoconvert ! gtksink sync=false"

with GstContext(), GstVideoSink(command, width=WIDTH, height=HEIGHT, video_frmt=VIDEO_FORMAT) as pipeline:

    for _ in range(NUM_BUFFERS):
        buffer = np.random.randint(low=0, high=255, size=(
            HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)
        pipeline.push(buffer)

    while pipeline.is_done:
        pass

    print("Displayed {} buffers".format(pipeline.total_buffers_count))
