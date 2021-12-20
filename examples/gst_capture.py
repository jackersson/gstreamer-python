import numpy as np
import time
import threading

from gstreamer import GstVideoSource, GstVideo, Gst, GLib, GstContext

WIDTH, HEIGHT, CHANNELS = 640, 480, 3
NUM_BUFFERS = 50
VIDEO_FORMAT = GstVideo.VideoFormat.RGB

video_format_str = GstVideo.VideoFormat.to_string(VIDEO_FORMAT)
caps_filter = "capsfilter caps=video/x-raw,format={video_format_str},width={WIDTH},height={HEIGHT}".format(**locals())
command = "videotestsrc num-buffers={NUM_BUFFERS} ! {caps_filter} ! appsink emit-signals=True sync=false".format(**locals())

last_buffer = None
with GstContext(), GstVideoSource(command) as pipeline:

    while pipeline.is_active or pipeline.queue_size > 0:
        buffer = pipeline.pop()
        if buffer:
            print("{}: shape {}".format(Gst.TIME_ARGS(buffer.pts), buffer.data.shape))
            last_buffer = buffer

print("Read {} buffers".format(last_buffer.offset))
