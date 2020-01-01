import time
import numpy as np
import threading

from gstreamer import GstVideoSource, GstVideoSink, GstVideo, Gst, GLib, GstContext

WIDTH, HEIGHT, CHANNELS = 640, 480, 3
NUM_BUFFERS = 1000
VIDEO_FORMAT = GstVideo.VideoFormat.RGB

video_format_str = GstVideo.VideoFormat.to_string(VIDEO_FORMAT)
caps_filter = f"capsfilter caps=video/x-raw,format={video_format_str},width={WIDTH},height={HEIGHT}"
capture_cmd = f"videotestsrc num-buffers={NUM_BUFFERS} ! {caps_filter} ! appsink emit-signals=True sync=false"

display_cmd = "appsrc emit-signals=True is-live=True ! videoconvert ! gtksink sync=false"


with GstContext(), GstVideoSource(capture_cmd) as capture, \
        GstVideoSink(display_cmd, width=WIDTH, height=HEIGHT, video_frmt=VIDEO_FORMAT) as display:

    # wait pipeline to initialize
    max_num_tries, num_tries = 5, 0
    while not display.is_active and num_tries <= max_num_tries:
        time.sleep(.1)
        num_tries += 1

    while not capture.is_done or capture.queue_size > 0:
        buffer = capture.pop()
        if buffer:
            display.push(buffer.data, pts=buffer.pts,
                         dts=buffer.dts, offset=buffer.offset)
            # print(f"{Gst.TIME_ARGS(buffer.pts)}: shape {buffer.data.shape}")
