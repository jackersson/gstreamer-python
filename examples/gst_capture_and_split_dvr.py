import time
import numpy as np
import threading

from gstreamer import GstVideoSource, GstVideoSink, GstVideo, Gst, GLib, GstContext

WIDTH, HEIGHT, CHANNELS = 640, 480, 3
NUM_BUFFERS = 1000
VIDEO_FORMAT = GstVideo.VideoFormat.RGB

video_format_str = GstVideo.VideoFormat.to_string(VIDEO_FORMAT)

# capturing pipeline
caps_filter = "capsfilter caps=video/x-raw,format={video_format_str},width={WIDTH},height={HEIGHT}".format(
    **locals())
capture_cmd = "videotestsrc num-buffers={NUM_BUFFERS} ! {caps_filter} ! appsink emit-signals=True sync=false".format(
    **locals())

# video record pipeline
dvr_cmd = "appsrc emit-signals=True is-live=True ! videoconvert ! x264enc tune=zerolatency ! mp4mux ! filesink location={}"

NUM_VIDEO_FILES = 2
NUM_FRAMES_PER_VIDEO_FILE = NUM_BUFFERS // NUM_VIDEO_FILES
with GstContext(), GstVideoSource(capture_cmd) as capture:

    idx_video_file, num_read = 0, -1
    video_writer = None
    try:
        while not capture.is_done or capture.queue_size > 0:
            buffer = capture.pop()  # GstBuffer

            # restart video_writer is necessary
            if num_read == -1 or num_read > NUM_FRAMES_PER_VIDEO_FILE:
                num_read = 0

                # shutdown previous video writer
                if video_writer:
                    video_writer.shutdown()

                # initialize new one
                video_writer = GstVideoSink(dvr_cmd.format(f"video_{idx_video_file}.mp4"),
                                            width=WIDTH, height=HEIGHT, video_frmt=VIDEO_FORMAT)
                video_writer.startup()

                idx_video_file += 1

            if buffer:
                num_read += 1
                video_writer.push(buffer.data)  # np.ndarray

    except Exception as e:
        print("Exception :", e)
    finally:
        if video_writer:
            video_writer.shutdown()
