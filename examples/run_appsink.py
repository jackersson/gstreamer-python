import sys
import traceback
import argparse
import typing as typ
import time
import attr

import numpy as np

from gstreamer import GstContext, GstPipeline, GstApp, Gst, GstVideo
import gstreamer.utils as utils

# Converts list of plugins to gst-launch string
# ['plugin_1', 'plugin_2', 'plugin_3'] => plugin_1 ! plugin_2 ! plugin_3
DEFAULT_PIPELINE = utils.to_gst_string([
    "videotestsrc num-buffers=100",
    "capsfilter caps=video/x-raw,format=GRAY16_LE,width=640,height=480",
    "queue",
    "appsink emit-signals=True"
])

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pipeline", required=False,
                default=DEFAULT_PIPELINE, help="Gstreamer pipeline without gst-launch")

args = vars(ap.parse_args())

command = args["pipeline"]


def extract_buffer(sample: Gst.Sample) -> np.ndarray:
    """Extracts Gst.Buffer from Gst.Sample and converts to np.ndarray"""

    buffer = sample.get_buffer()  # Gst.Buffer

    print("timestamp: ", Gst.TIME_ARGS(buffer.pts), "offset: ", buffer.offset)

    caps_format = sample.get_caps().get_structure(0)  # Gst.Structure

    # GstVideo.VideoFormat
    video_format = GstVideo.VideoFormat.from_string(
        caps_format.get_value('format'))

    w, h = caps_format.get_value('width'), caps_format.get_value('height')
    c = utils.get_num_channels(video_format)

    buffer_size = buffer.get_size()
    shape = (h, w, c) if (h * w * c == buffer_size) else buffer_size

    format_info = GstVideo.VideoFormat.get_info(video_format)  # GstVideo.VideoFormatInfo
    array = np.ndarray(shape=shape // (format_info.bits // utils.BITS_PER_BYTE),
                       buffer=buffer.extract_dup(0, buffer_size),
                       dtype=utils.get_np_dtype(video_format))

    return np.squeeze(array)  # remove single dimension if exists


def on_buffer(sink: GstApp.AppSink, data: typ.Any) -> Gst.FlowReturn:
    """Callback on 'new-sample' signal"""
    # Emit 'pull-sample' signal
    # https://lazka.github.io/pgi-docs/GstApp-1.0/classes/AppSink.html#GstApp.AppSink.signals.pull_sample

    sample = sink.emit("pull-sample")  # Gst.Sample

    if isinstance(sample, Gst.Sample):
        array = extract_buffer(sample)
        print(
            "Received {type} with shape {shape} of type {dtype}".format(type=type(array),
                                                                        shape=array.shape,
                                                                        dtype=array.dtype))
        return Gst.FlowReturn.OK

    return Gst.FlowReturn.ERROR


with GstContext():  # create GstContext (hides MainLoop)
    # create GstPipeline (hides Gst.parse_launch)
    with GstPipeline(command) as pipeline:
        appsink = pipeline.get_by_cls(GstApp.AppSink)[0]  # get AppSink
        # subscribe to <new-sample> signal
        appsink.connect("new-sample", on_buffer, None)
        while not pipeline.is_done:
            time.sleep(.1)
