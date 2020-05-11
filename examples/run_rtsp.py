#!/usr/bin/env python
# -*- coding:utf-8 vi:ts=4:noexpandtab
# Simple RTSP server. Run as-is or with a command-line to replace the default pipeline

import time
import sys
import abc
import numpy as np
import typing as typ
from fractions import Fraction
import functools

import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
gi.require_version('GstApp', '1.0')
gi.require_version('GstVideo', '1.0')

from gi.repository import Gst, GLib, GstRtspServer, GObject, GstApp, GstVideo  # noqa:F401,F402

import gstreamer as gst  # noqa:F401,F402

# Examples
# https://github.com/tamaggo/gstreamer-examples
# https://github.com/GStreamer/gst-rtsp-server/tree/master/examples
# https://stackoverflow.com/questions/47396372/write-opencv-frames-into-gstreamer-rtsp-server-pipeline


VIDEO_FORMAT = "RGB"
WIDTH, HEIGHT = 640, 480
FPS = Fraction(30)
GST_VIDEO_FORMAT = GstVideo.VideoFormat.from_string(VIDEO_FORMAT)


class GstBufferGenerator(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get(self) -> Gst.Buffer:
        pass

    @property
    def caps(self) -> Gst.Caps:
        pass

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def startup(self):
        pass

    def shutdown(self):
        pass


class FakeGstBufferGenerator(GstBufferGenerator):

    def __init__(self, *, width: int, height: int,
                 fps: typ.Union[Fraction, int] = Fraction('30/1'),
                 video_type: gst.gst_tools.VideoType = gst.gst_tools.VideoType.VIDEO_RAW,
                 video_frmt: GstVideo.VideoFormat = GstVideo.VideoFormat.RGB):

        self._width = width
        self._height = height

        self._fps = Fraction(fps)

        self._pts = 0
        self._dts = GLib.MAXUINT64

        self._duration = Gst.SECOND / (self._fps.numerator / self._fps.denominator)
        self._video_frmt = video_frmt
        self._video_type = video_type

        # Gst.Caps
        self._caps = gst.gst_tools.gst_video_format_plugin(
            width=width, height=height, fps=self._fps,
            video_type=video_type, video_frmt=video_frmt
        )

    @property
    def caps(self) -> Gst.Caps:
        return Gst.Caps.from_string(self._caps)

    def get(self) -> Gst.Buffer:

        np_dtype = gst.utils.get_np_dtype(self._video_frmt)
        channels = gst.utils.get_num_channels(self._video_frmt)

        array = np.random.randint(low=0, high=255,
                                  size=(self._height, self._width, channels), dtype=np_dtype)

        self._pts += self._duration

        gst_buffer = gst.utils.ndarray_to_gst_buffer(array)  # Gst.Buffer

        gst_buffer.pts = self._pts
        gst_buffer.dts = self._dts
        gst_buffer.duration = self._duration
        gst_buffer.offset = self._pts // self._duration

        return gst_buffer


class GstBufferGeneratorFromPipeline(GstBufferGenerator):

    def __init__(self, gst_launch: str, loop: bool = False):
        self._loop = loop
        self._gst_launch = gst_launch
        self._num_loops = 0

        self._pipeline = None  # gst.GstVideoSource

    def startup(self):
        self._pipeline = gst.GstVideoSource(self._gst_launch, max_buffers_size=8)
        self._pipeline.startup()

        self._num_loops += 1
        print(f"Starting {self._num_loops} loop")

    def shutdown(self):
        if isinstance(self._pipeline, gst.GstVideoSource):
            self._pipeline.shutdown()

    @property
    def caps(self) -> Gst.Caps:
        appsink = self._pipeline.get_by_cls(GstApp.AppSink)[0]
        return appsink.sinkpad.get_current_caps()

    def get(self) -> Gst.Buffer:

        buffer = self._pipeline.pop()
        if not buffer:
            if self._pipeline.is_done and self._loop:
                self.shutdown()
                self.startup()
            return None

        gst_buffer = gst.utils.ndarray_to_gst_buffer(buffer.data)  # Gst.Buffer

        gst_buffer.pts = buffer.pts
        gst_buffer.dts = buffer.dts
        gst_buffer.duration = buffer.duration
        gst_buffer.offset = buffer.offset

        return gst_buffer

    @classmethod
    def clone(cls) -> 'GstBufferGeneratorFromPipeline':
        return cls(self._gst_launch)


def get_child_by_cls(element: Gst.Element, cls: GObject.GType) -> typ.List[Gst.Element]:
    """ Get Gst.Element[] from pipeline by GType """
    return [e for e in element.iterate_elements() if isinstance(e, cls)]


# https://lazka.github.io/pgi-docs/GstRtspServer-1.0/classes/RTSPMediaFactory.html#gstrtspserver-rtspmediafactory
class RTSPMediaFactoryCustom(GstRtspServer.RTSPMediaFactory):

    def __init__(self, source: typ.Callable[..., GstBufferGenerator]):
        super().__init__()

        self._source = source
        self._sources = {}

    def do_create_element(self, url) -> Gst.Element:
        # https://lazka.github.io/pgi-docs/GstRtspServer-1.0/classes/RTSPMediaFactory.html#GstRtspServer.RTSPMediaFactory.do_create_element

        src = "appsrc emit-signals=True is-live=True"
        encoder = "x264enc tune=zerolatency"  # pass=quant
        color_convert = "videoconvert n-threads=0 ! video/x-raw,format=I420"
        rtp = "rtph264pay config-interval=1 name=pay0 pt=96"
        pipeline = "{src} ! {color_convert} ! {encoder} ! queue max-size-buffers=8 ! {rtp}".format(**locals())
        print(f"gst-launch-1.0 {pipeline}")
        return Gst.parse_launch(pipeline)

    def on_need_data(self, src: GstApp.AppSrc, length: int):
        """ Callback on "need-data" signal

        Signal:
            https://lazka.github.io/pgi-docs/GstApp-1.0/classes/AppSrc.html#GstApp.AppSrc.signals.need_data
        :param length: amount of bytes needed
        """

        buffer = None
        while not buffer:  # looping pipeline
            buffer = self._sources[src.name].get()  # Gst.Buffer
            time.sleep(.1)

        retval = src.emit('push-buffer', buffer)
        # print(f'Pushed buffer, frame {buffer.offset}, duration {Gst.TIME_ARGS(buffer.pts)}')
        if retval != Gst.FlowReturn.OK:
            print(retval)

    def do_configure(self, rtsp_media: GstRtspServer.RTSPMedia):
        # https://lazka.github.io/pgi-docs/GstRtspServer-1.0/classes/RTSPMedia.html#GstRtspServer.RTSPMedia

        appsrc = get_child_by_cls(rtsp_media.get_element(), GstApp.AppSrc)[0]

        self._sources[appsrc.name] = self._source()
        self._sources[appsrc.name].startup()
        time.sleep(.5)  # wait to start

        # this instructs appsrc that we will be dealing with timed buffer
        appsrc.set_property("format", Gst.Format.TIME)

        # instructs appsrc to block pushing buffers until ones in queue are preprocessed
        # allows to avoid huge queue internal queue size in appsrc
        appsrc.set_property("block", True)

        appsrc.set_property("caps", self._sources[appsrc.name].caps)

        appsrc.connect('need-data', self.on_need_data)

    def __del__(self):
        for source in self._sources.values():
            source.shutdown()


class GstServer():
    def __init__(self, shared: bool = False):
        # https://lazka.github.io/pgi-docs/GstRtspServer-1.0/classes/RTSPServer.html
        self.server = GstRtspServer.RTSPServer()

        # https://lazka.github.io/pgi-docs/GstRtspServer-1.0/classes/RTSPMediaFactory.html#GstRtspServer.RTSPMediaFactory.set_shared
        # f.set_shared(True)

        # https://lazka.github.io/pgi-docs/GstRtspServer-1.0/classes/RTSPServer.html#GstRtspServer.RTSPServer.get_mount_points
        # https://lazka.github.io/pgi-docs/GstRtspServer-1.0/classes/RTSPMountPoints.html#GstRtspServer.RTSPMountPoints
        m = self.server.get_mount_points()

        generator = functools.partial(FakeGstBufferGenerator, width=WIDTH, height=HEIGHT,
                                      fps=FPS, video_frmt=GST_VIDEO_FORMAT)

        # pipeline
        # pipeline = "videotestsrc num-buffers=1000 ! capsfilter caps=video/x-raw,format=RGB,width=640,height=480 ! appsink emit-signals=True"

        # path = "/home/taras/coder/datai/production/sales_zone/data/videos/letoile/sales_zone_letoile.mp4"
        # pipeline = f"filesrc location={path} ! decodebin ! videoconvert n-threads=0 ! video/x-raw,format=RGB ! appsink emit-signals=True"

        # generator = functools.partial(
        #     GstBufferGeneratorFromPipeline, gst_launch=pipeline, loop=True
        # )

        # https://lazka.github.io/pgi-docs/GstRtspServer-1.0/classes/RTSPMountPoints.html#GstRtspServer.RTSPMountPoints.add_factory
        mount_point = "/stream.rtp"
        factory = RTSPMediaFactoryCustom(generator)
        factory.set_shared(shared)
        m.add_factory(mount_point, factory)  # adding streams

        port = self.server.get_property("service")
        print(f"rtsp://localhost:{port}/{mount_point}")

        # https://lazka.github.io/pgi-docs/GstRtspServer-1.0/classes/RTSPServer.html#GstRtspServer.RTSPServer.attach
        self.server.attach(None)


if __name__ == '__main__':
    with gst.GstContext():
        s = GstServer(shared=True)

        while True:
            time.sleep(.1)
