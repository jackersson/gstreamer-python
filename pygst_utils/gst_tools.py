"""

Usage Example:
>>> width, height, num_buffers = 1920, 1080, 100
>>> caps_filter = 'capsfilter caps=video/x-raw,format=RGB,width={},height={}'.format(width, height)
>>> source_cmd = 'videotestsrc num-buffers={} ! {} ! appsink emit-signals=True sync=false'.format(
...     num_buffers, caps_filter)
>>> display_cmd = "appsrc emit-signals=True is-live=True ! videoconvert ! gtksink sync=false"
>>>
>>> with GstVideoSource(source_cmd) as pipeline, GstVideoSink(display_cmd, width=width, height=height) as display:
...     current_num_buffers = 0
...     while current_num_buffers < num_buffers:
...         buffer = pipeline.pop()
...         if buffer:
...             display.push(buffer.data)
...             current_num_buffers += 1
>>>
"""

import os
import queue
import logging
import threading
import typing as typ
from enum import Enum
from functools import partial
from fractions import Fraction

import attr
import numpy as np

from .. import base

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GLib, GObject, GstApp, GstVideo  # noqa:F401,F402

Gst.init(None)

class NamedEnum(Enum):

    def __repr__(self):
        return str(self)

    @classmethod
    def names(cls) -> typ.List[str]:
        return list(cls.__members__.keys())


class VideoType(NamedEnum):
    """
    https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/media-types.html?gi-language=c
    """
    VIDEO_RAW = 'video/x-raw'
    VIDEO_GL_RAW = "video/x-raw(memory:GLMemory)"
    VIDEO_NVVM_RAW = "video/x-raw(memory:NVMM)"


_CHANNELS = {
    GstVideo.VideoFormat.RGB: 3,
    GstVideo.VideoFormat.RGBA: 4,
    GstVideo.VideoFormat.RGBX: 4,
    GstVideo.VideoFormat.BGR: 3,
    GstVideo.VideoFormat.BGRA: 4,
    GstVideo.VideoFormat.BGRX: 4,
    GstVideo.VideoFormat.GRAY8: 1,
    GstVideo.VideoFormat.GRAY16_BE: 1
}


_SUPPORTED_CHANNELS_NUM = set(_CHANNELS.values())
_SUPPORTED_VIDEO_FORMATS = [GstVideo.VideoFormat.to_string(s) for s in _CHANNELS.keys()]


def gst_video_format_to_str(fmt: GstVideo.VideoFormat) -> str:
    return GstVideo.VideoFormat.to_string(fmt)


def is_num_channels_supported(num_channels: int) -> bool:
    return num_channels in _SUPPORTED_CHANNELS_NUM


def get_num_channels(fmt: GstVideo.VideoFormat) -> int:
    return _CHANNELS[fmt]


_DTYPES = {
    GstVideo.VideoFormat.GRAY16_BE: np.float16
}


def get_np_dtype(fmt: GstVideo.VideoFormat) -> np.number:
    return _DTYPES.get(fmt, np.uint8)


class GstPipeline:
    """Base class to initialize any Gstreamer Pipeline from string"""

    def __init__(self, command: str):
        """
        :param command: gst-launch string
        """

        self._command = command
        self._main_loop = None     # GObject.MainLoop
        self._pipeline = None      # Gst.Pipeline
        self._bus = None           # Gst.Bus

        self._log = logging.getLogger('pygst.gt.{}'.format(self.__class__.__name__))
        self._log.info("%s \n gst-launch-1.0 %s", self, command)

        self._end_event = threading.Event()
        self._pipeline_end_event = threading.Event()
        self._main_loop_thread = threading.Thread(target=self._main_loop_run)

    @property
    def log(self) -> logging.Logger:
        return self._log

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return '<{}>'.format(self)

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def get_by_cls(self, cls: GObject.GType) -> typ.List[Gst.Element]:
        """ Get Gst.Element[] from pipeline by GType """
        return [e for e in self._pipeline.iterate_elements() if isinstance(e, cls)]

    def get_by_name(self, name: str) -> Gst.Element:
        """Get Gst.Element from pipeline by name
        :param name: plugins name (name={} in gst-launch string)
        """
        return self._pipeline.get_by_name(name)

    def startup(self):
        """ Starts pipeline. (Blocking operation) """
        if self.is_active:
            self.log.warning("%s is already started. ", self)
            return

        self._pipeline = Gst.parse_launch(self._command)

        # Due to SIGINT handle
        # https://github.com/beetbox/audioread/issues/63#issuecomment-390394735
        self._main_loop = GLib.MainLoop.new(None, False)

        self._bus = self._pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.enable_sync_message_emission()
        self._bus.connect("message", self._bus_call, None)

        self._on_pipeline_init()
        self._pipeline.set_state(Gst.State.READY)

        self.log.info('Starting %s', self)

        self._end_event.clear()
        self._pipeline_end_event.clear()

        self.log.debug("%s Setting pipeline state to %s ... ", self, Gst.State.PLAYING)
        self._pipeline.set_state(Gst.State.PLAYING)
        self.log.debug("%s Pipeline state set to %s ", self, Gst.State.PLAYING)

        self._main_loop_thread.start()

    def _on_pipeline_init(self):
        """Sets additional properties for plugins in Pipeline"""
        pass

    def _main_loop_run(self):
        if self.shutdown_requested:
            return

        try:
            self.log.debug("%s Starting main loop ... ", self)
            self._main_loop.run()
        except KeyboardInterrupt:
            self.log.error("Error %s: %s", "KeyboardInterrupt", self)
        finally:
            self._stop_pipeline()

    @property
    def bus(self) -> Gst.Bus:
        return self._bus

    @property
    def pipeline(self) -> Gst.Pipeline:
        return self._pipeline

    def _stop_pipeline(self, timeout: int = 1, eos: bool = False) -> None:
        """ Stops pipeline
        :param eos: if True -> send EOS event
            - EOS event necessary for FILESINK finishes properly
            - Use when pipeline crushes
        """
        if self._pipeline_end_event.is_set():
            return

        self._pipeline_end_event.set()

        # print("EOS ", self.is_active)
        if eos or self.is_active:

            self.log.debug("%s Sending EOS event ...", self)
            try:
                thread = threading.Thread(target=self._pipeline.send_event, args=(Gst.Event.new_eos(),))
                thread.start()
                thread.join(timeout=timeout)
            except Exception:
                pass
            self.log.debug("%s EOS event sent successfully.", self)

        self.log.debug("%s Stoping main loop ...", self)
        self._main_loop.quit()
        self.log.debug("%s Main loop stopped", self)

        try:
            if self._main_loop_thread.is_alive():
                self._main_loop_thread.join(timeout=timeout)
        except Exception as err:
            self.log.error("Error %s: %s", self, err)

        try:
            def cleanup(pipeline: Gst.Pipeline) -> None:
                """Clean Gst.Pipline"""
                pipeline.set_state(Gst.State.NULL)
                del pipeline

            thread = threading.Thread(target=cleanup, args=(self._pipeline,))
            thread.start()
            thread.join(timeout=timeout)
        except Exception:
            self.log.debug("%s Reset pipeline state ....", self)

    def shutdown(self, timeout: int = 1, eos: bool = False) -> None:
        """Shutdown pipeline
        :param timeout: time to wait when pipeline fully stops
        """
        if self.shutdown_requested:
            return

        self.log.info('%s Shutdown started', self)

        self._end_event.set()

        self._stop_pipeline(timeout=timeout, eos=eos)

        self.log.info('%s successfully destroyed', self)

    @property
    def is_active(self) -> bool:
        return self._main_loop and self._main_loop.is_running() \
            and not self._pipeline_end_event.is_set()

    @property
    def shutdown_requested(self) -> bool:
        return self._end_event.is_set()

    def _bus_call(self, bus: Gst.Bus, message: Gst.Message, user_data: typ.Optional[typ.Any]) -> bool:
        mtype = message.type

        if mtype == Gst.MessageType.EOS:
            self.log.debug("%s received stream EOS event", self)
            self._stop_pipeline()
        elif mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.log.error("Error %s: %s. %s", err, debug, self)
            self._stop_pipeline()
        elif mtype == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            self.log.warning("%s: %s. %s", err, debug, self)
        return True


def gst_synced_plugin(plugin_string: str, sync: bool = False) -> str:
    """Adds sync attribute to plugin name

    Example:
        fakesink sync=False
    """
    return plugin_string + ' sync={}'.format(sync)


def fraction_to_str(fraction: Fraction) -> str:
    """Converts fraction to str"""
    return '{}/{}'.format(fraction.numerator, fraction.denominator)


def gst_appsrc_format(*, width: int, height: int, fps: Fraction = None,
                      video_type: VideoType = VideoType.VIDEO_RAW,
                      video_frmt: GstVideo.VideoFormat = GstVideo.VideoFormat.RGB) -> str:

    video_frmt_ = GstVideo.VideoFormat.to_string(video_frmt)
    gst_string = 'appsrc caps={},format={},width={},height={}'.format(video_type.value, video_frmt_, width, height)
    if fps:
        gst_string += ',fps={}'.format(fraction_to_str(fps))
    return gst_string


def gst_video_format_plugin(*, width: int = None, height: int = None, fps: Fraction = None,
                            video_type: VideoType = VideoType.VIDEO_RAW,
                            video_frmt: GstVideo.VideoFormat = GstVideo.VideoFormat.RGB) -> typ.Optional[str]:
    """
        https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-plugins/html/gstreamer-plugins-capsfilter.html
        Returns capsfilter
            video/x-raw,width=widht,height=height
            video/x-raw,framerate=fps/1
            video/x-raw,format=RGB
            video/x-raw,format=RGB,width=widht,height=height,framerate=1/fps
        :param width: image width
        :param height: image height
        :param fps: video fps
        :param video_type: gst specific (raw, h264, ..)
            https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html
        :param video_frmt: gst specific (RGB, BGR, RGBA)
            https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html
            https://lazka.github.io/pgi-docs/index.html#GstVideo-1.0/enums.html#GstVideo.VideoFormat
    """

    plugin = str(video_type.value)
    n = len(plugin)
    if video_frmt:
        plugin += ',format={}'.format(GstVideo.VideoFormat.to_string(video_frmt))
    if width and width > 0:
        plugin += ',width={}'.format(width)
    if height and height > 0:
        plugin += ',height={}'.format(height)
    if fps and fps > 0:
        plugin += ',framerate={}'.format(fraction_to_str(fps))

    if n == len(plugin):
        return None

    return plugin


def gst_video_format_bin(*, width: int = None, height: int = None, fps: Fraction = None,
                         video_type: VideoType = VideoType.VIDEO_RAW,
                         video_frmt: GstVideo.VideoFormat = GstVideo.VideoFormat.RGB) -> typ.Optional[typ.List[str]]:
    """
        Returns plugins string for
            video scaling:
                videoscale ! video/x-raw,width=width,height=height

            video fps change:
                videorate ! video/x-raw,framerate=1/fps

            video format change:
                video/x-raw,format=RGB

            all possible video change:
                videorate ! videoscale ! video/x-raw,format=RGB,width=widht,height=height,framerate=1/fps

        :param width: image width
        :param height: image height
        :param fps: video fps

        :param video_type: gst specific (raw, h264, ..)
            https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html

        :param video_frmt: gst specific (RGB, BGR, RGBA)
            https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html
            https://lazka.github.io/pgi-docs/index.html#GstVideo-1.0/enums.html#GstVideo.VideoFormat
    """

    plugin = gst_video_format_plugin(width=width, height=height, fps=fps, video_type=video_type, video_frmt=video_frmt)
    if plugin is None:
        return None

    plugins = [plugin]

    if "width" in plugin or "height" in plugin:
        # method: 0 - nearest neighbours
        # https://gstreamer.freedesktop.org/documentation/videoscale/index.html?gi-language=c#GstVideoScaleMethod
        plugins = ["videoscale method=0"] + plugins

    if "framerate" in plugin:
        plugins = ["videorate"] + plugins

    return plugins


def gst_named_plugin(plugin_string: str, plugin_name: typ.Optional[str] = None) -> str:
    """Returns plugin string with name """
    return plugin_string + ' name={}'.format(plugin_name) if plugin_name else plugin_string


def gst_source(source: str, plugin_name: typ.Optional[str] = None) -> str:
    """Returns gst source plugin for gst-launch

    :param source: file/device/stream
    :param plugin_name: unique plugin identifier

    Gstreamer Plugins*:
        filesrc: /data/doc/gstreamer/head/gstreamer-plugins/html/gstreamer-plugins-filesrc.html
        v4l2src: /documentation/video4linux2/v4l2src.html
        rtspsrc: /data/doc/gstreamer/head/gst-plugins-good/html/gst-plugins-good-plugins-rtspsrc.html
        TODO:
        splitmuxsrc : /data/doc/gstreamer/head/gst-plugins-good/html/gst-plugins-good-plugins-splitmuxsrc.html
        multifilesrc: /data/doc/gstreamer/head/gst-plugins-good/html/gst-plugins-good-plugins-multifilesrc.html
        souphttpsrc : /documentation/soup/souphttpsrc.html

        * Gst Host: https://gstreamer.freedesktop.org
    """
    plugin = ""
    if source.startswith('rtsp://'):
        plugin = 'rtspsrc location={}'.format(source)  # " latency=0 drop-on-latency=True"
    elif any(source.startswith('{}://'.format(schema)) for schema in ['https', 'rtsp']):
        raise NotImplementedError("HTTP stream not supported")
    elif "%0" in source or '*' in source:
        # images: multifilesrc (ex: source="img.%04d.png")
        # videos: splitmuxsrc (ex: source="video*.mov")
        # kind = filetype.guess(source)
        # if 'video' in kind.mime -> splitmuxsrc
        # if 'image' in kind.mime -> multifilesrc
        raise NotImplementedError("Multisource stream not supported")
    elif os.path.isfile(source):
        # FIXME: filetype not working with some files
        # mime_type = filetype.guess(source).mime
        # if 'video' not in mime_type:
        #     raise NotImplementedError('File type <{}> not supported'.format(mime_type))
        plugin = 'filesrc location={}'.format(source)
    elif any(source.startswith('{}'.format(schema)) for schema in ['/dev/video', '/device']):
        plugin = 'v4l2src device={}'.format(source)
    elif source == 'videotestsrc':
        plugin = 'videotestsrc'
    else:
        raise NotImplementedError("Not supported stream")

    return gst_named_plugin(plugin, plugin_name)


def gst_plugin_set_properties(plugin: str, **properties) -> str:
    # TODO: check if property exists and it's value
    plugin_ = plugin
    for k, v in properties.items():
        plugin_ += " {prop}={val}".format(prop=k, val=v)
    return plugin_


def flatten_list(in_list: typ.List) -> typ.List:
    """Flattens list"""
    result = []
    for item in in_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def to_gst_string(plugins: typ.List[str]) -> str:
    """ Generates string representation from list of plugins """

    if not plugins:
        return ""

    plugins_ = flatten_list(plugins)

    result = plugins_[0]
    for i in range(1, len(plugins_)):
        if '.' in plugins_[i][-1]:  # tee case
            result = result + ' ' + plugins_[i]
        else:
            result = result + ' ! ' + plugins_[i]  # ! between plugins
    return result


class GstVideoSink(GstPipeline):
    """Gstreamer Video Sink Base Class

    Usage Example:
        >>> width, height = 1920, 1080
        ... command = "appsrc emit-signals=True is-live=True ! videoconvert ! fakesink sync=false"
        ... with GstVideoSink(command, width=width, height=height) as pipeline:
        ...     for _ in range(10):
        ...         pipeline.push(buffer=np.random.randint(low=0, high=255, size=(height, width, 3), dtype=np.uint8))
        >>>
    """

    def __init__(self, command: str, *, width: int, height: int,
                 fps: typ.Union[Fraction, int] = Fraction('30/1'),
                 video_type: VideoType = VideoType.VIDEO_RAW,
                 video_frmt: GstVideo.VideoFormat = GstVideo.VideoFormat.RGB):

        super(GstVideoSink, self).__init__(command)

        self._fps = Fraction(fps)
        self._width = width
        self._height = height
        self._video_type = video_type
        self._video_frmt = video_frmt

        self._pts = 0
        self._dts = GLib.MAXUINT64
        self._duration = 10**9 / (fps.numerator / fps.denominator)

        self._src = None    # GstApp.AppSrc

    @property
    def video_frmt(self):
        return self._video_frmt

    def _on_pipeline_init(self):
        """Sets additional properties for plugins in Pipeline"""
        # find src element
        appsrcs = self.get_by_cls(GstApp.AppSrc)
        self._src = appsrcs[0] if len(appsrcs) == 1 else None
        if not self._src:
            raise ValueError("%s not found", GstApp.AppSrc)

        if self._src:
            self._src.set_property("format", Gst.Format.TIME)

            # set src caps
            caps = gst_video_format_plugin(
                width=self._width, height=self._height,
                fps=self._fps, video_type=self._video_type, video_frmt=self._video_frmt)

            self.log.debug("%s Caps: %s", self, caps)
            if caps is not None:
                self._src.set_property("caps", Gst.Caps.from_string(caps))

    @staticmethod
    def to_gst_buffer(buffer: typ.Union[Gst.Buffer, np.ndarray], *, pts: typ.Optional[int] = None,
                      dts: typ.Optional[int] = None,
                      offset: typ.Optional[int] = None, duration: typ.Optional[int] = None) -> Gst.Buffer:
        """ Parameters explained:
            https://lazka.github.io/pgi-docs/Gst-1.0/classes/Buffer.html#gst-buffer
        """
        gst_buffer = buffer
        if isinstance(gst_buffer, np.ndarray):
            gst_buffer = Gst.Buffer.new_wrapped(bytes(buffer))

        if not isinstance(gst_buffer, Gst.Buffer):
            raise ValueError('Invalid buffer format {} != {}'.format(type(gst_buffer), Gst.Buffer))

        gst_buffer.pts = pts or GLib.MAXUINT64
        gst_buffer.dts = dts or GLib.MAXUINT64
        gst_buffer.offset = offset or GLib.MAXUINT64
        gst_buffer.duration = duration or GLib.MAXUINT64
        return gst_buffer

    def push(self, buffer: typ.Union[Gst.Buffer, np.ndarray], *,
             pts: typ.Optional[int] = None, dts: typ.Optional[int] = None,
             offset: typ.Optional[int] = None):

        if self.shutdown_requested or not self.is_active:
            stop_requested = "Stop requested" if self.shutdown_requested else ""
            is_active = "Not active" if not self.is_active else ""
            self.log.warning("Warning %s: Can't push buffer. %s. %s", self, stop_requested, is_active)
            return

        if not self._src:
            raise RuntimeError('Src {} is not initialized'.format(Gst.AppSrc))

        self._pts += self._duration
        offset_ = int(self._pts / self._duration)

        gst_buffer = self.to_gst_buffer(buffer,
                                        pts=pts or self._pts,
                                        dts=dts or self._dts,
                                        offset=offset or offset_,
                                        duration=self._duration)

        self._src.emit("push-buffer", gst_buffer)

    @property
    def total_buffers_count(self) -> int:
        """Total pushed buffers count """
        return int(self._pts / self._duration)

    def shutdown(self, timeout: int = 1, eos: bool = False):
        if self.shutdown_requested:
            return

        if isinstance(self._src, GstApp.AppSrc):
            self._src.emit("end-of-stream")

        super().shutdown(timeout=timeout, eos=eos)


class LeakyQueue(queue.Queue):
    """Queue that contains only the last actual items and drops the oldest one."""

    def __init__(self, maxsize: int = 100):
        super().__init__(maxsize=maxsize)
        self._dropped = 0

    def put(self, item, block=True, timeout=None):
        if self.full():
            dropped_item = self.get_nowait()
            self._dropped += 1
            self._on_drop(self, dropped_item)
        super().put(item, block=block, timeout=timeout)

    @property
    def dropped(self):
        return self._dropped


@attr.s(slots=True, frozen=True)
class GstBuffer:
    data = attr.ib()                                   # type: np.ndarray
    pts = attr.ib(default=GLib.MAXUINT64)              # type: int
    dts = attr.ib(default=GLib.MAXUINT64)              # type: int
    offset = attr.ib(default=GLib.MAXUINT64)           # type: int
    duration = attr.ib(default=GLib.MAXUINT64)         # type: int


class GstVideoSource(GstPipeline):
    """Gstreamer Video Source Base Class

    Usage Example:
        >>> width, height, num_buffers = 1920, 1080, 100
        >>> caps_filter = 'capsfilter caps=video/x-raw,format=RGB,width={},height={}'.format(width, height)
        >>> command = 'videotestsrc num-buffers={} ! {} ! appsink emit-signals=True sync=false'.format(
        ...     num_buffers, caps_filter)
        >>> with GstVideoSource(command) as pipeline:
        ...     buffers = []
        ...     while len(buffers) < num_buffers:
        ...         buffer = pipeline.pop()
        ...         if buffer:
        ...             buffers.append(buffer)
        ...     print('Got: {} buffers'.format(len(buffers)))
        >>>
    """

    def __init__(self, command: str, leaky: bool = False, max_buffers_size: int = 100):
        """
        :param command: gst-launch-1.0 command (last element: appsink)
        """
        super(GstVideoSource, self).__init__(command)

        self._sink = None    # GstApp.AppSink
        self._counter = 0

        queue_cls = LeakyQueue if leaky else queue.Queue
        self._queue = queue_cls(maxsize=max_buffers_size)  # Queue of GstBuffer

    @property
    def total_buffers_count(self) -> int:
        """Total read buffers count """
        return self._counter

    @staticmethod
    def _clean_queue(q: queue.Queue):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def _on_drop(self, queue: base.LeakyQueue, buffer: GstBuffer) -> None:
        self.log.warning(
            'Buffer #%d for %s is dropped (totally dropped %d buffers)',
            int(buffer.pts / buffer.duration), self, queue.dropped)

    def _on_pipeline_init(self):
        """Sets additional properties for plugins in Pipeline"""

        appsinks = self.get_by_cls(GstApp.AppSink)
        self._sink = appsinks[0] if len(appsinks) == 1 else None
        if not self._sink:
            # TODO: force pipeline to have appsink
            raise ValueError("%s not found", GstApp.AppSink)

        if self._sink:
            self._sink.connect("new-sample", self._on_buffer, None)

    def _extract_buffer(self, sample: Gst.Sample) -> typ.Optional[GstBuffer]:
        """Converts Gst.Sample to GstBuffer"""
        buffer = sample.get_buffer()
        caps = sample.get_caps()

        cnt = buffer.n_memory()
        if cnt <= 0:
            self.log.warning("%s No data in Gst.Buffer", self)
            return None

        caps_ = caps.get_structure(0)

        memory = buffer.get_memory(0)
        if not memory:
            self.log.warning("%s No Gst.Memory in Gst.Buffer", self)
            return None

        video_format = GstVideo.VideoFormat.from_string(
            caps_.get_value('format'))

        w, h = caps_.get_value('width'), caps_.get_value('height')
        c = get_num_channels(video_format)
        dtype = get_np_dtype(np.uint8)

        array = np.ndarray((h, w, c), dtype=dtype,
                           buffer=buffer.extract_dup(0, buffer.get_size()))

        return GstBuffer(data=array, pts=buffer.pts, dts=buffer.dts,
                         duration=buffer.duration, offset=buffer.offset)

    def _on_buffer(self, sink: GstApp.AppSink, data: typ.Any) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if isinstance(sample, Gst.Sample):
            self._queue.put(self._extract_buffer(sample))
            self._counter += 1

            return Gst.FlowReturn.OK

        self.log.error("Error : Not expected buffer type: %s != %s. %s", type(sample), Gst.Sample, self)
        return Gst.FlowReturn.ERROR

    def pop(self, timeout: float = 0.1) -> typ.Optional[GstBuffer]:

        if self.shutdown_requested:
            self.log.warning("Warning %s: %s", self, "Can't pop buffer. Shutdown Requested ")
            return None

        if not self._sink:
            raise RuntimeError('Sink {} is not initialized'.format(Gst.AppSink))

        buffer = None
        while (self.is_active or not self._queue.empty()) and not buffer:
            if self.shutdown_requested and self._queue.empty():
                break

            try:
                buffer = self._queue.get(timeout=timeout)
            except queue.Empty:
                pass

        return buffer

    @property
    def queue_size(self):
        return self._queue.qsize()

    def shutdown(self, timeout: int = 1, eos: bool = False):
        super().shutdown(timeout=timeout, eos=eos)

        self._clean_queue(self._queue)
