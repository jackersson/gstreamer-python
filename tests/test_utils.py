import os
import time
import math
import typing as typ
from random import randint
from fractions import Fraction

import numpy as np
import pytest

from pygst_utils import GstVideo
import pygst_utils as pygst


NUM_BUFFERS = 10
WIDTH, HEIGHT = 1920, 1080
FPS = 15
FORMAT = "RGB"

Frame = typ.NamedTuple(
    'Frame', [
        ('buffer_format', GstVideo.VideoFormat),
        ('buffer', np.ndarray),
    ])


FRAMES = [
    Frame(GstVideo.VideoFormat.RGB, np.random.randint(
        low=0, high=255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)),
    Frame(GstVideo.VideoFormat.RGBA, np.random.randint(
        low=0, high=255, size=(HEIGHT, WIDTH, 4), dtype=np.uint8)),
    Frame(GstVideo.VideoFormat.GRAY8, np.random.randint(
        low=0, high=255, size=(HEIGHT, WIDTH), dtype=np.uint8)),
    Frame(GstVideo.VideoFormat.GRAY16_BE, np.random.uniform(
        0, 1, (HEIGHT, WIDTH)).astype(np.float32))
]


def test_video_sink():
    num_buffers = NUM_BUFFERS

    command = "appsrc emit-signals=True is-live=True ! videoconvert ! fakesink sync=false"

    for frame in FRAMES:
        h, w = frame.buffer.shape[:2]
        with pygst.GstVideoSink(command, width=w, height=h, video_frmt=frame.buffer_format) as pipeline:
            assert pipeline.total_buffers_count == 0

            # wait pipeline to initialize
            max_num_tries, num_tries = 5, 0
            while not pipeline.is_active and num_tries <= max_num_tries:
                time.sleep(.1)
                num_tries += 1

            assert pipeline.is_active

            for _ in range(num_buffers):
                pipeline.push(frame.buffer)

            assert pipeline.total_buffers_count == num_buffers


def test_video_source():
    num_buffers = NUM_BUFFERS
    width, height = WIDTH, HEIGHT

    formats = [GstVideo.VideoFormat.to_string(f.buffer_format) for f in FRAMES]

    for fmt in formats:
        caps_filter = 'capsfilter caps=video/x-raw,format={},width={},height={}'.format(
            fmt, width, height)
        command = 'videotestsrc num-buffers={} ! {} ! appsink emit-signals=True sync=false'.format(
            num_buffers, caps_filter)
        with pygst.GstVideoSource(command) as pipeline:

            num_read = 0
            while num_read < num_buffers:
                buffer = pipeline.pop()
                if buffer:
                    num_read += 1
                    h, w = buffer.data.shape[:2]
                    assert h == height and w == width

            assert pipeline.total_buffers_count == num_buffers


def test_gst_pipeline():
    command = "videotestsrc num-buffers=100 ! fakesink sync=false"
    with pygst.GstPipeline(command) as pipeline:
        assert isinstance(pipeline, pygst.GstPipeline)


@pytest.mark.skip
def test_video_src_to_source():

    num_buffers = NUM_BUFFERS

    for frame in FRAMES:
        buffer = frame.buffer
        h, w = buffer.shape[:2]

        sink_command = "appsrc emit-signals=True is-live=True ! videoconvert ! fakesink sync=false"

        fmt = GstVideo.VideoFormat.to_string(frame.buffer_format)
        caps_filter = f'capsfilter caps=video/x-raw,format={fmt},width={w},height={h}'
        source_command = f'videotestsrc num-buffers={num_buffers} ! {caps_filter} ! appsink emit-signals=True sync=false'

        with pygst.GstVideoSink(sink_command, width=w, height=h, video_frmt=frame.buffer_format) as sink, \
                pygst.GstVideoSource(source_command) as src:
            assert sink.total_buffers_count == 0

            # wait pipeline to initialize
            max_num_tries, num_tries = 5, 0
            while not sink.is_active and num_tries <= max_num_tries:
                time.sleep(.1)
                num_tries += 1

            assert sink.is_active

            num_read = 0
            while num_read < num_buffers:
                buffer = src.pop()
                if buffer:
                    num_read += 1
                    sink.push(buffer.data, pts=buffer.pts,
                              dts=buffer.dts, offset=buffer.offset)

            assert src.total_buffers_count == num_buffers
            assert sink.total_buffers_count == num_buffers


def test_metadata():
    np_buffer = np.random.randint(
        low=0, high=255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)

    gst_buffer = pygst.numpy_to_gst_buffer(np_buffer)

    from pygst_utils.gst_objects_info_meta import gst_meta_write, gst_meta_get, gst_meta_remove

    objects = [
        {'class_name': "person", 'bounding_box': [
            8, 10, 100, 100], 'confidence': 0.6, 'track_id': 1},
        {'class_name': "person", 'bounding_box': [
            10, 9, 120, 110], 'confidence': 0.67, 'track_id': 2},
    ]

    # no metadata at the beginning
    assert len(gst_meta_get(gst_buffer)) == 0

    # write metadata
    gst_meta_write(gst_buffer, objects)

    # read metadata
    meta_objects = gst_meta_get(gst_buffer)
    assert len(gst_meta_get(gst_buffer)) == len(objects)

    for gst_meta_obj, py_obj in zip(meta_objects, objects):
        for key, val in py_obj.items():
            if isinstance(gst_meta_obj[key], float):
                assert math.isclose(gst_meta_obj[key], val, rel_tol=1e-07)
            else:
                assert gst_meta_obj[key] == val

    # remove metadata
    gst_meta_remove(gst_buffer)
    assert len(gst_meta_get(gst_buffer)) == 0
