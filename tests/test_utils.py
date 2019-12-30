import os
import time
import typing as typ
from random import randint
from fractions import Fraction

import numpy as np
import pytest

from pygst_utils import GstVideoSource, GstVideoSink, GstVideo


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
        with GstVideoSink(command, width=w, height=h, video_frmt=frame.buffer_format) as pipeline:
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
        with GstVideoSource(command) as pipeline:

            num_read = 0
            while num_read < num_buffers:
                buffer = pipeline.pop()
                if buffer:
                    num_read += 1
                    h, w = buffer.data.shape[:2]
                    assert h == height and w == width

            assert pipeline.total_buffers_count == num_buffers
