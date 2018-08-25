import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import numpy as np

from .gst_hacks import map_gst_buffer


def gst_buffer_to_ndarray(buffer, width, height, channels=3):
    """
        Converts Gst.Buffer with known format (width, height, channels) to np.ndarray

        :param buffer: 
        :type buffer: Gst.Buffer

        :param width: 
        :type width: int

        :param height: 
        :type height: int

        :param channels: 
        :type channels: int

        :rtype: np.ndarray (height, width, channels)
    """
    with map_gst_buffer(buffer, Gst.MapFlags.READ) as mapped:
        # TODO: Check format
        return np.ndarray((height, width, channels), buffer=mapped, dtype=np.uint8)


def gst_buffer_with_pad_to_ndarray(buffer, pad, channels=3):
    """

        Converts Gst.Buffer with Gst.Pad (stores buffer format) to np.ndarray

        :param buffer: 
        :type buffer: Gst.Buffer

        :param pad: Gst.Pad allow access to buffer format (width, height)
        :type pad: Gst.Pad

        :param channels: 
        :type channels: int

        :rtype: np.ndarray (height, width, channels)
    """

    success, (width, height) = get_buffer_size(pad.get_current_caps())
    if not success:
        raise ValueError('Invalid buffer size.')

    return to_ndarray(buffer, width, height, channels)