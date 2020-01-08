import typing as typ
from fractions import Fraction

import numpy as np

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GstVideo  # noqa:F401,F402

from .gst_hacks import map_gst_buffer  # noqa:F401,F402


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


def get_num_channels(fmt: GstVideo.VideoFormat) -> int:
    return _CHANNELS[fmt]


_DTYPES = {
    GstVideo.VideoFormat.GRAY16_BE: np.float16
}


def get_np_dtype(fmt: GstVideo.VideoFormat) -> np.number:
    return _DTYPES.get(fmt, np.uint8)


def fraction_to_str(fraction: Fraction) -> str:
    """Converts fraction to str"""
    return '{}/{}'.format(fraction.numerator, fraction.denominator)


def gst_state_to_str(state: Gst.State) -> str:
    """Converts Gst.State to str representation

    Explained: https://lazka.github.io/pgi-docs/Gst-1.0/classes/Element.html#Gst.Element.state_get_name
    """
    return Gst.Element.state_get_name(state)


def gst_video_format_from_string(frmt: str) -> GstVideo.VideoFormat:
    return GstVideo.VideoFormat.from_string(frmt)


def gst_buffer_to_ndarray(buffer: Gst.Buffer, *, width: int, height: int, channels: int, dtype: np.dtype) -> np.ndarray:
    """Converts Gst.Buffer with known format (w, h, c, dtype) to np.ndarray"""
    with map_gst_buffer(buffer, Gst.MapFlags.READ) as mapped:
        return np.ndarray((height, width, channels), buffer=mapped, dtype=dtype)


def gst_buffer_with_pad_to_ndarray(buffer: Gst.Buffer, pad: Gst.Pad) -> np.ndarray:
    """Converts Gst.Buffer with Gst.Pad (stores Gst.Caps) to np.ndarray """
    return gst_buffer_with_caps_to_ndarray(buffer, pad.get_current_caps())


def gst_buffer_with_caps_to_ndarray(buffer: Gst.Buffer, caps: Gst.Caps) -> np.ndarray:
    """ Converts Gst.Buffer with Gst.Caps (stores buffer info) to np.ndarray """

    structure = caps.get_structure(0)  # Gst.Structure

    width, height = structure.get_value("width"), structure.get_value("height")

    # GstVideo.VideoFormat
    video_format = gst_video_format_from_string(structure.get_value('format'))

    channels = get_num_channels(video_format)
    dtype = get_np_dtype(video_format)  # np.dtype

    return gst_buffer_to_ndarray(buffer, width=width, height=height, channels=channels, dtype=dtype)


def get_buffer_size_from_gst_caps(caps: Gst.Caps) -> typ.Tuple[int, int]:
    """Returns buffers width, height from Gst.Caps """
    structure = caps.get_structure(0)  # Gst.Structure
    return structure.get_value("width"), structure.get_value("height")


def ndarray_to_gst_buffer(array: np.ndarray) -> Gst.Buffer:
    """Converts numpy array to Gst.Buffer"""
    return Gst.Buffer.new_wrapped(array.tobytes())


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

    if len(plugins) < 2:
        return ""

    plugins_ = flatten_list(plugins)

    # <!> between plugins (except tee)
    return plugins_[0] + "".join([f"{'' if pl[-1] == '.' else ' !'} {pl}" for pl in plugins_[1:]])
