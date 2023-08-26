import math
import typing as typ
from fractions import Fraction

import numpy as np

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GstVideo  # noqa:F401,F402

from .gst_hacks import map_gst_buffer  # noqa:F401,F402


BITS_PER_BYTE = 8

_ALL_VIDEO_FORMATS = [GstVideo.VideoFormat.from_string(
    f.strip()) for f in GstVideo.VIDEO_FORMATS_ALL.strip('{ }').split(',')]


def has_flag(value: GstVideo.VideoFormatFlags,
             flag: GstVideo.VideoFormatFlags) -> bool:

    # in VideoFormatFlags each new value is 1 << 2**{0...8}
    return bool(value & (1 << max(1, math.ceil(math.log2(int(flag))))))


def _get_num_channels(fmt: GstVideo.VideoFormat) -> int:
    """
        -1: means complex format (YUV, ...)
    """
    frmt_info = GstVideo.VideoFormat.get_info(fmt)
    
    # temporal fix
    if fmt == GstVideo.VideoFormat.BGRX:
        return 4
    
    if has_flag(frmt_info.flags, GstVideo.VideoFormatFlags.ALPHA):
        return 4

    if has_flag(frmt_info.flags, GstVideo.VideoFormatFlags.RGB):
        return 3

    if has_flag(frmt_info.flags, GstVideo.VideoFormatFlags.YUV):
        return 3

    if has_flag(frmt_info.flags, GstVideo.VideoFormatFlags.GRAY):
        return 1

    return -1


_ALL_VIDEO_FORMAT_CHANNELS = {fmt: _get_num_channels(fmt) for fmt in _ALL_VIDEO_FORMATS}


def get_num_channels(fmt: GstVideo.VideoFormat):
    return _ALL_VIDEO_FORMAT_CHANNELS[fmt]


_DTYPES = {
    16: np.int16,
}


def get_np_dtype(fmt: GstVideo.VideoFormat) -> np.number:
    format_info = GstVideo.VideoFormat.get_info(fmt)
    return _DTYPES.get(format_info.bits, np.uint8)


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


def gst_buffer_to_ndarray(buffer: Gst.Buffer, *, width: int, height: int, channels: int,
                          dtype: np.dtype, bpp: int = 1, do_copy: bool = False) -> np.ndarray:
    """Converts Gst.Buffer with known format (w, h, c, dtype) to np.ndarray"""

    result = None
    if do_copy:
        result = np.ndarray(buffer.get_size() // (bpp // BITS_PER_BYTE),
                            buffer=buffer.extract_dup(0, buffer.get_size()), dtype=dtype)
    else:
        with map_gst_buffer(buffer, Gst.MapFlags.READ) as mapped:
            result = np.ndarray(buffer.get_size() // (bpp // BITS_PER_BYTE),
                                buffer=mapped, dtype=dtype)
    if channels > 0:
        result = result.reshape(height, width, channels).squeeze()
    return result


def gst_buffer_with_pad_to_ndarray(buffer: Gst.Buffer, pad: Gst.Pad, do_copy: bool = False) -> np.ndarray:
    """Converts Gst.Buffer with Gst.Pad (stores Gst.Caps) to np.ndarray """
    return gst_buffer_with_caps_to_ndarray(buffer, pad.get_current_caps(), do_copy=do_copy)


def gst_buffer_with_caps_to_ndarray(buffer: Gst.Buffer, caps: Gst.Caps, do_copy: bool = False) -> np.ndarray:
    """ Converts Gst.Buffer with Gst.Caps (stores buffer info) to np.ndarray """

    structure = caps.get_structure(0)  # Gst.Structure

    width, height = structure.get_value("width"), structure.get_value("height")

    # GstVideo.VideoFormat
    video_format = gst_video_format_from_string(structure.get_value('format'))

    channels = get_num_channels(video_format)

    dtype = get_np_dtype(video_format)  # np.dtype

    format_info = GstVideo.VideoFormat.get_info(video_format)  # GstVideo.VideoFormatInfo

    return gst_buffer_to_ndarray(buffer, width=width, height=height, channels=channels,
                                 dtype=dtype, bpp=format_info.bits, do_copy=do_copy)


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
    return plugins_[0] + "".join(["{} {}".format('' if pl[-1] == '.' else ' !', pl) for pl in plugins_[1:]])
