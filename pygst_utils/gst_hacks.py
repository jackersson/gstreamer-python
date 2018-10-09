'''
    Took from https://github.com/stb-tester/stb-tester/blob/master/_stbt/gst_hacks.py
'''
from ctypes import *
from contextlib import contextmanager


import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

_GST_PADDING = 4  # From gstconfig.h


class _GstMapInfo(Structure):
    _fields_ = [("memory", c_void_p),   # GstMemory *memory
                ("flags", c_int),       # GstMapFlags flags
                ("data", POINTER(c_byte)),    # guint8 *data
                ("size", c_size_t),     # gsize size
                ("maxsize", c_size_t),  # gsize maxsize
                ("user_data", c_void_p * 4),     # gpointer user_data[4]
                ("_gst_reserved", c_void_p * _GST_PADDING)]

_GST_MAP_INFO_POINTER = POINTER(_GstMapInfo)

_libgst = CDLL("libgstreamer-1.0.so.0")
_libgst.gst_buffer_map.argtypes = [c_void_p, _GST_MAP_INFO_POINTER, c_int]
_libgst.gst_buffer_map.restype = c_int

_libgst.gst_buffer_unmap.argtypes = [c_void_p, _GST_MAP_INFO_POINTER]
_libgst.gst_buffer_unmap.restype = None

_libgst.gst_mini_object_is_writable.argtypes = [c_void_p]
_libgst.gst_mini_object_is_writable.restype = c_int

_libgst.gst_memory_map.argtypes = [c_void_p, _GST_MAP_INFO_POINTER, c_int]
_libgst.gst_memory_map.restype = c_int

_libgst.gst_memory_unmap.argtypes = [c_void_p, _GST_MAP_INFO_POINTER]
_libgst.gst_memory_unmap.restype = None


@contextmanager
def map_gst_buffer(pbuffer, flags):
    """
        Map Gst.Buffer with READ/WRITE flags

        Example:
            with map_gst_buffer(pbuffer, Gst.MapFlags.READ | Gst.MapFlags.WRITE) as mapped:
                // do_something with mapped

        :param pbuffer:
        :type pbuffer: Gst.Buffer

        :param flags:
        :type flags: Gst.MapFlags

        :rtype: bool (success flag)
        :rtype: tuple (int, int) -> (width, height)
    """

    if pbuffer is None:
        raise TypeError("Cannot pass NULL to _map_gst_buffer")

    ptr = hash(pbuffer)
    if flags & Gst.MapFlags.WRITE and _libgst.gst_mini_object_is_writable(ptr) == 0:
        raise ValueError("Writable array requested but buffer is not writeable")

    mapping = _GstMapInfo()
    success = _libgst.gst_buffer_map(ptr, mapping, flags)
    if not success:
        raise RuntimeError("Couldn't map buffer")
    try:
        yield cast(
            mapping.data, POINTER(c_byte * mapping.size)).contents
    finally:
        _libgst.gst_buffer_unmap(ptr, mapping)


@contextmanager
def map_gst_memory(memory, flags):
    if memory is None:
        raise TypeError("Cannot pass NULL to _map_gst_buffer")

    ptr = hash(memory)
    if flags & Gst.MapFlags.WRITE and _libgst.gst_mini_object_is_writable(ptr) == 0:
        raise ValueError("Writable array requested but buffer is not writeable")

    mapping = _GstMapInfo()
    success = _libgst.gst_memory_map(ptr, mapping, flags)
    if not success:
        raise RuntimeError("Couldn't map buffer")
    try:
        yield cast(
            mapping.data, POINTER(c_byte * mapping.size)).contents
    finally:
        _libgst.gst_memory_unmap(ptr, mapping)



def get_buffer_size(caps):
    """
        Get Gst.Buffer's (width, height) from Gst.Caps

        :param caps:
        :type caps: Gst.Caps

        :rtype: bool (success flag)
        :rtype: tuple (int, int) -> (width, height)
    """

    caps_struct = caps.get_structure(0)
    (success, width) = caps_struct .get_int('width')
    if not success:
        return False, (0, 0)
    (success, height) = caps_struct .get_int('height')
    if not success:
        return False, (0, 0)
    return True, (width, height)