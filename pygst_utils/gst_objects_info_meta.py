import os
from ctypes import *
from typing import List

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


class GstObjectInfo(Structure):
    _fields_ = [("x", c_int),
                ("y", c_int),
                ("width", c_int),
                ("height", c_int),
                ("confidence", c_float),
                ("class_name", c_char_p)]


class GstObjectInfoArray(Structure):
    _fields_ = [("items", POINTER(GstObjectInfo)),
                ("size", c_int)]

GstObjectInfoArrayPtr = POINTER(GstObjectInfoArray)

cwd = os.path.dirname(os.path.abspath(__file__))
libc = CDLL(os.path.join(cwd, "3rd_party/gstreamer/build/libgst_objects_info_meta.so"))

libc.gst_buffer_add_objects_info_meta.argtypes = [c_void_p, GstObjectInfoArrayPtr]
libc.gst_buffer_add_objects_info_meta.restype = c_void_p

libc.gst_buffer_get_objects_info_meta.argtypes = [c_void_p]
libc.gst_buffer_get_objects_info_meta.restype = GstObjectInfoArrayPtr

libc.gst_buffer_remove_objects_info_meta.argtypes = [c_void_p]
libc.gst_buffer_remove_objects_info_meta.restype = c_bool


def to_gst_objects_info(objects: List[dict]) -> GstObjectInfoArray:
    """ Converts List of objects to GstObjectInfoMeta """
    gst_objects_info = GstObjectInfoArray()
    gst_objects_info.size = len(objects)
    gst_objects_info.items = (GstObjectInfo * gst_objects_info.size)()

    for i, obj in enumerate(objects):
        x, y, width, height = obj['bounding_box']
        gst_objects_info.items[i] = (x, y, width, height,
                                     obj["confidence"],
                                     obj["class_name"].encode("utf-8"))

    return gst_objects_info


def to_list(gst_object_info: GstObjectInfoArray) -> List[dict]:
    """ Converts GstObjectInfoMeta to List of objects"""
    objects = []
    for i in range(gst_object_info.size):
        obj = gst_object_info.items[i]
        objects.append({"bounding_box": [obj.x, obj.y, obj.width, obj.height],
                        "confidence": obj.confidence,
                        "class_name": obj.class_name.decode("utf-8") if obj.class_name else ""})
    return objects


def gst_meta_write(buffer: Gst.Buffer, objects: List[dict]):
    """ Writes List of objects to Gst.Buffer"""
    gst_objects_info = to_gst_objects_info(objects)
    _ = libc.gst_buffer_add_objects_info_meta(hash(buffer), gst_objects_info)


def gst_meta_get(buffer: Gst.Buffer) -> List[dict]:
    """ Gets List of objects from Gst.Buffer"""
    res = libc.gst_buffer_get_objects_info_meta(hash(buffer))
    return to_list(res.contents)


def gst_meta_remove(buffer: Gst.Buffer):
    """ Removes all objects from Gst.Buffer """
    libc.gst_buffer_remove_objects_info_meta(hash(buffer))
