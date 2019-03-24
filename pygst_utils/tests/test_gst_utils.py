import sys
from unittest import TestCase

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from pygst_utils.gst_objects_info_meta import gst_meta_get, gst_meta_write, gst_meta_remove

Gst.init(sys.argv)


class TestPyGstMeta(TestCase):

    def test_gst_objects_info_meta(self):
        buffer = Gst.Buffer.new_wrapped(b"test_buffer")

        assert len(gst_meta_get(buffer)) == 0

        objects = [
            {'class_name': "person", 'bounding_box': [10, 10, 100, 100], 'confidence': 0.6},
            {'class_name': "person", 'bounding_box': [10, 10, 100, 100], 'confidence': 0.6},
        ]

        gst_meta_write(buffer, objects)

        result = gst_meta_get(buffer)
        assert len(result) == len(objects)
        for py_object, gst_object in zip(objects, result):
            assert py_object['bounding_box'] == gst_object['bounding_box']
            assert py_object['class_name'] == gst_object['class_name']
            self.assertAlmostEqual(py_object['confidence'], gst_object['confidence'])

        gst_meta_remove(buffer)
        assert len(gst_meta_get(buffer)) == 0


