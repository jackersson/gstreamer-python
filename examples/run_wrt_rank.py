"""
Set decoder priority for video processing pipelines

Examples:
    python examples/run_wrt_rank.py -d "avdec_h264"

    python examples/run_wrt_rank.py -p "rtspsrc location=rtsp://... ! decodebin ! nvvideoconvert ! gtksink" -d "nvv4l2decoder"
"""

import time
import argparse

from gstreamer import Gst, GstContext, GstPipeline, GObject
import gstreamer.utils as utils

TARGET_PLUGIN_NAME = "avdec_h264"
FREE_RTSP_SOURCE = [
    "rtsp://freja.hiof.no:1935/rtplive/definst/hessdalen03.stream",
    "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov"
]
DEFAULT_PIPELINE = "rtspsrc location={} ! decodebin ! videoconvert ! gtksink".format(
    FREE_RTSP_SOURCE[0])

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pipeline", required=False,
                default=DEFAULT_PIPELINE, help="Gstreamer pipeline without gst-launch")

ap.add_argument("-d", "--decoder", required=False,
                default=TARGET_PLUGIN_NAME, help="Specify plugin name (decoder) to use")

args = vars(ap.parse_args())

pipeline = args['pipeline']
target_element_name = args['decoder']

# Filter target elements with Gst.Constans (Gst.ELEMENT_FACTORY_TYPE_*)
# https://lazka.github.io/pgi-docs/#Gst-1.0/constants.html#constants
filt = Gst.ELEMENT_FACTORY_TYPE_DECODER  # only decoders
filt |= Gst.ELEMENT_FACTORY_TYPE_MEDIA_VIDEO  # only for video

# https://lazka.github.io/pgi-docs/Gst-1.0/classes/ElementFactory.html#Gst.ElementFactory.list_get_elements
# list all elements by filter
factories = Gst.ElementFactory.list_get_elements(
    filt, Gst.Rank.MARGINAL)  # Gst.ElementFactory[]

# get target
target_element = Gst.ElementFactory.find(
    target_element_name)  # Gst.ElementFactory

# factory object extends Gst.PluginFeature
# with get_name(), get_rank()
# https://lazka.github.io/pgi-docs/Gst-1.0/classes/PluginFeature.html#gst-pluginfeature
# sort by plugin rank in descending order
factories = sorted(factories, key=lambda f: f.get_rank(), reverse=True)

# Gst.Rank
# https://lazka.github.io/pgi-docs/index.html#Gst-1.0/enums.html#Gst.Rank
# get max rank element
max_rank_element = factories[0]
if max_rank_element.get_name() != target_element_name:
    print("--- Before ---")
    print("Max rank plugin:", max_rank_element.get_name(),
          "(", max_rank_element.get_rank(), ")")
    print("Rank of target plugin:", target_element.get_name(),
          "(", target_element.get_rank(), ")")

    print("--- After ---")

# Increase target's element rank
target_element.set_rank(max_rank_element.get_rank() + 1)
print("Rank of target plugin:", target_element.get_name(),
      "(", target_element.get_rank(), ")")

pipeline_str = pipeline

with GstContext(), GstPipeline(pipeline_str) as p:
    try:
        while not p.is_done:
            time.sleep(1)
    except Exception:
        pass
    finally:
        # print all elements and notify of target plugin presence
        elements = [el.get_factory().get_name()
                    for el in p.pipeline.iterate_recurse()]
        print("All elements: ", elements)
        print("Target element ({}) is {}".format(target_element_name,
                                                 'present' if target_element_name in set(elements) else "missing"))
