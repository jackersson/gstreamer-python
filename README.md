### Install
#### Install OS packages
- [How to install Gstreamer on Ubuntu](http://lifestyletransfer.com/how-to-install-gstreamer-on-ubuntu/)
- [How to install Gstreamer Python Bindings](http://lifestyletransfer.com/how-to-install-gstreamer-python-bindings/)

#### in-place
```bash
python3 -m venv venv

source venv/bin/activate
pip install --upgrade wheel pip setuptools
pip install --upgrade --requirement requirements.txt

./build-3rd-party.sh
./build-gst-python.sh
```

#### pip-package
    pip install git+https://github.com/jackersson/gstreamer-python.git

### Test
```bash
PYTHONPATH=. pytest tests/ -s --verbose
```

### Tools

#### Setup
- By default Gstreamer tools use **libgstreamer-1.0.so.0**
```bash
export LIB_GSTREAMER_PATH=libgstreamer-1.0.so.0
```
Export **LIB_GSTREAMER_PATH** with custom path to **libgstreamer.so**

##### Setup Log Level
```bash
export GST_PYTHON_LOG_LEVEL=0, 1, 2, 3, 4, 5
```

#### [Make Gst.Buffer writable](http://lifestyletransfer.com/how-to-make-gstreamer-buffer-writable-in-python/)
    from gstreamer import map_gst_buffer
    with map_gst_buffer(pbuffer, Gst.MapFlags.READ | Gst.MapFlags.WRITE) as mapped:
                // do_something with mapped

#### Make Gst.Memory writable
    from gstreamer import map_gst_memory
    with map_gst_memory(memory, Gst.MapFlags.READ | Gst.MapFlags.WRITE) as mapped:
                // do_something with mapped

#### Get Gst.Buffer shape (width,height) from Gst.Caps
    from gstreamer import get_buffer_size
    ret, (width, height) = get_buffer_size(Gst.Caps)

#### Convert Gst.Buffer to np.ndarray
    from gstreamer import gst_buffer_to_ndarray, gst_buffer_with_pad_to_ndarray

    array = gst_buffer_to_ndarray(Gst.Buffer, width, height, channels)
    # or
    array = gst_buffer_with_pad_to_ndarray(Gst.Buffer, Gst.Pad, channels)

### GstPipeline
- With **GstPipeline** run any **gst-launch** pipeline in Python
```bash
from gstreamer import GstPipeline

command = "videotestsrc num-buffers=100 ! fakesink sync=false"
with GstPipeline(command) as pipeline:
    ...
```


#### GstVideoSource based on AppSink
- With **GstVideoSource** run any **gst-launch** pipeline and receive buffers in Python
```bash
from gstreamer import GstVideoSource

width, height, num_buffers = 1920, 1080, 100
caps_filter = 'capsfilter caps=video/x-raw,format=RGB,width={},height={}'.format(width, height)
command = 'videotestsrc num-buffers={} ! {} ! appsink emit-signals=True sync=false'.format(
num_buffers, caps_filter)
with GstVideoSource(command) as pipeline:
    buffers = []
    while len(buffers) < num_buffers:
        buffer = pipeline.pop()
        if buffer:
            buffers.append(buffer)
    print('Got: {} buffers'.format(len(buffers)))
```

#### GstVideoSink based on AppSrc
- With **GstVideoSink** push buffers in Python to any **gst-launch** pipeline
```bash
from gstreamer import GstVideoSink

width, height = 1920, 1080
command = "appsrc emit-signals=True is-live=True ! videoconvert ! fakesink sync=false"
with GstVideoSink(command, width=width, height=height) as pipeline:
    for _ in range(10):
        pipeline.push(buffer=np.random.randint(low=0, high=255, size=(height, width, 3), dtype=np.uint8))
```

### Metadata

#### [Object Info MedataData](https://github.com/jackersson/gstreamer-python/blob/master/gstreamer/gst_objects_info_meta.py)

       x
       y
       width
       height
       confidence
       class_name
       track_id

