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

./build
./install_pygst.sh
```

#### pip-package
    pip install git+https://github.com/jackersson/pygst-utils.git

### Test
```bash
PYTHONPATH=. pytest tests/ -s --verbose
```

### Tools
#### [Make Gst.Buffer writable](http://lifestyletransfer.com/how-to-make-gstreamer-buffer-writable-in-python/)
    from pygst_utils import map_gst_buffer
    with map_gst_buffer(pbuffer, Gst.MapFlags.READ | Gst.MapFlags.WRITE) as mapped:
                // do_something with mapped

#### Make Gst.Memory writable
    from pygst_utils import map_gst_memory
    with map_gst_memory(memory, Gst.MapFlags.READ | Gst.MapFlags.WRITE) as mapped:
                // do_something with mapped

#### Get Gst.Buffer shape (width,height) from Gst.Caps
    from pygst_utils import get_buffer_size
    ret, (width, height) = get_buffer_size(Gst.Caps)

#### Convert Gst.Buffer to np.ndarray
    from pygst_utils import gst_buffer_to_ndarray, gst_buffer_with_pad_to_ndarray

    array = gst_buffer_to_ndarray(Gst.Buffer, width, height, channels)
    # or
    array = gst_buffer_with_pad_to_ndarray(Gst.Buffer, Gst.Pad, channels)


### Metadata

#### [Object Info MedataData](https://github.com/jackersson/pygst-utils/blob/master/pygst_utils/gst_objects_info_meta.py)

       x
       y
       width
       height
       confidence
       class_name
       track_id

