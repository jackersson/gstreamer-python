## GstContext
### Purpose
- Hides [GObject.MainLoop](https://lazka.github.io/pgi-docs/GLib-2.0/structs/MainLoop.html) routine in single class

### Example
```python
import time
from gstreamer import GstContext

with GstContext():
  ...
  
  # run pipeline 1
  ...
  # run pipeline N
  
  while any(pipeline):
    time.sleep(1)
```

## GstPipeline
### Purpose
- Hides [Gst.Pipeline](https://lazka.github.io/pgi-docs/Gst-1.0/classes/Pipeline.html) creation and message handling in single class

### Example
```python
import time
from gstreamer import GstContext, GstPipeline

with GstContext():
  with GstPipeline("videotestsrc num-buffers=100 ! autovideosink") as p
    while not p.is_done:
      time.sleep(1)  
```
