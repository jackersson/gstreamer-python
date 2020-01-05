## GstContext
### Purpose
- Hides GObject.MainLoop routine in single class

### Example
```bash
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
- Hides Gst.Pipeline creation and message handling in single class

### Example
```bash
import time
from gstreamer import GstContext, GstPipeline

with GstContext():
  with GstPipeline("videotestsrc num-buffers=100 ! autovideosink") as p
    while not p.is_done:
      time.sleep(1)  
```
