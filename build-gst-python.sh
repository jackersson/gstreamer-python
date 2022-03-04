
# After PyGObject (https://lazka.github.io/pgi-docs/) installed
# Run current script to override Gstreamer related Scripts

LIBPYTHONPATH=""
PYTHON=${PYTHON:-$(which python3)}
GST_VERSION=${GST_VERSION:-$(gst-launch-1.0 --version | grep version | tr -s ' ' '\n' | tail -1 | $PYTHON -c "import sys; print('.'.join(next(sys.stdin).strip().split('.')[:2]))")}

# Ensure pygst to be installed in current environment
LIBPYTHONPATH=$($PYTHON -c 'from distutils import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

GST_PREFIX=${GST_PREFIX:-$(dirname $(dirname $(which python)))} 

echo "Python Executable: $PYTHON"
echo "Python Library Path: $LIBPYTHONPATH"
echo "Current Python Path $GST_PREFIX"
echo "Gstreamer Version: $GST_VERSION"

TEMP_DIR="temp"
mkdir $TEMP_DIR
cd $TEMP_DIR

# Build gst-python
git clone https://github.com/GStreamer/gst-python.git
cd gst-python

export PYTHON=$PYTHON
git checkout $GST_VERSION

#./autogen.sh --disable-gtk-doc --noconfigure
#./configure --with-libpython-dir=$LIBPYTHONPATH --prefix $GST_PREFIX
#make
#make install
meson --prefix=$GST_PREFIX --libdir=$GST_PREFIX/lib -Dpython=$PYTHON -Dlibpython-dir=$LIBPYTHONPATH -Dbuildtype=release build
#ninja -C build
ninja -C build install

cd ../..

# Clear folder
rm -rf $TEMP_DIR

