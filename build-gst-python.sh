
# After PyGObject (https://lazka.github.io/pgi-docs/) installed
# Run current script to override Gstreamer related Scripts

LIBPYTHONPATH=""
PYTHON=${PYTHON:-$(which python3)}
GST_VERSION=${GST_VERSION:-$(gst-launch-1.0 --version | grep version | tr -s ' ' '\n' | tail -1 | $PYTHON -c "import sys; print('.'.join(next(sys.stdin).strip().split('.')[:2]))")}

# Ensure pygst to be installed in current environment
LIBPYTHONPATH=$($PYTHON -c 'from distutils import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

GST_PREFIX=${GST_PREFIX:-$(dirname $(dirname $(which python)))} 

MESON_VERSION=$(meson --version)
PYTHON_VERSION=$($PYTHON -c 'import sys; print(sys.version.split()[0])')

echo "Python Executable: $PYTHON"
echo "Python Version: $PYTHON_VERSION"
echo "Python Library Path: $LIBPYTHONPATH"
echo "Current Python Path $GST_PREFIX"
echo "Gstreamer Version: $GST_VERSION"
echo "Meson Version: $MESON_VERSION"

TEMP_DIR="temp"
mkdir $TEMP_DIR
cd $TEMP_DIR

# Build gst-python
git clone https://github.com/GStreamer/gst-python.git
cd gst-python

export PYTHON=$PYTHON
git checkout $GST_VERSION

# py >= 3.8 && meson >= 0.53 && gstreamer < 1.19, apply patch
TEST_PYVER=$($PYTHON -c "from packaging.version import parse; ans = 1 if parse('$PYTHON_VERSION') >= parse('3.8.0') else 0; print(ans)")
TEST_MESONVER=$($PYTHON -c "from packaging.version import parse; ans = 1 if parse('$MESON_VERSION') >= parse('0.53.0') else 0; print(ans)")
TEST_GSTVER=$($PYTHON -c "from packaging.version import parse; ans = 1 if parse('$GST_VERSION') < parse('1.19') else 0; print(ans)")
if [ 1 -eq $TEST_PYVER -a 1 -eq $TEST_MESONVER -a 1 -eq $TEST_GSTVER ] ; then
    sed -i 's|python.dependency(required : true)|python.dependency(embed:true, required : true)|g' meson.build
    sed -i "s|meson_version : '.*',|meson_version : '>= 0.53.0',|g" meson.build
fi

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

