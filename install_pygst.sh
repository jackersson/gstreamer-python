
# After PyGObject (https://lazka.github.io/pgi-docs/) installed
# Run current script to override Gstreamer related Scripts

LIBPYTHONPATH=""
PYTHON=${PYTHON:-/usr/bin/python3}
GST_VERSION=${GST_VERSION:"1.14.5"}


# Ensure pygst to be installed in current environment
LIBPYTHON=$($PYTHON -c 'from distutils import sysconfig; print(sysconfig.get_config_var("LDLIBRARY"))')
LIBPYTHONPATH=$(dirname $(ldconfig -p | grep -w $LIBPYTHON | head -1 | tr ' ' '\n' | grep /))
GST_VERSION=$(gst-launch-1.0 --version | grep version | tr -s ' ' '\n' | tail -1)

PYTHON_LIB=$(dirname $(dirname $(which python)))

echo "Python Executable: $PYTHON"
echo "Python Library Path: $LIBPYTHONPATH"
echo "Current Python Path $PYTHON_LIB"
echo "Gstreamer Version: $GST_VERSION"

TEMP_DIR="temp"
mkdir $TEMP_DIR
cd $TEMP_DIR

# Build gst-python
git clone https://github.com/GStreamer/gst-python.git
cd gst-python

export PYTHON=$PYTHON
git checkout $GST_VERSION

./autogen.sh --disable-gtk-doc --noconfigure
./configure --with-libpython-dir=$LIBPYTHONPATH --prefix $PYTHON_LIB
make
make install

cd ../..

# Clear folder
rm -rf $TEMP_DIR

