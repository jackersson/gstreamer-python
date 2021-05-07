ROOT=$PWD

echo "PWD: $PWD"

# Gstreamer
#cd gstreamer
#./build.sh
BUILD_OPTS=""
if [[ -n "$GST_PREFIX" ]]; then
    BUILD_OPTS="${BUILD_OPTS} --prefix ${GST_PREFIX} --libdir=${GST_PREFIX}/lib"
fi
echo $BUILD_OPTS
meson setup $BUILD_OPTS build
#ninja -C build
ninja -C build install
