ROOT=$PWD

echo "PWD: $PWD"

# Gstreamer
#cd gstreamer
#./build.sh
meson build
#ninja -C build
ninja -C build install
