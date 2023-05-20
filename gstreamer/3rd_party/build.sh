ROOT=$PWD

echo "PWD: $PWD"
rm -rf ./build
# Gstreamer
#cd gstreamer
#./build.sh
#--prefix=~/myinstalls/
if [ -z ${PREFIX+x} ]; then 
    echo "PREFIX is unset"; 
    meson -Dbuildtype=debug build

else 
    echo "PREFIX is set to '$PREFIX'"; 
    meson --prefix=$PREFIX -Dbuildtype=debug build

fi
# compile and install
ninja -C build install
