cd gstreamer/3rd_party

export GST_PREFIX=${GST_PREFIX:-$(dirname $(dirname $(which python)))} 

./build.sh