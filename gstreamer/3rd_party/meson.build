project('gst-rtp-viewport-metadata', 'c', version : '1.0.0.1', license : 'Proprietary')

plugins_install_dir = join_paths(get_option('libdir'), 'gstreamer-1.0')

cc = meson.get_compiler('c')

gst_version = meson.project_version()

api_version = '1.0'

gst_dep = dependency('gstreamer-1.0',
    fallback : ['gstreamer', 'gst_dep'])

subdir('gstreamer')
