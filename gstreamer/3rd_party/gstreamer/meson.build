plugin_c_args = ['-DHAVE_CONFIG_H']

cdata = configuration_data()
cdata.set_quoted('PACKAGE_VERSION', gst_version)
cdata.set_quoted('PACKAGE', 'gst-itt4rt-plugin')
# Gstreamer license list: https://gitlab.freedesktop.org/gstreamer/gstreamer/blob/1.16.0/gst/gstplugin.c
# BSD or MIT/X11
cdata.set_quoted('GST_LICENSE', 'Proprietary')
cdata.set_quoted('GST_API_VERSION', api_version)
cdata.set_quoted('GST_PACKAGE_NAME', 'GStreamer template Plug-ins')
cdata.set_quoted('GST_PACKAGE_ORIGIN', 'https://gstreamer.freedesktop.org')
configure_file(output : 'config.h', configuration : cdata)

#gstaudio_dep = dependency('gstreamer-audio-1.0',
#    fallback: ['gst-plugins-base', 'audio_dep'])

#gstrtp_dep = dependency('gstreamer-rtp-1.0')


# Viewport Glib metadata dynamic library
viewport_meta_sources = [
  'gst_objects_info_meta.c'
  ]

# viewport meta
# create gst_viewport_meta.dll or libgst_viewport_meta.so
gst_objects_info_meta = library('gst_objects_info_meta',
  viewport_meta_sources,
  c_args: plugin_c_args,
  dependencies : [gst_dep],
  install : true,
  install_dir : join_paths(get_option('libdir'), 'gstreamer-1.0'),
)
