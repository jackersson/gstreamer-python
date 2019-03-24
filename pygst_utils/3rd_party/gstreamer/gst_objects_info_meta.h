/*
    Meta implementation example
    https://github.com/EricssonResearch/openwebrtc-gst-plugins/tree/master/gst-libs/gst/sctp
*/


#ifndef __GST_OBJECTS_INFO_META_H__
#define __GST_OBJECTS_INFO_META_H__

#include <gst/gst.h>


G_BEGIN_DECLS

typedef enum {
    GST_OBJECTS_INFO_META_PARTIAL_RELIABILITY_NONE,
    GST_OBJECTS_INFO_META_PARTIAL_RELIABILITY_TTL,
    GST_OBJECTS_INFO_META_PARTIAL_RELIABILITY_BUF,
    GST_OBJECTS_INFO_META_PARTIAL_RELIABILITY_RTX

} GstObjectsInfoMetaPartiallyReliability;


// Api Type
// 1-st field of GstMetaInfo
#define GST_OBJECTS_INFO_META_API_TYPE (gst_objects_info_meta_api_get_type())
#define GST_OBJECTS_INFO_META_INFO     (gst_objects_info_meta_get_info())

typedef struct _GstObjectsInfoMeta  GstObjectsInfoMeta;
typedef struct _GstObjectInfo       GstObjectInfo     ;
typedef struct _GstObjectsInfoArray GstObjectsInfoArray;

struct _GstObjectInfo {

    // bounding box
    guint32 x, y, width, height;

    gfloat confidence;
    gchar* class_name;
};

struct _GstObjectsInfoArray {
    GstObjectInfo* items;
    guint32 size;
};


struct _GstObjectsInfoMeta {

    // Required as it is base structure for metadata
    // https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer/html/gstreamer-GstMeta.html
    GstMeta meta;

    // Custom fields
    GstObjectsInfoArray objects;
};

GType gst_objects_info_meta_api_get_type(void);

GST_EXPORT
const GstMetaInfo * gst_objects_info_meta_get_info(void);

GST_EXPORT
GstObjectsInfoMeta * gst_buffer_add_objects_info_meta(GstBuffer *buffer, GstObjectsInfoArray*);

GST_EXPORT
GstObjectsInfoArray* gst_buffer_get_objects_info_meta(GstBuffer* b);

G_END_DECLS

#endif /* __GST_SCTP_SEND_META_H__ */