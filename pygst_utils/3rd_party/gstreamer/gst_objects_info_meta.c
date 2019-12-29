#include "gst_objects_info_meta.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

static gboolean gst_objects_info_meta_init(GstMeta *meta, gpointer params, GstBuffer *buffer);
static gboolean gst_objects_info_meta_transform(GstBuffer *transbuf, GstMeta *meta, GstBuffer *buffer,
                                                GQuark type, gpointer data);

GstObjectsInfoArray* empty(){
    static GstObjectsInfoMeta meta;
    meta.objects.size = 0;
    return  &meta.objects;
}

// Register metadata type and returns Gtype
// https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer/html/gstreamer-GstMeta.html#gst-meta-api-type-register
GType gst_objects_info_meta_api_get_type(void)
{
    static const gchar *tags[] = {NULL};
    static volatile GType type;
    if (g_once_init_enter (&type)) {
        GType _type = gst_meta_api_type_register("GstObjectsInfoMetaAPI", tags);
        g_once_init_leave(&type, _type);
    }
    return type;
}

// GstMetaInfo provides info for specific metadata implementation
// https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer/html/gstreamer-GstMeta.html#GstMetaInfo
const GstMetaInfo *gst_objects_info_meta_get_info(void)
{
    static const GstMetaInfo *gst_objects_info_meta_info = NULL;

    if (g_once_init_enter (&gst_objects_info_meta_info)) {
        // Explanation of fields
        // https://gstreamer.freedesktop.org/documentation/design/meta.html#gstmeta1
        const GstMetaInfo *meta = gst_meta_register (GST_OBJECTS_INFO_META_API_TYPE,
            "GstObjectsInfoMeta",
            sizeof (GstObjectsInfoMeta),
            gst_objects_info_meta_init,
            (GstMetaFreeFunction) NULL,
            gst_objects_info_meta_transform);
        g_once_init_leave (&gst_objects_info_meta_info, meta);
    }
    return gst_objects_info_meta_info;
}

// Meta init function
// 4-th field in GstMetaInfo
static gboolean gst_objects_info_meta_init(GstMeta *meta, gpointer params, GstBuffer *buffer)
{
    GstObjectsInfoMeta *gst_objects_info_meta = (GstObjectsInfoMeta *)meta;
    gst_objects_info_meta->objects.size = 0;
    return TRUE;
}

// Meta transform function
// 5-th field in GstMetaInfo
// https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer/html/gstreamer-GstMeta.html#GstMetaTransformFunction
static gboolean gst_objects_info_meta_transform(GstBuffer *transbuf, GstMeta *meta, GstBuffer *buffer,
                                                GQuark type, gpointer data)
{
    GstObjectsInfoMeta *gst_objects_info_meta = (GstObjectsInfoMeta *)meta;
    gst_buffer_add_objects_info_meta(transbuf, &(gst_objects_info_meta->objects));
    return TRUE;
}


// Only for Python : return GstObjectsInfoArray instead of GsObjectsInfoMeta
// // To avoid GstMeta (C) map to Gst.Meta (Python)
GstObjectsInfoArray* gst_buffer_get_objects_info_meta(GstBuffer* b)
{
    GstObjectsInfoMeta* meta = (GstObjectsInfoMeta*)gst_buffer_get_meta((b), GST_OBJECTS_INFO_META_API_TYPE);

    if (meta == NULL)
        return empty();
    else
        return &meta->objects;
}


GstObjectsInfoMeta * gst_buffer_add_objects_info_meta(GstBuffer *buffer, GstObjectsInfoArray* objects)
{
    GstObjectsInfoMeta *gst_objects_info_meta = NULL;

    g_return_val_if_fail(GST_IS_BUFFER(buffer), NULL);
    g_return_val_if_fail(gst_buffer_is_writable(buffer), NULL);

    gst_objects_info_meta = (GstObjectsInfoMeta *) gst_buffer_add_meta (buffer, GST_OBJECTS_INFO_META_INFO, NULL);

    guint32 size = objects->size;
    if (size > 0){
        gst_objects_info_meta->objects.size = size;
        guint32 total_size = sizeof(GstObjectInfo)*size;
        gst_objects_info_meta->objects.items = malloc(total_size);
        memcpy ( gst_objects_info_meta->objects.items, objects->items, total_size );
    }
    return gst_objects_info_meta;
}


// Removes metadata (GstBufferInfo) from buffer
gboolean gst_buffer_remove_objects_info_meta(GstBuffer *buffer)
{
    g_return_val_if_fail(GST_IS_BUFFER(buffer), NULL);

    GstObjectsInfoMeta* meta = (GstObjectsInfoMeta*)gst_buffer_get_meta((buffer), GST_OBJECTS_INFO_META_API_TYPE);

    if (meta == NULL)
        return TRUE;

    if ( !gst_buffer_is_writable(buffer))
        return FALSE;

    // https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer/html/GstBuffer.html#gst-buffer-remove-meta
    return gst_buffer_remove_meta(buffer, &meta->meta);
}
