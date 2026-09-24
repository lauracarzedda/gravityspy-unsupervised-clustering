#include "vae_resize_u8.h"

#include <stdio.h>
#include <string.h>


static void gs_resize_bilinear_gray_u8(
    const uint8_t *src,
    int src_w,
    int src_h,
    uint8_t *dst,
    int dst_w,
    int dst_h
)
{
    int x;
    int y;

    for (y = 0; y < dst_h; y++) {
        float gy;
        int y0;
        int y1;
        float dy;

        if (dst_h > 1) {
            gy = ((float)y * (float)(src_h - 1)) /
                 (float)(dst_h - 1);
        } else {
            gy = 0.0f;
        }

        y0 = (int)gy;
        y1 = y0 + 1;

        if (y1 >= src_h) {
            y1 = src_h - 1;
        }

        dy = gy - (float)y0;

        for (x = 0; x < dst_w; x++) {
            float gx;
            int x0;
            int x1;
            float dx;

            float p00;
            float p10;
            float p01;
            float p11;

            float top_value;
            float bottom_value;
            float value;

            if (dst_w > 1) {
                gx = ((float)x * (float)(src_w - 1)) /
                     (float)(dst_w - 1);
            } else {
                gx = 0.0f;
            }

            x0 = (int)gx;
            x1 = x0 + 1;

            if (x1 >= src_w) {
                x1 = src_w - 1;
            }

            dx = gx - (float)x0;

            p00 = (float)src[y0 * src_w + x0];
            p10 = (float)src[y0 * src_w + x1];
            p01 = (float)src[y1 * src_w + x0];
            p11 = (float)src[y1 * src_w + x1];

            top_value =
                p00 * (1.0f - dx) +
                p10 * dx;

            bottom_value =
                p01 * (1.0f - dx) +
                p11 * dx;

            value =
                top_value * (1.0f - dy) +
                bottom_value * dy;

            if (value < 0.0f) {
                value = 0.0f;
            }

            if (value > 255.0f) {
                value = 255.0f;
            }

            dst[y * dst_w + x] =
                (uint8_t)(value + 0.5f);
        }
    }
}


int gs_resize_tensor_u8(
    const uint8_t *src,
    const GSTensorSpec *src_spec,
    uint8_t *dst,
    const GSTensorSpec *dst_spec
)
{
    int c;

    if (src == NULL || dst == NULL) {
        printf("Errore: buffer resize NULL.\n");
        return 1;
    }

    if (!gs_tensor_spec_is_valid(src_spec) ||
        !gs_tensor_spec_is_valid(dst_spec)) {

        printf("Errore: tensor spec non valida nel resize.\n");
        return 1;
    }

    if (src_spec->channels != dst_spec->channels) {
        printf("Errore: numero canali diverso.\n");
        return 1;
    }

    if (src_spec->height == dst_spec->height &&
        src_spec->width == dst_spec->width) {

        memcpy(
            dst,
            src,
            src_spec->values_per_event *
            sizeof(uint8_t)
        );

        return 0;
    }

    for (c = 0; c < src_spec->channels; c++) {
        const uint8_t *src_channel;
        uint8_t *dst_channel;

        src_channel =
            src +
            (size_t)c *
            src_spec->pixels_per_channel;

        dst_channel =
            dst +
            (size_t)c *
            dst_spec->pixels_per_channel;

        gs_resize_bilinear_gray_u8(
            src_channel,
            src_spec->width,
            src_spec->height,
            dst_channel,
            dst_spec->width,
            dst_spec->height
        );
    }

    return 0;
}