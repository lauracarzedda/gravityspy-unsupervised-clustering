#ifndef VAE_C_IIC_PNG_IDENTITY_H
#define VAE_C_IIC_PNG_IDENTITY_H

#include <stdbool.h>
#include <stddef.h>

#include "vae_event_dataset.h"


#define VAE_C_IIC_PNG_CONTENT_IDENTITY_FORMAT_V1 \
    "MINI75_IIC_PNG_CONTENT_IDENTITY_V1"
#define VAE_C_IIC_PNG_CONTENT_IDENTITY_SHA256_CAP 65u

/*
 * V1 hashes the UTF-8 text formed by one line for each manifest record and
 * each of its four views, in manifest order and view order:
 *
 *   manifest_index|view_index|relative_path|byte_count|sha256
 *
 * Indices and byte counts are unsigned decimal; SHA-256 is lowercase hex.
 * Paths are normalized relative paths using '/', with no '.' or '..'
 * components. Lines are separated by LF, with no final LF or BOM. The
 * repository root and all event labels are excluded from the preimage.
 * This function reads but never modifies image files. It publishes both
 * outputs only after every referenced file has been read successfully.
 */
bool vae_c_iic_png_content_identity_v1(
    const char *repository_root,
    const GSEventManifest *events,
    char sha256[VAE_C_IIC_PNG_CONTENT_IDENTITY_SHA256_CAP],
    size_t *png_count
);

/*
 * Bind a validated V1 PNG identity to an existing run-id preimage by hashing
 * prefix followed by "png_content_identity_sha256=<hex>\n". On failure,
 * digest is unchanged. This helper is pure and performs no file I/O.
 */
bool vae_c_iic_png_bind_run_preimage_v1(
    const char *prefix,
    size_t prefix_len,
    const char png_sha256[VAE_C_IIC_PNG_CONTENT_IDENTITY_SHA256_CAP],
    char digest[VAE_C_IIC_PNG_CONTENT_IDENTITY_SHA256_CAP]
);

#endif
