#include "vae_c_iic_png_identity.h"

#include "vae_c_sha256.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef struct {
    unsigned char *bytes;
    size_t length;
    size_t capacity;
} PNGCanonicalText;


static bool append_bytes(
    PNGCanonicalText *text,
    const void *bytes,
    size_t length
)
{
    size_t needed;

    if (length == 0u) {
        return true;
    }
    if (text == NULL || bytes == NULL || length > SIZE_MAX - text->length) {
        return false;
    }
    needed = text->length + length;
    if (needed > text->capacity) {
        size_t capacity = text->capacity == 0u ? 4096u : text->capacity;
        unsigned char *grown;

        while (capacity < needed) {
            if (capacity > SIZE_MAX / 2u) {
                capacity = needed;
                break;
            }
            capacity *= 2u;
        }
        grown = (unsigned char *)realloc(text->bytes, capacity);
        if (grown == NULL) {
            return false;
        }
        text->bytes = grown;
        text->capacity = capacity;
    }
    memcpy(text->bytes + text->length, bytes, length);
    text->length = needed;
    return true;
}


static bool append_number(PNGCanonicalText *text, size_t number)
{
    char digits[3u * sizeof(size_t) + 1u];
    int written = snprintf(digits, sizeof(digits), "%zu", number);

    return written > 0 && (size_t)written < sizeof(digits) &&
        append_bytes(text, digits, (size_t)written);
}


static bool is_utf8_relative_path(const char *path, size_t length)
{
    const unsigned char *bytes = (const unsigned char *)path;
    size_t index = 0u;
    size_t component_start = 0u;

    if (length == 0u) {
        return false;
    }
    while (index < length) {
        unsigned char first = bytes[index];
        size_t width = 1u;

        if (first < 0x80u) {
            if (first < 0x20u || first == 0x7fu || first == '\\' ||
                first == ':' || first == '|' || first == '"' ||
                first == '<' || first == '>' || first == '?' ||
                first == '*') {
                return false;
            }
            if (first == '/') {
                size_t component_length = index - component_start;

                if (component_length == 0u ||
                    (component_length == 1u &&
                     bytes[component_start] == '.') ||
                    (component_length == 2u &&
                     bytes[component_start] == '.' &&
                     bytes[component_start + 1u] == '.')) {
                    return false;
                }
                component_start = index + 1u;
            }
        } else {
            unsigned char second;

            if (first >= 0xc2u && first <= 0xdfu) {
                width = 2u;
            } else if (first >= 0xe0u && first <= 0xefu) {
                width = 3u;
            } else if (first >= 0xf0u && first <= 0xf4u) {
                width = 4u;
            } else {
                return false;
            }
            if (width > length - index) {
                return false;
            }
            second = bytes[index + 1u];
            if (second < 0x80u || second > 0xbfu ||
                (first == 0xe0u && second < 0xa0u) ||
                (first == 0xedu && second > 0x9fu) ||
                (first == 0xf0u && second < 0x90u) ||
                (first == 0xf4u && second > 0x8fu)) {
                return false;
            }
            if (width >= 3u &&
                (bytes[index + 2u] < 0x80u ||
                 bytes[index + 2u] > 0xbfu)) {
                return false;
            }
            if (width == 4u &&
                (bytes[index + 3u] < 0x80u ||
                 bytes[index + 3u] > 0xbfu)) {
                return false;
            }
        }
        index += width;
    }
    {
        size_t component_length = length - component_start;

        return component_length != 0u &&
            !(component_length == 1u && bytes[component_start] == '.') &&
            !(component_length == 2u && bytes[component_start] == '.' &&
              bytes[component_start + 1u] == '.');
    }
}


static bool hash_png_file(
    const char *path,
    char sha256[VAE_C_SHA256_HEX_SIZE],
    size_t *byte_count
)
{
    FILE *file = fopen(path, "rb");
    unsigned char *bytes;
    long length;
    size_t size;
    bool read_ok;
    int next_byte;
    int close_status;

    if (file == NULL) {
        return false;
    }
    if (fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        (void)fclose(file);
        return false;
    }
    size = (size_t)length;
    if ((long)size != length) {
        (void)fclose(file);
        return false;
    }
    bytes = (unsigned char *)malloc(size == 0u ? 1u : size);
    if (bytes == NULL) {
        (void)fclose(file);
        return false;
    }
    read_ok = size == 0u || fread(bytes, 1u, size, file) == size;
    next_byte = read_ok ? fgetc(file) : 0;
    read_ok = read_ok && next_byte == EOF && !ferror(file);
    close_status = fclose(file);
    if (!read_ok || close_status != 0 ||
        vae_c_sha256_digest_hex_lower(bytes, size, sha256) !=
            VAE_C_SHA256_OK) {
        free(bytes);
        return false;
    }
    free(bytes);
    *byte_count = size;
    return true;
}


bool vae_c_iic_png_content_identity_v1(
    const char *repository_root,
    const GSEventManifest *events,
    char sha256[VAE_C_IIC_PNG_CONTENT_IDENTITY_SHA256_CAP],
    size_t *png_count
)
{
    PNGCanonicalText text = {NULL, 0u, 0u};
    size_t root_length;
    size_t image_count;
    char candidate[VAE_C_SHA256_HEX_SIZE];
    size_t manifest_index;
    bool okay = false;

    if (repository_root == NULL || repository_root[0] == '\0' ||
        events == NULL || events->records == NULL || events->count == 0u ||
        events->count > SIZE_MAX / GS_EVENT_VIEW_COUNT ||
        sha256 == NULL || png_count == NULL) {
        return false;
    }
    root_length = strlen(repository_root);
    image_count = events->count * GS_EVENT_VIEW_COUNT;
    for (manifest_index = 0u; manifest_index < events->count;
         manifest_index++) {
        size_t view_index;

        for (view_index = 0u; view_index < GS_EVENT_VIEW_COUNT;
             view_index++) {
            const char *relative =
                events->records[manifest_index].image_paths[view_index];
            const char *end = memchr(relative, '\0', GS_EVENT_MAX_PATH_LEN);
            size_t relative_length;
            char *absolute;
            char file_sha256[VAE_C_SHA256_HEX_SIZE];
            size_t byte_count;
            bool row_ok;

            if (end == NULL) {
                goto done;
            }
            relative_length = (size_t)(end - relative);
            if (!is_utf8_relative_path(relative, relative_length) ||
                root_length > SIZE_MAX - relative_length - 2u) {
                goto done;
            }
            absolute = (char *)malloc(root_length + relative_length + 2u);
            if (absolute == NULL) {
                goto done;
            }
            memcpy(absolute, repository_root, root_length);
            absolute[root_length] = '/';
            memcpy(absolute + root_length + 1u, relative, relative_length);
            absolute[root_length + 1u + relative_length] = '\0';
            row_ok = hash_png_file(absolute, file_sha256, &byte_count);
            free(absolute);
            if (!row_ok) {
                goto done;
            }
            if ((manifest_index != 0u || view_index != 0u) &&
                !append_bytes(&text, "\n", 1u)) {
                goto done;
            }
            if (!append_number(&text, manifest_index) ||
                !append_bytes(&text, "|", 1u) ||
                !append_number(&text, view_index) ||
                !append_bytes(&text, "|", 1u) ||
                !append_bytes(&text, relative, relative_length) ||
                !append_bytes(&text, "|", 1u) ||
                !append_number(&text, byte_count) ||
                !append_bytes(&text, "|", 1u) ||
                !append_bytes(&text, file_sha256, 64u)) {
                goto done;
            }
        }
    }
    if (vae_c_sha256_digest_hex_lower(text.bytes, text.length, candidate) !=
        VAE_C_SHA256_OK) {
        goto done;
    }
    memcpy(sha256, candidate, sizeof(candidate));
    *png_count = image_count;
    okay = true;

done:
    free(text.bytes);
    return okay;
}


bool vae_c_iic_png_bind_run_preimage_v1(
    const char *prefix,
    size_t prefix_len,
    const char png_sha256[VAE_C_IIC_PNG_CONTENT_IDENTITY_SHA256_CAP],
    char digest[VAE_C_IIC_PNG_CONTENT_IDENTITY_SHA256_CAP]
)
{
    static const char field[] = "png_content_identity_sha256=";
    enum { MAX_PREFIX_BYTES = 1024 * 1024 };
    const size_t suffix_length = sizeof(field) - 1u + 64u + 1u;
    unsigned char *preimage;
    char candidate[VAE_C_SHA256_HEX_SIZE];
    size_t index;
    bool okay;

    if ((prefix_len != 0u && prefix == NULL) ||
        prefix_len > MAX_PREFIX_BYTES || png_sha256 == NULL ||
        digest == NULL || prefix_len > SIZE_MAX - suffix_length ||
        png_sha256[64] != '\0') {
        return false;
    }
    for (index = 0u; index < 64u; index++) {
        char value = png_sha256[index];

        if (!((value >= '0' && value <= '9') ||
              (value >= 'a' && value <= 'f'))) {
            return false;
        }
    }
    preimage = (unsigned char *)malloc(prefix_len + suffix_length);
    if (preimage == NULL) {
        return false;
    }
    if (prefix_len != 0u) {
        memcpy(preimage, prefix, prefix_len);
    }
    memcpy(preimage + prefix_len, field, sizeof(field) - 1u);
    memcpy(preimage + prefix_len + sizeof(field) - 1u,
           png_sha256, 64u);
    preimage[prefix_len + suffix_length - 1u] = '\n';
    okay = vae_c_sha256_digest_hex_lower(
        preimage, prefix_len + suffix_length, candidate) == VAE_C_SHA256_OK;
    free(preimage);
    if (okay) {
        memcpy(digest, candidate, sizeof(candidate));
    }
    return okay;
}
