#ifndef VAE_C_SHA256_H
#define VAE_C_SHA256_H

#include <stddef.h>
#include <stdint.h>


#define VAE_C_SHA256_DIGEST_SIZE 32u
#define VAE_C_SHA256_HEX_SIZE 65u


typedef enum {
    VAE_C_SHA256_OK = 0,
    VAE_C_SHA256_INVALID_ARGUMENT,
    VAE_C_SHA256_LENGTH_OVERFLOW
} VAECSha256Status;


/*
 * Compute SHA-256 for an arbitrary byte buffer.  For size == 0, data may be
 * NULL.  On failure, digest is left unchanged.
 */
VAECSha256Status vae_c_sha256_digest(
    const void *data,
    size_t size,
    uint8_t digest[VAE_C_SHA256_DIGEST_SIZE]
);

/*
 * Convert a binary SHA-256 digest to exactly 64 lowercase hexadecimal
 * characters followed by a null terminator.  On failure, hex is unchanged.
 */
VAECSha256Status vae_c_sha256_hex_lower(
    const uint8_t digest[VAE_C_SHA256_DIGEST_SIZE],
    char hex[VAE_C_SHA256_HEX_SIZE]
);

/*
 * Compute SHA-256 and publish its canonical lowercase hexadecimal form.
 * For size == 0, data may be NULL.  On failure, hex is left unchanged.
 */
VAECSha256Status vae_c_sha256_digest_hex_lower(
    const void *data,
    size_t size,
    char hex[VAE_C_SHA256_HEX_SIZE]
);

const char *vae_c_sha256_status_name(VAECSha256Status status);

#endif
