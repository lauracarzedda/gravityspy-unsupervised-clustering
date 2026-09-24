#include "vae_c_sha256.h"

#include <string.h>


typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    uint8_t block[64];
    size_t block_size;
} VAECSha256Context;


static uint32_t sha256_rotate_right(uint32_t value, unsigned count)
{
    return (value >> count) | (value << (32u - count));
}


static void sha256_transform(
    VAECSha256Context *context,
    const uint8_t block[64]
)
{
    static const uint32_t constants[64] = {
        UINT32_C(0x428a2f98), UINT32_C(0x71374491),
        UINT32_C(0xb5c0fbcf), UINT32_C(0xe9b5dba5),
        UINT32_C(0x3956c25b), UINT32_C(0x59f111f1),
        UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5),
        UINT32_C(0xd807aa98), UINT32_C(0x12835b01),
        UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
        UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe),
        UINT32_C(0x9bdc06a7), UINT32_C(0xc19bf174),
        UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786),
        UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc),
        UINT32_C(0x2de92c6f), UINT32_C(0x4a7484aa),
        UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
        UINT32_C(0x983e5152), UINT32_C(0xa831c66d),
        UINT32_C(0xb00327c8), UINT32_C(0xbf597fc7),
        UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147),
        UINT32_C(0x06ca6351), UINT32_C(0x14292967),
        UINT32_C(0x27b70a85), UINT32_C(0x2e1b2138),
        UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
        UINT32_C(0x650a7354), UINT32_C(0x766a0abb),
        UINT32_C(0x81c2c92e), UINT32_C(0x92722c85),
        UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b),
        UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3),
        UINT32_C(0xd192e819), UINT32_C(0xd6990624),
        UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
        UINT32_C(0x19a4c116), UINT32_C(0x1e376c08),
        UINT32_C(0x2748774c), UINT32_C(0x34b0bcb5),
        UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a),
        UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3),
        UINT32_C(0x748f82ee), UINT32_C(0x78a5636f),
        UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
        UINT32_C(0x90befffa), UINT32_C(0xa4506ceb),
        UINT32_C(0xbef9a3f7), UINT32_C(0xc67178f2)
    };
    uint32_t words[64];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
    uint32_t f;
    uint32_t g;
    uint32_t h;
    size_t index;

    for (index = 0; index < 16u; index++) {
        size_t offset = index * 4u;

        words[index] =
            ((uint32_t)block[offset] << 24) |
            ((uint32_t)block[offset + 1u] << 16) |
            ((uint32_t)block[offset + 2u] << 8) |
            (uint32_t)block[offset + 3u];
    }
    for (index = 16u; index < 64u; index++) {
        uint32_t s0 =
            sha256_rotate_right(words[index - 15u], 7u) ^
            sha256_rotate_right(words[index - 15u], 18u) ^
            (words[index - 15u] >> 3);
        uint32_t s1 =
            sha256_rotate_right(words[index - 2u], 17u) ^
            sha256_rotate_right(words[index - 2u], 19u) ^
            (words[index - 2u] >> 10);

        words[index] = words[index - 16u] + s0 +
            words[index - 7u] + s1;
    }
    a = context->state[0];
    b = context->state[1];
    c = context->state[2];
    d = context->state[3];
    e = context->state[4];
    f = context->state[5];
    g = context->state[6];
    h = context->state[7];
    for (index = 0; index < 64u; index++) {
        uint32_t sum1 =
            sha256_rotate_right(e, 6u) ^
            sha256_rotate_right(e, 11u) ^
            sha256_rotate_right(e, 25u);
        uint32_t choice = (e & f) ^ ((~e) & g);
        uint32_t temporary1 =
            h + sum1 + choice + constants[index] + words[index];
        uint32_t sum0 =
            sha256_rotate_right(a, 2u) ^
            sha256_rotate_right(a, 13u) ^
            sha256_rotate_right(a, 22u);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temporary2 = sum0 + majority;

        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }
    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}


static void sha256_init(VAECSha256Context *context)
{
    memset(context, 0, sizeof(*context));
    context->state[0] = UINT32_C(0x6a09e667);
    context->state[1] = UINT32_C(0xbb67ae85);
    context->state[2] = UINT32_C(0x3c6ef372);
    context->state[3] = UINT32_C(0xa54ff53a);
    context->state[4] = UINT32_C(0x510e527f);
    context->state[5] = UINT32_C(0x9b05688c);
    context->state[6] = UINT32_C(0x1f83d9ab);
    context->state[7] = UINT32_C(0x5be0cd19);
}


static void sha256_update(
    VAECSha256Context *context,
    const uint8_t *data,
    size_t size
)
{
    size_t index;

    context->bit_count += (uint64_t)size * UINT64_C(8);
    for (index = 0; index < size; index++) {
        context->block[context->block_size++] = data[index];
        if (context->block_size == sizeof(context->block)) {
            sha256_transform(context, context->block);
            context->block_size = 0;
        }
    }
}


static void sha256_final(
    VAECSha256Context *context,
    uint8_t digest[VAE_C_SHA256_DIGEST_SIZE]
)
{
    uint64_t bit_count = context->bit_count;
    size_t index;

    context->block[context->block_size++] = 0x80u;
    if (context->block_size > 56u) {
        while (context->block_size < 64u) {
            context->block[context->block_size++] = 0u;
        }
        sha256_transform(context, context->block);
        context->block_size = 0;
    }
    while (context->block_size < 56u) {
        context->block[context->block_size++] = 0u;
    }
    for (index = 0; index < 8u; index++) {
        context->block[63u - index] =
            (uint8_t)(bit_count >> (8u * index));
    }
    sha256_transform(context, context->block);
    for (index = 0; index < 8u; index++) {
        digest[index * 4u] = (uint8_t)(context->state[index] >> 24);
        digest[index * 4u + 1u] =
            (uint8_t)(context->state[index] >> 16);
        digest[index * 4u + 2u] =
            (uint8_t)(context->state[index] >> 8);
        digest[index * 4u + 3u] = (uint8_t)context->state[index];
    }
}


VAECSha256Status vae_c_sha256_digest(
    const void *data,
    size_t size,
    uint8_t digest[VAE_C_SHA256_DIGEST_SIZE]
)
{
    VAECSha256Context context;
    uint8_t candidate[VAE_C_SHA256_DIGEST_SIZE];

    if (digest == NULL || (size != 0u && data == NULL)) {
        return VAE_C_SHA256_INVALID_ARGUMENT;
    }
    if (size > UINT64_MAX / UINT64_C(8)) {
        return VAE_C_SHA256_LENGTH_OVERFLOW;
    }
    sha256_init(&context);
    sha256_update(&context, (const uint8_t *)data, size);
    sha256_final(&context, candidate);
    memcpy(digest, candidate, sizeof(candidate));
    return VAE_C_SHA256_OK;
}


VAECSha256Status vae_c_sha256_hex_lower(
    const uint8_t digest[VAE_C_SHA256_DIGEST_SIZE],
    char hex[VAE_C_SHA256_HEX_SIZE]
)
{
    static const char digits[] = "0123456789abcdef";
    char candidate[VAE_C_SHA256_HEX_SIZE];
    size_t index;

    if (digest == NULL || hex == NULL) {
        return VAE_C_SHA256_INVALID_ARGUMENT;
    }
    for (index = 0; index < VAE_C_SHA256_DIGEST_SIZE; index++) {
        candidate[index * 2u] = digits[digest[index] >> 4];
        candidate[index * 2u + 1u] = digits[digest[index] & 0x0fu];
    }
    candidate[VAE_C_SHA256_HEX_SIZE - 1u] = '\0';
    memcpy(hex, candidate, sizeof(candidate));
    return VAE_C_SHA256_OK;
}


VAECSha256Status vae_c_sha256_digest_hex_lower(
    const void *data,
    size_t size,
    char hex[VAE_C_SHA256_HEX_SIZE]
)
{
    uint8_t digest[VAE_C_SHA256_DIGEST_SIZE];
    char candidate[VAE_C_SHA256_HEX_SIZE];
    VAECSha256Status status;

    if (hex == NULL) {
        return VAE_C_SHA256_INVALID_ARGUMENT;
    }
    status = vae_c_sha256_digest(data, size, digest);
    if (status != VAE_C_SHA256_OK) {
        return status;
    }
    status = vae_c_sha256_hex_lower(digest, candidate);
    if (status == VAE_C_SHA256_OK) {
        memcpy(hex, candidate, sizeof(candidate));
    }
    return status;
}


const char *vae_c_sha256_status_name(VAECSha256Status status)
{
    switch (status) {
        case VAE_C_SHA256_OK:
            return "ok";
        case VAE_C_SHA256_INVALID_ARGUMENT:
            return "invalid_argument";
        case VAE_C_SHA256_LENGTH_OVERFLOW:
            return "length_overflow";
        default:
            return "unknown";
    }
}
