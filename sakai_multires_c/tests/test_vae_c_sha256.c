#include "vae_c_sha256.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }
    printf("[FAIL] %s\n", name);
    return 1;
}


static int hex_is_canonical(const char hex[VAE_C_SHA256_HEX_SIZE])
{
    size_t index;

    if (hex == NULL || strlen(hex) != 64u) {
        return 0;
    }
    for (index = 0; index < 64u; index++) {
        char value = hex[index];

        if (!((value >= '0' && value <= '9') ||
              (value >= 'a' && value <= 'f'))) {
            return 0;
        }
    }
    return 1;
}


static int check_hex(
    const void *data,
    size_t size,
    const char *expected
)
{
    uint8_t digest[VAE_C_SHA256_DIGEST_SIZE];
    char converted[VAE_C_SHA256_HEX_SIZE];
    char direct[VAE_C_SHA256_HEX_SIZE];

    return
        vae_c_sha256_digest(data, size, digest) == VAE_C_SHA256_OK &&
        vae_c_sha256_hex_lower(digest, converted) == VAE_C_SHA256_OK &&
        vae_c_sha256_digest_hex_lower(data, size, direct) ==
            VAE_C_SHA256_OK &&
        strcmp(converted, expected) == 0 &&
        strcmp(direct, expected) == 0 &&
        hex_is_canonical(converted) &&
        hex_is_canonical(direct);
}


static int test_standard_known_answers(void)
{
    static const uint8_t abc[] = {'a', 'b', 'c'};

    return
        check_hex(
            NULL,
            0u,
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855") &&
        check_hex(
            abc,
            sizeof(abc),
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad");
}


static int test_pre_refactor_reference_vectors(void)
{
    static const uint8_t long_text[] =
        "The quick brown fox jumps over the lazy dog. SHA-256 block "
        "crossing reference payload 0123456789";
    static const uint8_t binary[] = {
        0x00u, 0x01u, 0x7fu, 0x80u, 0xffu, 0x00u, 0x42u
    };
    uint8_t bytes[80];
    size_t index;

    for (index = 0; index < sizeof(bytes); index++) {
        bytes[index] = (uint8_t)(index * 37u + 11u);
    }
    return
        check_hex(
            long_text,
            sizeof(long_text) - 1u,
            "86603c9c67535b02af8b320be8c383b9"
            "c03550c0d030907964888127930343c9") &&
        check_hex(
            binary,
            sizeof(binary),
            "8184569834ae09c86f52f84662be2313"
            "9718c4ac9b489954e4ac6f7e183782fa") &&
        check_hex(
            bytes, 55u,
            "2900465fcb533e05a158fd2b3be0e5e3"
            "b03740d83060aa3580e0d98a96bf2384") &&
        check_hex(
            bytes, 56u,
            "31454ff48ef36af2f08fd511bdc37d9d"
            "5855ac23e992e5ff5445cb6b7674a674") &&
        check_hex(
            bytes, 63u,
            "5f6401b96532c36de4e65beec0409b69"
            "b1d181864c8009b7a04f43e5d56350d1") &&
        check_hex(
            bytes, 64u,
            "94eb5de4943613fd048dc93393ab0687"
            "7405faa39c11f53e9386083339833e7e") &&
        check_hex(
            bytes, 65u,
            "fc518669b6eb4b4dd91827ecacef8668"
            "9c725bd5bab888fd3b26dbb196eec954");
}


static int test_determinism_and_alias_safe_conversion(void)
{
    static const uint8_t data[] = {
        0x00u, 0xffu, 0x10u, 0x00u, 0x80u, 0x55u, 0xaau
    };
    union {
        uint8_t digest[VAE_C_SHA256_DIGEST_SIZE];
        char hex[VAE_C_SHA256_HEX_SIZE];
    } alias;
    uint8_t first[VAE_C_SHA256_DIGEST_SIZE];
    uint8_t second[VAE_C_SHA256_DIGEST_SIZE];
    char expected[VAE_C_SHA256_HEX_SIZE];

    return
        vae_c_sha256_digest(data, sizeof(data), first) == VAE_C_SHA256_OK &&
        vae_c_sha256_digest(data, sizeof(data), second) == VAE_C_SHA256_OK &&
        memcmp(first, second, sizeof(first)) == 0 &&
        vae_c_sha256_hex_lower(first, expected) == VAE_C_SHA256_OK &&
        (memcpy(alias.digest, first, sizeof(first)), 1) &&
        vae_c_sha256_hex_lower(alias.digest, alias.hex) == VAE_C_SHA256_OK &&
        strcmp(alias.hex, expected) == 0;
}


static int test_invalid_arguments_are_atomic(void)
{
    uint8_t byte = 0x5au;
    uint8_t digest[VAE_C_SHA256_DIGEST_SIZE];
    uint8_t digest_before[VAE_C_SHA256_DIGEST_SIZE];
    char hex[VAE_C_SHA256_HEX_SIZE];
    char hex_before[VAE_C_SHA256_HEX_SIZE];

    memset(digest, 0xa5, sizeof(digest));
    memcpy(digest_before, digest, sizeof(digest));
    memset(hex, 'Q', sizeof(hex));
    memcpy(hex_before, hex, sizeof(hex));
    if (vae_c_sha256_digest(NULL, 1u, digest) !=
            VAE_C_SHA256_INVALID_ARGUMENT ||
        memcmp(digest, digest_before, sizeof(digest)) != 0 ||
        vae_c_sha256_digest(&byte, 1u, NULL) !=
            VAE_C_SHA256_INVALID_ARGUMENT ||
        vae_c_sha256_hex_lower(NULL, hex) !=
            VAE_C_SHA256_INVALID_ARGUMENT ||
        memcmp(hex, hex_before, sizeof(hex)) != 0 ||
        vae_c_sha256_hex_lower(digest, NULL) !=
            VAE_C_SHA256_INVALID_ARGUMENT ||
        vae_c_sha256_digest_hex_lower(NULL, 1u, hex) !=
            VAE_C_SHA256_INVALID_ARGUMENT ||
        memcmp(hex, hex_before, sizeof(hex)) != 0 ||
        vae_c_sha256_digest_hex_lower(&byte, 1u, NULL) !=
            VAE_C_SHA256_INVALID_ARGUMENT) {

        return 0;
    }
#if SIZE_MAX > UINT64_MAX / UINT64_C(8)
    if (vae_c_sha256_digest(
            &byte,
            (size_t)(UINT64_MAX / UINT64_C(8) + UINT64_C(1)),
            digest) != VAE_C_SHA256_LENGTH_OVERFLOW ||
        memcmp(digest, digest_before, sizeof(digest)) != 0 ||
        vae_c_sha256_digest_hex_lower(
            &byte,
            (size_t)(UINT64_MAX / UINT64_C(8) + UINT64_C(1)),
            hex) != VAE_C_SHA256_LENGTH_OVERFLOW ||
        memcmp(hex, hex_before, sizeof(hex)) != 0) {

        return 0;
    }
#endif
    return
        strcmp(vae_c_sha256_status_name(VAE_C_SHA256_OK), "ok") == 0 &&
        strcmp(
            vae_c_sha256_status_name(VAE_C_SHA256_INVALID_ARGUMENT),
            "invalid_argument") == 0 &&
        strcmp(
            vae_c_sha256_status_name(VAE_C_SHA256_LENGTH_OVERFLOW),
            "length_overflow") == 0 &&
        strcmp(
            vae_c_sha256_status_name((VAECSha256Status)999),
            "unknown") == 0;
}


int main(void)
{
    int failures = 0;

    failures += report_check(
        "standard_known_answers_empty_and_abc",
        test_standard_known_answers()
    );
    failures += report_check(
        "pre_refactor_binary_and_padding_boundary_vectors",
        test_pre_refactor_reference_vectors()
    );
    failures += report_check(
        "determinism_lowercase_and_alias_safe_conversion",
        test_determinism_and_alias_safe_conversion()
    );
    failures += report_check(
        "invalid_arguments_preserve_outputs",
        test_invalid_arguments_are_atomic()
    );
    if (failures != 0) {
        fprintf(stderr, "%d VAE C SHA-256 TEST(S) FAILED\n", failures);
        return 1;
    }
    printf("SHA256_DIGEST_SIZE=32\n");
    printf("SHA256_HEX_LOWER_SIZE=64\n");
    printf("SHA256_PRE_REFACTOR_REFERENCE_EQUIVALENT=true\n");
    printf("ALL VAE C SHA-256 TESTS PASSED\n");
    return 0;
}
