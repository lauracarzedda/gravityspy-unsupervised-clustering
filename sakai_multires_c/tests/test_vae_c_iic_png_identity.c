#include "vae_c_iic_png_identity.h"

#include "vae_c_sha256.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif


static int report(const char *name, int passed)
{
    printf("[%s] %s\n", passed ? "PASS" : "FAIL", name);
    return passed ? 0 : 1;
}


static int join_path(
    char *destination,
    size_t capacity,
    const char *root,
    const char *relative
)
{
    size_t root_length = strlen(root);
    size_t relative_length = strlen(relative);

    if (root_length > capacity || relative_length > capacity - root_length ||
        capacity - root_length - relative_length < 2u) {
        return 0;
    }
    memcpy(destination, root, root_length);
    destination[root_length] = '/';
    memcpy(destination + root_length + 1u, relative, relative_length + 1u);
    return 1;
}


static int create_directory(const char *path)
{
#ifdef _WIN32
    return _mkdir(path) == 0 || errno == EEXIST;
#else
    return mkdir(path, 0700) == 0 || errno == EEXIST;
#endif
}


static int write_bytes(const char *path, const char *bytes, size_t length)
{
    FILE *file = fopen(path, "wb");
    int wrote;
    int close_status;

    if (file == NULL) {
        return 0;
    }
    wrote = fwrite(bytes, 1u, length, file) == length;
    close_status = fclose(file);
    return wrote && close_status == 0;
}


static int fixture_file(
    const char *root,
    size_t index,
    const char *bytes
)
{
    char relative[GS_EVENT_MAX_PATH_LEN];
    char path[2048];
    int written = snprintf(
        relative, sizeof(relative), "png_identity_fixture_v1/view_%zu.png",
        index);

    return written > 0 && (size_t)written < sizeof(relative) &&
        join_path(path, sizeof(path), root, relative) &&
        write_bytes(path, bytes, strlen(bytes));
}


static int make_fixture(
    const char *root,
    GSEventRecord records[2]
)
{
    char directory[2048];
    size_t image_index;

    if (!join_path(directory, sizeof(directory), root,
                   "png_identity_fixture_v1") ||
        !create_directory(directory)) {
        return 0;
    }
    memset(records, 0, 2u * sizeof(*records));
    for (image_index = 0u; image_index < 8u; image_index++) {
        char content[32];
        char *relative =
            records[image_index / GS_EVENT_VIEW_COUNT]
                .image_paths[image_index % GS_EVENT_VIEW_COUNT];
        int path_written = snprintf(
            relative, GS_EVENT_MAX_PATH_LEN,
            "png_identity_fixture_v1/view_%zu.png", image_index);
        int content_written = snprintf(
            content, sizeof(content), "PNG-fixture-%zu", image_index);

        if (path_written <= 0 || path_written >= GS_EVENT_MAX_PATH_LEN ||
            content_written <= 0 || (size_t)content_written >= sizeof(content) ||
            !fixture_file(root, image_index, content)) {
            return 0;
        }
    }
    records[0].event_index = 1;
    records[1].event_index = 2;
    (void)strcpy(records[0].label, "label-a");
    (void)strcpy(records[1].label, "label-b");
    return 1;
}


static int check_fixture_identity(const char *root)
{
    GSEventRecord records[2];
    GSEventManifest events;
    char original[65];
    char repeated[65];
    char changed[65];
    char swapped[65];
    char preserved[65];
    char temporary[GS_EVENT_MAX_PATH_LEN];
    size_t count = 0u;
    int failures = 0;

    if (!make_fixture(root, records)) {
        return report("create external-build PNG fixture", 0);
    }
    events.records = records;
    events.count = 2u;
    failures += report(
        "V1 PNG fixture accepted and all eight views counted",
        vae_c_iic_png_content_identity_v1(
            root, &events, original, &count) && count == 8u);
    if (failures != 0) {
        return failures;
    }
    count = 0u;
    failures += report(
        "V1 PNG digest deterministic",
        vae_c_iic_png_content_identity_v1(
            root, &events, repeated, &count) &&
        count == 8u && strcmp(original, repeated) == 0);
    (void)strcpy(records[0].label, "a completely different label");
    failures += report(
        "labels excluded from PNG identity",
        vae_c_iic_png_content_identity_v1(
            root, &events, repeated, &count) &&
        strcmp(original, repeated) == 0);

    if (!fixture_file(root, 3u, "PNG-fixture-X")) {
        return failures + report("mutate one fixture PNG", 0);
    }
    failures += report(
        "single same-length PNG content mutation changes identity",
        vae_c_iic_png_content_identity_v1(
            root, &events, changed, &count) &&
        strcmp(original, changed) != 0);
    if (!fixture_file(root, 3u, "PNG-fixture-3")) {
        return failures + report("restore fixture PNG", 0);
    }

    memcpy(temporary, records[0].image_paths[0], sizeof(temporary));
    memcpy(records[0].image_paths[0], records[0].image_paths[1],
           sizeof(temporary));
    memcpy(records[0].image_paths[1], temporary, sizeof(temporary));
    failures += report(
        "path/view order swap changes identity",
        vae_c_iic_png_content_identity_v1(
            root, &events, swapped, &count) &&
        strcmp(original, swapped) != 0);
    memcpy(records[0].image_paths[1], records[0].image_paths[0],
           sizeof(temporary));
    memcpy(records[0].image_paths[0], temporary, sizeof(temporary));

    (void)strcpy(preserved, "unchanged");
    count = 777u;
    (void)strcpy(records[0].image_paths[0], "../escape.png");
    failures += report(
        "unnormalized relative path rejected without publishing outputs",
        !vae_c_iic_png_content_identity_v1(
            root, &events, preserved, &count) &&
        strcmp(preserved, "unchanged") == 0 && count == 777u);
    return failures;
}


static int check_run_preimage_binding(void)
{
    static const char prefix[] = "format=MINI75_IIC_REAL_RUN_V1\n";
    static const char field[] = "png_content_identity_sha256=";
    char png_a[65] =
        "0000000000000000000000000000000000000000000000000000000000000000";
    char png_b[65];
    char run_a[65];
    char run_b[65];
    char expected[65];
    char full_preimage[sizeof(prefix) - 1u + sizeof(field) - 1u + 64u + 1u];
    int failures = 0;

    memcpy(full_preimage, prefix, sizeof(prefix) - 1u);
    memcpy(full_preimage + sizeof(prefix) - 1u,
           field, sizeof(field) - 1u);
    memcpy(full_preimage + sizeof(prefix) + sizeof(field) - 2u,
           png_a, 64u);
    full_preimage[sizeof(full_preimage) - 1u] = '\n';
    failures += report(
        "run-id preimage binds exact PNG field and final LF",
        vae_c_sha256_digest_hex_lower(
            full_preimage, sizeof(full_preimage), expected) ==
            VAE_C_SHA256_OK &&
        vae_c_iic_png_bind_run_preimage_v1(
            prefix, sizeof(prefix) - 1u, png_a, run_a) &&
        strcmp(expected, run_a) == 0);
    memcpy(png_b, png_a, sizeof(png_b));
    png_b[0] = '1';
    failures += report(
        "run-id digest changes with PNG content identity",
        vae_c_iic_png_bind_run_preimage_v1(
            prefix, sizeof(prefix) - 1u, png_b, run_b) &&
        strcmp(run_a, run_b) != 0);
    png_b[0] = 'G';
    (void)strcpy(run_b, "unchanged");
    failures += report(
        "noncanonical PNG SHA rejected without output",
        !vae_c_iic_png_bind_run_preimage_v1(
            prefix, sizeof(prefix) - 1u, png_b, run_b) &&
        strcmp(run_b, "unchanged") == 0);
    return failures;
}


int main(int argc, char **argv)
{
    int failures;

    if (argc != 2 || argv[1][0] == '\0') {
        fprintf(stderr, "usage: test_vae_c_iic_png_identity BUILD_DIRECTORY\n");
        return 2;
    }
    failures = check_fixture_identity(argv[1]);
    failures += check_run_preimage_binding();
    return failures == 0 ? 0 : 1;
}
