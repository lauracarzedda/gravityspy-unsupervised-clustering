#include "vae_c_mu_contract.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

#define PATH_CAP 2048u
#define BASE_COMMIT "3619ba1bd6705cd31fe3fc0b8edf16d18b0b3a8d"
#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "affinity producer test failed line %d: %s\n", \
            __LINE__, #expr); exit(1); \
} } while (0)

static int join(char *out, size_t cap, const char *base, const char *name)
{
    int n = snprintf(out, cap, "%s/%s", base, name);
    return n >= 0 && (size_t)n < cap;
}

static int make_dir(const char *path)
{
#ifdef _WIN32
    return _mkdir(path) == 0;
#else
    return mkdir(path, 0777) == 0;
#endif
}

static int remove_dir(const char *path)
{
#ifdef _WIN32
    return _rmdir(path) == 0;
#else
    return rmdir(path) == 0;
#endif
}

static int present(const char *path)
{
    struct stat state;
    return stat(path, &state) == 0;
}

static int file_sha(const char *path, char sha[65])
{
    return vae_c_mu_file_sha256(path, sha) == VAE_C_MU_OK;
}

static int make_fixture(const char *directory, int extreme)
{
    static const float h[5u * 4u] = {
        0.0f, 0.0f, 0.25f, -0.5f,
        3.0f, 4.0f, 0.25f, -0.5f,
        0.0f, 0.0f, 0.25f, -0.5f,
        1.0f, 2.0f, -0.75f, 0.125f,
        -1.0f, 0.5f, 0.75f, 0.625f
    };
    const float bad[2] = {0.0f, 1000.0f};
    char path[PATH_CAP];
    FILE *file;
    size_t n = extreme ? 2u : 5u;
    CHECK(make_dir(directory));
    CHECK(join(path, sizeof(path), directory, "h_f32.bin"));
    file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(extreme ? bad : h, sizeof(float),
                 extreme ? 2u : 20u, file) == (extreme ? 2u : 20u));
    CHECK(fclose(file) == 0);
    CHECK(join(path, sizeof(path), directory, "rows.csv"));
    file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fputs("manifest_index,event_index,event_id,source_split,split_pos\n",
                file) >= 0);
    for (size_t i = 0u; i < n; i++) {
        CHECK(fprintf(file, "%zu,%zu,F%zu,train,%zu\n",
                      i, i + 1u, i, i) >= 0);
    }
    CHECK(fclose(file) == 0);
    CHECK(join(path, sizeof(path), directory, "h_spec.txt"));
    file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fprintf(file, "format=TEST_H_V1\nrows=%zu\ncols=%zu\n",
                  n, extreme ? (size_t)1u : (size_t)4u) >= 0);
    CHECK(fclose(file) == 0);
    return 1;
}

static int run_producer(const char *executable, const char *root,
                        const char *fixture, const char *output,
                        const char *binary_sha, size_t rows,
                        size_t cols, size_t block_rows, int validate_only)
{
    char command[8192];
    int n = snprintf(command, sizeof(command),
        "\"%s\" --repository-root \"%s\" --output-dir \"%s\" "
        "--producer-commit %s --producer-binary-sha256 %s "
        "--fixture-dir \"%s\" --fixture-rows %zu --fixture-cols %zu "
        "--block-rows %zu%s",
        executable, root, output, BASE_COMMIT, binary_sha,
        fixture, rows, cols, block_rows,
        validate_only ? " --validate-only" : "");
    if (n < 0 || (size_t)n >= sizeof(command)) return -1;
#ifdef _WIN32
    {
        STARTUPINFOA start;
        PROCESS_INFORMATION process;
        DWORD code = 1u;
        memset(&start, 0, sizeof(start));
        memset(&process, 0, sizeof(process));
        start.cb = sizeof(start);
        if (!CreateProcessA(executable, command, NULL, NULL, FALSE, 0,
                            NULL, NULL, &start, &process)) return -1;
        (void)WaitForSingleObject(process.hProcess, INFINITE);
        (void)GetExitCodeProcess(process.hProcess, &code);
        (void)CloseHandle(process.hThread);
        (void)CloseHandle(process.hProcess);
        return (int)code;
    }
#else
    return system(command);
#endif
}

static int read_payload(const char *directory, double a[25])
{
    char path[PATH_CAP];
    FILE *file;
    CHECK(join(path, sizeof(path), directory, "affinity_f64.bin"));
    file = fopen(path, "rb");
    CHECK(file != NULL);
    CHECK(fseek(file, 0L, SEEK_END) == 0);
    CHECK(ftell(file) == 200L);
    CHECK(fseek(file, 0L, SEEK_SET) == 0);
    CHECK(fread(a, sizeof(double), 25u, file) == 25u);
    CHECK(fclose(file) == 0);
    for (size_t i = 0u; i < 5u; i++) {
        CHECK(a[i * 5u + i] == 1.0);
        for (size_t j = 0u; j < 5u; j++) {
            CHECK(isfinite(a[i * 5u + j]));
            CHECK(a[i * 5u + j] > 0.0 && a[i * 5u + j] <= 1.0);
            CHECK(memcmp(&a[i * 5u + j], &a[j * 5u + i], 8u) == 0);
        }
    }
    CHECK(fabs(a[1] - exp(-5.0)) < 1.0e-15);
    CHECK(a[2] == 1.0);
    {
        const unsigned char *p = (const unsigned char *)a;
        const unsigned char one_le[8] = {0u, 0u, 0u, 0u, 0u, 0u, 240u, 63u};
        CHECK(memcmp(p, one_le, 8u) == 0);
    }
    return 1;
}

static void remove_artifact(const char *directory)
{
    static const char *names[] = {"affinity_f64.bin", "rows.csv",
                                  "affinity_spec.txt", "sha256.txt"};
    char path[PATH_CAP];
    for (size_t i = 0u; i < 4u; i++) {
        if (join(path, sizeof(path), directory, names[i])) (void)remove(path);
    }
    (void)remove_dir(directory);
}

static void remove_fixture(const char *directory)
{
    static const char *names[] = {"h_f32.bin", "rows.csv", "h_spec.txt"};
    char path[PATH_CAP];
    for (size_t i = 0u; i < 3u; i++) {
        if (join(path, sizeof(path), directory, names[i])) (void)remove(path);
    }
    (void)remove_dir(directory);
}

int main(int argc, char **argv)
{
    char base[PATH_CAP], fixture[PATH_CAP], bad_fixture[PATH_CAP];
    char first[PATH_CAP], second[PATH_CAP], pending[PATH_CAP];
    char failed[PATH_CAP], path[PATH_CAP], binary_sha[65];
    char source_before[3][65], source_after[65], payload_sha[2][65];
    double a[25], b[25];
    static const char *source_names[] = {"h_f32.bin", "rows.csv", "h_spec.txt"};
    FILE *file;
    unsigned long pid;
    if (argc != 4) return 2;
#ifdef _WIN32
    pid = (unsigned long)GetCurrentProcessId();
#else
    pid = (unsigned long)getpid();
#endif
    {
        int n = snprintf(base, sizeof(base), "%s/affinity_fixture_%lu",
                         argv[2], pid);
        CHECK(n > 0 && (size_t)n < sizeof(base));
    }
    CHECK(!present(base));
    CHECK(make_dir(base));
    CHECK(join(fixture, sizeof(fixture), base, "source"));
    CHECK(join(bad_fixture, sizeof(bad_fixture), base, "bad_source"));
    CHECK(join(first, sizeof(first), base, "first"));
    CHECK(join(second, sizeof(second), base, "second"));
    CHECK(join(pending, sizeof(pending), base, "pending"));
    CHECK(join(failed, sizeof(failed), base, "failed"));
    CHECK(make_fixture(fixture, 0));
    CHECK(make_fixture(bad_fixture, 1));
    CHECK(file_sha(argv[1], binary_sha));
    for (size_t i = 0u; i < 3u; i++) {
        CHECK(join(path, sizeof(path), fixture, source_names[i]));
        CHECK(file_sha(path, source_before[i]));
    }
    CHECK(run_producer(argv[1], argv[3], fixture, first,
                       binary_sha, 5u, 4u, 2u, 1) == 0);
    CHECK(!present(first));
    CHECK(run_producer(argv[1], argv[3], fixture, first,
                       binary_sha, 5u, 4u, 2u, 0) == 0);
    CHECK(run_producer(argv[1], argv[3], fixture, second,
                       binary_sha, 5u, 4u, 3u, 0) == 0);
    CHECK(read_payload(first, a));
    CHECK(read_payload(second, b));
    CHECK(memcmp(a, b, sizeof(a)) == 0);
    for (size_t i = 0u; i < 2u; i++) {
        const char *directory = i == 0u ? first : second;
        CHECK(join(path, sizeof(path), directory, "affinity_f64.bin"));
        CHECK(file_sha(path, payload_sha[i]));
        CHECK(join(path, sizeof(path), directory, "rows.csv"));
        CHECK(file_sha(path, source_after));
        CHECK(strcmp(source_after, source_before[1]) == 0);
        CHECK(join(path, sizeof(path), directory, "affinity_spec.txt"));
        CHECK(present(path));
        CHECK(join(path, sizeof(path), directory, "sha256.txt"));
        CHECK(present(path));
    }
    CHECK(strcmp(payload_sha[0], payload_sha[1]) == 0);
    CHECK(run_producer(argv[1], argv[3], fixture, first,
                       binary_sha, 5u, 4u, 2u, 0) != 0);
    CHECK(snprintf(path, sizeof(path), "%s.partial", pending) > 0);
    CHECK(make_dir(path));
    {
        char sentinel[PATH_CAP];
        CHECK(join(sentinel, sizeof(sentinel), path, "sentinel.txt"));
        file = fopen(sentinel, "wb");
        CHECK(file != NULL);
        CHECK(fputs("do not replace\n", file) >= 0);
        CHECK(fclose(file) == 0);
        CHECK(run_producer(argv[1], argv[3], fixture, pending,
                           binary_sha, 5u, 4u, 2u, 0) != 0);
        CHECK(present(sentinel));
        CHECK(remove(sentinel) == 0);
    }
    CHECK(remove_dir(path));
    CHECK(run_producer(argv[1], argv[3], bad_fixture, failed,
                       binary_sha, 2u, 1u, 1u, 0) != 0);
    CHECK(!present(failed));
    CHECK(snprintf(path, sizeof(path), "%s.partial", failed) > 0);
    CHECK(!present(path));
    CHECK(run_producer(argv[1], argv[3], fixture, failed,
                       binary_sha, 6u, 4u, 2u, 1) != 0);
    CHECK(run_producer(argv[1], argv[3], fixture, failed,
                       binary_sha, 5u, 3u, 2u, 1) != 0);
    CHECK(join(path, sizeof(path), bad_fixture, "h_f32.bin"));
    for (size_t i = 0u; i < 2u; i++) {
        const float nonfinite[2] = {0.0f, i == 0u ? NAN : INFINITY};
        file = fopen(path, "wb");
        CHECK(file != NULL);
        CHECK(fwrite(nonfinite, sizeof(float), 2u, file) == 2u);
        CHECK(fclose(file) == 0);
        CHECK(run_producer(argv[1], argv[3], bad_fixture, failed,
                           binary_sha, 2u, 1u, 1u, 1) != 0);
        CHECK(!present(failed));
    }
    for (size_t i = 0u; i < 3u; i++) {
        CHECK(join(path, sizeof(path), fixture, source_names[i]));
        CHECK(file_sha(path, source_after));
        CHECK(strcmp(source_after, source_before[i]) == 0);
    }
    remove_artifact(first);
    remove_artifact(second);
    remove_fixture(fixture);
    remove_fixture(bad_fixture);
    CHECK(remove_dir(base));
    puts("IIC_AFFINITY_PRODUCER_FIXTURE_PASS");
    return 0;
}
