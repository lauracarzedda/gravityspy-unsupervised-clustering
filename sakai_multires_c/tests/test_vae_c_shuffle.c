#include "vae_c_shuffle.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }

    fprintf(stderr, "[FAIL] %s\n", name);
    return 1;
}


static int permutation_is_valid(const size_t *permutation, size_t count)
{
    unsigned char *seen;
    size_t index;
    int valid;

    if (count == 0) {
        return 1;
    }
    if (permutation == NULL) {
        return 0;
    }

    seen = (unsigned char *)calloc(count, sizeof(*seen));
    if (seen == NULL) {
        return 0;
    }

    valid = 1;
    for (index = 0; index < count; index++) {
        size_t value = permutation[index];

        if (value >= count || seen[value] != 0) {
            valid = 0;
            break;
        }
        seen[value] = 1;
    }

    if (valid) {
        for (index = 0; index < count; index++) {
            if (seen[index] == 0) {
                valid = 0;
                break;
            }
        }
    }

    free(seen);
    return valid;
}


static int test_count_zero_and_one(void)
{
    size_t one = SIZE_MAX;

    return
        vae_c_shuffle_make_permutation(
            0,
            UINT64_C(101112),
            UINT64_C(0),
            NULL) == VAE_C_RNG_OK &&
        vae_c_shuffle_make_permutation(
            1,
            UINT64_C(101112),
            UINT64_C(0),
            &one) == VAE_C_RNG_OK &&
        one == 0;
}


static int test_invalid_arguments_strong_guarantee(void)
{
    size_t permutation[] = {9, 8, 7, 6};
    const size_t original[] = {9, 8, 7, 6};
    VAECRng rng;
    uint64_t output;
    uint64_t original_state;

    if (vae_c_shuffle_make_permutation(
            4,
            UINT64_C(1),
            UINT64_C(2),
            NULL) != VAE_C_RNG_INVALID_ARGUMENT ||
        memcmp(permutation, original, sizeof(permutation)) != 0 ||
        vae_c_rng_init(&rng, UINT64_C(77)) != VAE_C_RNG_OK) {

        return 0;
    }

    original_state = rng.state;
    output = UINT64_C(0x1122334455667788);
    if (vae_c_shuffle_bounded_u64(
            NULL,
            UINT64_C(10),
            &output) != VAE_C_RNG_INVALID_ARGUMENT ||
        output != UINT64_C(0x1122334455667788) ||
        vae_c_shuffle_bounded_u64(
            &rng,
            UINT64_C(0),
            &output) != VAE_C_RNG_INVALID_ARGUMENT ||
        rng.state != original_state ||
        output != UINT64_C(0x1122334455667788) ||
        vae_c_shuffle_bounded_u64(
            &rng,
            UINT64_C(10),
            NULL) != VAE_C_RNG_INVALID_ARGUMENT ||
        rng.state != original_state) {

        return 0;
    }

    return memcmp(permutation, original, sizeof(permutation)) == 0;
}


static int test_permutation_validity(void)
{
    const size_t counts[] = {2, 3, 7, 16, 49, 257};
    size_t case_index;

    for (case_index = 0;
         case_index < sizeof(counts) / sizeof(counts[0]);
         case_index++) {

        size_t count = counts[case_index];
        size_t *permutation =
            (size_t *)malloc(count * sizeof(*permutation));
        int valid;

        if (permutation == NULL) {
            return 0;
        }

        valid =
            vae_c_shuffle_make_permutation(
                count,
                UINT64_C(101112),
                (uint64_t)case_index,
                permutation) == VAE_C_RNG_OK &&
            permutation_is_valid(permutation, count);

        free(permutation);
        if (!valid) {
            return 0;
        }
    }

    return 1;
}


static int test_replicas_and_call_independence(void)
{
    size_t first[32];
    size_t second[32];
    size_t intervening[19];
    size_t third[32];

    return
        vae_c_shuffle_make_permutation(
            32,
            UINT64_C(101112),
            UINT64_C(14),
            first) == VAE_C_RNG_OK &&
        vae_c_shuffle_make_permutation(
            32,
            UINT64_C(101112),
            UINT64_C(14),
            second) == VAE_C_RNG_OK &&
        memcmp(first, second, sizeof(first)) == 0 &&
        vae_c_shuffle_make_permutation(
            19,
            UINT64_C(999),
            UINT64_C(888),
            intervening) == VAE_C_RNG_OK &&
        vae_c_shuffle_make_permutation(
            32,
            UINT64_C(101112),
            UINT64_C(14),
            third) == VAE_C_RNG_OK &&
        memcmp(first, third, sizeof(first)) == 0;
}


static int test_golden_vectors(void)
{
    static const struct {
        size_t count;
        uint64_t seed;
        uint64_t epoch;
        size_t expected[12];
    } cases[] = {
        {
            8,
            UINT64_C(101112),
            UINT64_C(0),
            {0, 5, 4, 7, 1, 3, 6, 2}
        },
        {
            8,
            UINT64_C(101112),
            UINT64_C(1),
            {2, 3, 1, 4, 0, 5, 6, 7}
        },
        {
            8,
            UINT64_C(202224),
            UINT64_C(0),
            {3, 6, 4, 2, 1, 7, 5, 0}
        },
        {
            12,
            UINT64_C(101112),
            UINT64_C(7),
            {8, 11, 10, 6, 2, 0, 9, 4, 5, 1, 7, 3}
        }
    };
    size_t case_index;

    for (case_index = 0;
         case_index < sizeof(cases) / sizeof(cases[0]);
         case_index++) {

        size_t actual[12];

        if (vae_c_shuffle_make_permutation(
                cases[case_index].count,
                cases[case_index].seed,
                cases[case_index].epoch,
                actual) != VAE_C_RNG_OK ||
            memcmp(
                actual,
                cases[case_index].expected,
                cases[case_index].count * sizeof(actual[0])) != 0) {

            return 0;
        }
    }

    return 1;
}


static int test_rejection_path(void)
{
    const uint64_t state_before_zero_output =
        UINT64_C(0x61C8864680B583EB);
    const uint64_t splitmix_increment =
        UINT64_C(0x9E3779B97F4A7C15);
    VAECRng rng;
    uint64_t value = UINT64_MAX;

    if (vae_c_rng_init(&rng, state_before_zero_output) != VAE_C_RNG_OK ||
        vae_c_shuffle_bounded_u64(
            &rng,
            UINT64_C(10),
            &value) != VAE_C_RNG_OK) {

        return 0;
    }

    /* range 10 has threshold 6: the first generated zero is rejected. */
    return
        rng.state == splitmix_increment &&
        value == UINT64_C(5);
}


static int test_project_seed_full_split(void)
{
    const size_t count = 6866;
    size_t *first = (size_t *)malloc(count * sizeof(*first));
    size_t *second = (size_t *)malloc(count * sizeof(*second));
    int valid;

    if (first == NULL || second == NULL) {
        free(first);
        free(second);
        return 0;
    }

    valid =
        vae_c_shuffle_make_permutation(
            count,
            UINT64_C(101112),
            UINT64_C(0),
            first) == VAE_C_RNG_OK &&
        vae_c_shuffle_make_permutation(
            count,
            UINT64_C(101112),
            UINT64_C(0),
            second) == VAE_C_RNG_OK &&
        permutation_is_valid(first, count) &&
        memcmp(first, second, count * sizeof(*first)) == 0;

    free(first);
    free(second);
    return valid;
}


static int test_source_arrays_unchanged(void)
{
    const size_t original[] = {
        7663, 2606, 7841, 4067, 634, 5033, 2410, 1046
    };
    size_t source[sizeof(original) / sizeof(original[0])];
    size_t permutation[sizeof(original) / sizeof(original[0])];
    size_t reordered[sizeof(original) / sizeof(original[0])];
    size_t index;

    memcpy(source, original, sizeof(source));
    if (vae_c_shuffle_make_permutation(
            sizeof(source) / sizeof(source[0]),
            UINT64_C(101112),
            UINT64_C(23),
            permutation) != VAE_C_RNG_OK) {

        return 0;
    }

    for (index = 0; index < sizeof(source) / sizeof(source[0]); index++) {
        reordered[index] = source[permutation[index]];
    }

    return
        memcmp(source, original, sizeof(source)) == 0 &&
        permutation_is_valid(
            permutation,
            sizeof(permutation) / sizeof(permutation[0])) &&
        memcmp(reordered, source, sizeof(source)) != 0;
}


int main(void)
{
    int failures = 0;

    failures += report_check(
        "count_zero_and_one",
        test_count_zero_and_one()
    );
    failures += report_check(
        "invalid_arguments_strong_guarantee",
        test_invalid_arguments_strong_guarantee()
    );
    failures += report_check(
        "permutation_validity",
        test_permutation_validity()
    );
    failures += report_check(
        "replicas_and_call_independence",
        test_replicas_and_call_independence()
    );
    failures += report_check("golden_vectors", test_golden_vectors());
    failures += report_check("rejection_path", test_rejection_path());
    failures += report_check(
        "project_seed_full_split",
        test_project_seed_full_split()
    );
    failures += report_check(
        "source_arrays_unchanged",
        test_source_arrays_unchanged()
    );

    if (failures != 0) {
        fprintf(stderr, "%d VAE C SHUFFLE TEST(S) FAILED\n", failures);
        return 1;
    }

    printf("SHUFFLE_BASE_SEED_PROJECT=101112\n");
    printf("SHUFFLE_DOMAIN=VAE_C_RNG_DOMAIN_SHUFFLE\n");
    printf("SHUFFLE_MAPPING=UNBIASED_REJECTION\n");
    printf("ALL VAE C SHUFFLE TESTS PASSED\n");
    return 0;
}
