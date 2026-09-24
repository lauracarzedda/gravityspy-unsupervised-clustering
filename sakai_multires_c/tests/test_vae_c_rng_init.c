#include "vae_c_init.h"
#include "vae_c_rng.h"

#include <math.h>
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


static int arrays_bitwise_equal(
    const float *left,
    const float *right,
    size_t count
)
{
    return memcmp(left, right, count * sizeof(float)) == 0;
}


static int arrays_differ(
    const float *left,
    const float *right,
    size_t count
)
{
    return !arrays_bitwise_equal(left, right, count);
}


static int all_equal(const float *values, size_t count, float expected)
{
    size_t index;

    for (index = 0; index < count; index++) {
        if (values[index] != expected) {
            return 0;
        }
    }

    return 1;
}


static int test_splitmix64_known_answers(void)
{
    const uint64_t expected[] = {
        UINT64_C(0xE220A8397B1DCDAF),
        UINT64_C(0x6E789E6AA1B965F4),
        UINT64_C(0x06C45D188009454F),
        UINT64_C(0xF88BB8A8724C81EC),
        UINT64_C(0x1B39896A51A8749B)
    };
    VAECRng rng;
    size_t index;

    if (vae_c_rng_init(&rng, UINT64_C(0)) != VAE_C_RNG_OK) {
        return 0;
    }

    for (index = 0; index < sizeof(expected) / sizeof(expected[0]);
         index++) {
        uint64_t value;

        if (vae_c_rng_next_u64(&rng, &value) != VAE_C_RNG_OK ||
            value != expected[index]) {

            return 0;
        }
    }

    return 1;
}


static int test_seed_derivation_known_answers(void)
{
    static const struct {
        uint64_t base_seed;
        VAECRngDomain domain;
        uint64_t epoch;
        uint64_t manifest_index;
        uint64_t stream_index;
        uint64_t expected;
    } cases[] = {
        {
            UINT64_C(123),
            VAE_C_RNG_DOMAIN_INITIALIZATION,
            UINT64_C(0),
            UINT64_C(0),
            UINT64_C(7),
            UINT64_C(0x1E073756348C0AD9)
        },
        {
            UINT64_C(456),
            VAE_C_RNG_DOMAIN_LATENT_TRAIN,
            UINT64_C(3),
            UINT64_C(42),
            UINT64_C(0),
            UINT64_C(0xBFF9489E190BCDD4)
        },
        {
            UINT64_C(456),
            VAE_C_RNG_DOMAIN_LATENT_VALIDATION,
            UINT64_C(0),
            UINT64_C(42),
            UINT64_C(0),
            UINT64_C(0xB849E85BFE7894E7)
        },
        {
            UINT64_C(0),
            VAE_C_RNG_DOMAIN_INITIALIZATION,
            UINT64_C(0),
            UINT64_C(0),
            UINT64_C(0),
            UINT64_C(0xAAFF81EB53AF65A4)
        }
    };
    uint64_t reference;
    uint64_t candidate;
    uint64_t sentinel;
    size_t index;

    for (index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        if (vae_c_rng_derive_seed(
                cases[index].base_seed,
                cases[index].domain,
                cases[index].epoch,
                cases[index].manifest_index,
                cases[index].stream_index,
                &candidate) != VAE_C_RNG_OK ||
            candidate != cases[index].expected) {

            return 0;
        }
    }

    if (vae_c_rng_derive_seed(
            UINT64_C(123),
            VAE_C_RNG_DOMAIN_LATENT_TRAIN,
            UINT64_C(4),
            UINT64_C(91),
            UINT64_C(2),
            &reference) != VAE_C_RNG_OK) {

        return 0;
    }

    if (vae_c_rng_derive_seed(
            UINT64_C(124),
            VAE_C_RNG_DOMAIN_LATENT_TRAIN,
            UINT64_C(4),
            UINT64_C(91),
            UINT64_C(2),
            &candidate) != VAE_C_RNG_OK ||
        candidate == reference) {

        return 0;
    }

    if (vae_c_rng_derive_seed(
            UINT64_C(123),
            VAE_C_RNG_DOMAIN_LATENT_VALIDATION,
            UINT64_C(4),
            UINT64_C(91),
            UINT64_C(2),
            &candidate) != VAE_C_RNG_OK ||
        candidate == reference) {

        return 0;
    }

    if (vae_c_rng_derive_seed(
            UINT64_C(123),
            VAE_C_RNG_DOMAIN_LATENT_TRAIN,
            UINT64_C(5),
            UINT64_C(91),
            UINT64_C(2),
            &candidate) != VAE_C_RNG_OK ||
        candidate == reference) {

        return 0;
    }

    if (vae_c_rng_derive_seed(
            UINT64_C(123),
            VAE_C_RNG_DOMAIN_LATENT_TRAIN,
            UINT64_C(4),
            UINT64_C(92),
            UINT64_C(2),
            &candidate) != VAE_C_RNG_OK ||
        candidate == reference) {

        return 0;
    }

    if (vae_c_rng_derive_seed(
            UINT64_C(123),
            VAE_C_RNG_DOMAIN_LATENT_TRAIN,
            UINT64_C(4),
            UINT64_C(91),
            UINT64_C(3),
            &candidate) != VAE_C_RNG_OK ||
        candidate == reference) {

        return 0;
    }

    sentinel = UINT64_C(0x1122334455667788);
    if (vae_c_rng_derive_seed(
            UINT64_C(1),
            (VAECRngDomain)0,
            UINT64_C(0),
            UINT64_C(0),
            UINT64_C(0),
            &sentinel) != VAE_C_RNG_INVALID_DOMAIN ||
        sentinel != UINT64_C(0x1122334455667788)) {

        return 0;
    }

    return
        vae_c_rng_derive_seed(
            UINT64_C(1),
            (VAECRngDomain)6,
            UINT64_C(0),
            UINT64_C(0),
            UINT64_C(0),
            &sentinel) == VAE_C_RNG_INVALID_DOMAIN;
}


static int test_uniform_contracts(void)
{
    const size_t sample_count = 100000;
    VAECRng float_rng_a;
    VAECRng float_rng_b;
    VAECRng float_rng_different;
    VAECRng double_rng_a;
    VAECRng double_rng_b;
    double float_sum;
    double double_sum;
    int saw_different;
    size_t index;

    if (vae_c_rng_init(&float_rng_a, UINT64_C(2022)) != VAE_C_RNG_OK ||
        vae_c_rng_init(&float_rng_b, UINT64_C(2022)) != VAE_C_RNG_OK ||
        vae_c_rng_init(&float_rng_different, UINT64_C(2023)) !=
            VAE_C_RNG_OK ||
        vae_c_rng_init(&double_rng_a, UINT64_C(9090)) != VAE_C_RNG_OK ||
        vae_c_rng_init(&double_rng_b, UINT64_C(9090)) != VAE_C_RNG_OK) {

        return 0;
    }

    float_sum = 0.0;
    double_sum = 0.0;
    saw_different = 0;

    for (index = 0; index < sample_count; index++) {
        float float_a;
        float float_b;
        float float_different;
        double double_a;
        double double_b;

        if (vae_c_rng_uniform_f32(&float_rng_a, &float_a) !=
                VAE_C_RNG_OK ||
            vae_c_rng_uniform_f32(&float_rng_b, &float_b) !=
                VAE_C_RNG_OK ||
            vae_c_rng_uniform_f32(
                &float_rng_different,
                &float_different) != VAE_C_RNG_OK ||
            vae_c_rng_uniform_open_f64(&double_rng_a, &double_a) !=
                VAE_C_RNG_OK ||
            vae_c_rng_uniform_open_f64(&double_rng_b, &double_b) !=
                VAE_C_RNG_OK) {

            return 0;
        }

        if (memcmp(&float_a, &float_b, sizeof(float)) != 0 ||
            memcmp(&double_a, &double_b, sizeof(double)) != 0 ||
            float_a < 0.0f ||
            float_a >= 1.0f ||
            double_a <= 0.0 ||
            double_a >= 1.0) {

            return 0;
        }

        if (memcmp(
                &float_a,
                &float_different,
                sizeof(float)) != 0) {

            saw_different = 1;
        }

        float_sum += (double)float_a;
        double_sum += double_a;
    }

    return
        saw_different &&
        fabs(float_sum / (double)sample_count - 0.5) < 0.01 &&
        fabs(double_sum / (double)sample_count - 0.5) < 0.01;
}


static int test_standard_normal_contracts(void)
{
    const size_t sample_count = 100000;
    float *filled;
    float *repeated;
    float *scalar;
    VAECRng fill_rng;
    VAECRng repeat_rng;
    VAECRng scalar_rng;
    double sum;
    double square_sum;
    double mean;
    double variance;
    size_t index;
    int result;

    filled = (float *)malloc(sample_count * sizeof(float));
    repeated = (float *)malloc(sample_count * sizeof(float));
    scalar = (float *)malloc(sample_count * sizeof(float));
    if (filled == NULL || repeated == NULL || scalar == NULL) {
        free(filled);
        free(repeated);
        free(scalar);
        return 0;
    }

    result = 0;

    if (vae_c_rng_init(&fill_rng, UINT64_C(7331)) != VAE_C_RNG_OK ||
        vae_c_rng_init(&repeat_rng, UINT64_C(7331)) != VAE_C_RNG_OK ||
        vae_c_rng_init(&scalar_rng, UINT64_C(7331)) != VAE_C_RNG_OK ||
        vae_c_rng_fill_standard_normal_f32(
            &fill_rng,
            filled,
            sample_count) != VAE_C_RNG_OK ||
        vae_c_rng_fill_standard_normal_f32(
            &repeat_rng,
            repeated,
            sample_count) != VAE_C_RNG_OK) {

        goto cleanup;
    }

    for (index = 0; index < sample_count; index++) {
        if (vae_c_rng_standard_normal_f32(
                &scalar_rng,
                &scalar[index]) != VAE_C_RNG_OK) {

            goto cleanup;
        }
    }

    if (!arrays_bitwise_equal(filled, repeated, sample_count) ||
        !arrays_bitwise_equal(filled, scalar, sample_count) ||
        fill_rng.state != scalar_rng.state) {

        goto cleanup;
    }

    sum = 0.0;
    square_sum = 0.0;

    for (index = 0; index < sample_count; index++) {
        double value = (double)filled[index];

        if (!isfinite(value)) {
            goto cleanup;
        }

        sum += value;
        square_sum += value * value;
    }

    mean = sum / (double)sample_count;
    variance =
        square_sum / (double)sample_count -
        mean * mean;

    result =
        fabs(mean) < 0.02 &&
        variance > 0.95 &&
        variance < 1.05;

cleanup:
    free(filled);
    free(repeated);
    free(scalar);
    return result;
}


static int test_event_and_batch_independence(void)
{
    enum { latent_count = 64 };
    float train_reference[latent_count];
    float train_repeat[latent_count];
    float train_next_epoch[latent_count];
    float train_other_event[latent_count];
    float validation_a[latent_count];
    float validation_repeat[latent_count];
    float order_ab_a[latent_count];
    float order_ab_b[latent_count];
    float order_ba_a[latent_count];
    float order_ba_b[latent_count];

    if (vae_c_rng_fill_latent_train_epsilon_f32(
            UINT64_C(456),
            UINT64_C(3),
            UINT64_C(42),
            train_reference,
            latent_count) != VAE_C_RNG_OK ||
        vae_c_rng_fill_latent_train_epsilon_f32(
            UINT64_C(456),
            UINT64_C(3),
            UINT64_C(42),
            train_repeat,
            latent_count) != VAE_C_RNG_OK ||
        vae_c_rng_fill_latent_train_epsilon_f32(
            UINT64_C(456),
            UINT64_C(4),
            UINT64_C(42),
            train_next_epoch,
            latent_count) != VAE_C_RNG_OK ||
        vae_c_rng_fill_latent_train_epsilon_f32(
            UINT64_C(456),
            UINT64_C(3),
            UINT64_C(43),
            train_other_event,
            latent_count) != VAE_C_RNG_OK ||
        vae_c_rng_fill_latent_validation_epsilon_f32(
            UINT64_C(456),
            UINT64_C(42),
            validation_a,
            latent_count) != VAE_C_RNG_OK ||
        vae_c_rng_fill_latent_validation_epsilon_f32(
            UINT64_C(456),
            UINT64_C(42),
            validation_repeat,
            latent_count) != VAE_C_RNG_OK) {

        return 0;
    }

    if (!arrays_bitwise_equal(
            train_reference,
            train_repeat,
            latent_count) ||
        !arrays_differ(
            train_reference,
            train_next_epoch,
            latent_count) ||
        !arrays_differ(
            train_reference,
            train_other_event,
            latent_count) ||
        !arrays_bitwise_equal(
            validation_a,
            validation_repeat,
            latent_count)) {

        return 0;
    }

    if (vae_c_rng_fill_latent_train_epsilon_f32(
            UINT64_C(9001),
            UINT64_C(8),
            UINT64_C(100),
            order_ab_a,
            latent_count) != VAE_C_RNG_OK ||
        vae_c_rng_fill_latent_train_epsilon_f32(
            UINT64_C(9001),
            UINT64_C(8),
            UINT64_C(200),
            order_ab_b,
            latent_count) != VAE_C_RNG_OK ||
        vae_c_rng_fill_latent_train_epsilon_f32(
            UINT64_C(9001),
            UINT64_C(8),
            UINT64_C(200),
            order_ba_b,
            latent_count) != VAE_C_RNG_OK ||
        vae_c_rng_fill_latent_train_epsilon_f32(
            UINT64_C(9001),
            UINT64_C(8),
            UINT64_C(100),
            order_ba_a,
            latent_count) != VAE_C_RNG_OK) {

        return 0;
    }

    return
        arrays_bitwise_equal(order_ab_a, order_ba_a, latent_count) &&
        arrays_bitwise_equal(order_ab_b, order_ba_b, latent_count);
}


static int test_basic_initializers(void)
{
    enum { count = 32 };
    float zero_values[count];
    float one_values[count];
    float gamma[count];
    float beta[count];
    float running_mean[count];
    float running_variance[count];

    memset(zero_values, 0xA5, sizeof(zero_values));
    memset(one_values, 0xA5, sizeof(one_values));
    memset(gamma, 0xA5, sizeof(gamma));
    memset(beta, 0xA5, sizeof(beta));
    memset(running_mean, 0xA5, sizeof(running_mean));
    memset(running_variance, 0xA5, sizeof(running_variance));

    if (vae_c_init_zero_f32(zero_values, count) != VAE_C_INIT_OK ||
        vae_c_init_one_f32(one_values, count) != VAE_C_INIT_OK ||
        vae_c_init_batchnorm_f32(
            gamma,
            beta,
            running_mean,
            running_variance,
            count) != VAE_C_INIT_OK) {

        return 0;
    }

    return
        all_equal(zero_values, count, 0.0f) &&
        all_equal(one_values, count, 1.0f) &&
        all_equal(gamma, count, 1.0f) &&
        all_equal(beta, count, 0.0f) &&
        all_equal(running_mean, count, 0.0f) &&
        all_equal(running_variance, count, 1.0f);
}


static int test_uniform_initializers(void)
{
    const size_t count = 100000;
    const double he_bound = sqrt(6.0 / 128.0);
    const double xavier_bound = sqrt(6.0 / (128.0 + 64.0));
    float *he_a;
    float *he_b;
    float *he_other_key;
    float *xavier_a;
    float *xavier_b;
    double he_sum;
    double xavier_sum;
    size_t index;
    int result;

    he_a = (float *)malloc(count * sizeof(float));
    he_b = (float *)malloc(count * sizeof(float));
    he_other_key = (float *)malloc(count * sizeof(float));
    xavier_a = (float *)malloc(count * sizeof(float));
    xavier_b = (float *)malloc(count * sizeof(float));

    if (he_a == NULL ||
        he_b == NULL ||
        he_other_key == NULL ||
        xavier_a == NULL ||
        xavier_b == NULL) {

        free(he_a);
        free(he_b);
        free(he_other_key);
        free(xavier_a);
        free(xavier_b);
        return 0;
    }

    result = 0;

    if (vae_c_init_he_uniform_f32(
            he_a,
            count,
            UINT64_C(123),
            UINT64_C(10),
            128) != VAE_C_INIT_OK ||
        vae_c_init_he_uniform_f32(
            he_b,
            count,
            UINT64_C(123),
            UINT64_C(10),
            128) != VAE_C_INIT_OK ||
        vae_c_init_he_uniform_f32(
            he_other_key,
            count,
            UINT64_C(123),
            UINT64_C(11),
            128) != VAE_C_INIT_OK ||
        vae_c_init_xavier_uniform_f32(
            xavier_a,
            count,
            UINT64_C(123),
            UINT64_C(20),
            128,
            64) != VAE_C_INIT_OK ||
        vae_c_init_xavier_uniform_f32(
            xavier_b,
            count,
            UINT64_C(123),
            UINT64_C(20),
            128,
            64) != VAE_C_INIT_OK) {

        goto cleanup;
    }

    if (!arrays_bitwise_equal(he_a, he_b, count) ||
        !arrays_bitwise_equal(xavier_a, xavier_b, count) ||
        !arrays_differ(he_a, he_other_key, count)) {

        goto cleanup;
    }

    he_sum = 0.0;
    xavier_sum = 0.0;

    for (index = 0; index < count; index++) {
        double he_value = (double)he_a[index];
        double xavier_value = (double)xavier_a[index];

        if (!isfinite(he_value) ||
            he_value < -(double)(float)he_bound ||
            he_value > (double)(float)he_bound ||
            !isfinite(xavier_value) ||
            xavier_value < -(double)(float)xavier_bound ||
            xavier_value > (double)(float)xavier_bound) {

            goto cleanup;
        }

        he_sum += he_value;
        xavier_sum += xavier_value;
    }

    result =
        fabs(he_sum / (double)count) < 0.002 &&
        fabs(xavier_sum / (double)count) < 0.002;

cleanup:
    free(he_a);
    free(he_b);
    free(he_other_key);
    free(xavier_a);
    free(xavier_b);
    return result;
}


static int test_initialization_order_independence(void)
{
    enum { count_a = 73, count_b = 41 };
    float order_ab_a[count_a];
    float order_ab_b[count_b];
    float order_ba_a[count_a];
    float order_ba_b[count_b];

    if (vae_c_init_he_uniform_f32(
            order_ab_a,
            count_a,
            UINT64_C(777),
            UINT64_C(1001),
            64) != VAE_C_INIT_OK ||
        vae_c_init_xavier_uniform_f32(
            order_ab_b,
            count_b,
            UINT64_C(777),
            UINT64_C(1002),
            64,
            32) != VAE_C_INIT_OK ||
        vae_c_init_xavier_uniform_f32(
            order_ba_b,
            count_b,
            UINT64_C(777),
            UINT64_C(1002),
            64,
            32) != VAE_C_INIT_OK ||
        vae_c_init_he_uniform_f32(
            order_ba_a,
            count_a,
            UINT64_C(777),
            UINT64_C(1001),
            64) != VAE_C_INIT_OK) {

        return 0;
    }

    return
        arrays_bitwise_equal(order_ab_a, order_ba_a, count_a) &&
        arrays_bitwise_equal(order_ab_b, order_ba_b, count_b);
}


static int test_invalid_arguments_and_strong_guarantee(void)
{
    float sentinel[8];
    float original[8];
    float gamma[8];
    float beta[8];
    float mean[8];
    float variance[8];
    float gamma_original[8];
    float beta_original[8];
    float mean_original[8];
    float variance_original[8];
    VAECRng rng;
    uint64_t next_sentinel;
    size_t index;

    for (index = 0; index < 8; index++) {
        sentinel[index] = (float)(index + 1);
        gamma[index] = (float)(10 + index);
        beta[index] = (float)(20 + index);
        mean[index] = (float)(30 + index);
        variance[index] = (float)(40 + index);
    }

    memcpy(original, sentinel, sizeof(sentinel));
    memcpy(gamma_original, gamma, sizeof(gamma));
    memcpy(beta_original, beta, sizeof(beta));
    memcpy(mean_original, mean, sizeof(mean));
    memcpy(variance_original, variance, sizeof(variance));

    if (vae_c_init_zero_f32(NULL, 0) != VAE_C_INIT_OK ||
        vae_c_init_one_f32(NULL, 0) != VAE_C_INIT_OK ||
        vae_c_init_batchnorm_f32(NULL, NULL, NULL, NULL, 0) !=
            VAE_C_INIT_OK ||
        vae_c_init_zero_f32(NULL, 1) !=
            VAE_C_INIT_INVALID_ARGUMENT ||
        vae_c_init_one_f32(NULL, 1) !=
            VAE_C_INIT_INVALID_ARGUMENT ||
        vae_c_init_batchnorm_f32(
            gamma,
            beta,
            NULL,
            variance,
            8) != VAE_C_INIT_INVALID_ARGUMENT ||
        memcmp(gamma, gamma_original, sizeof(gamma)) != 0 ||
        memcmp(beta, beta_original, sizeof(beta)) != 0 ||
        memcmp(mean, mean_original, sizeof(mean)) != 0 ||
        memcmp(variance, variance_original, sizeof(variance)) != 0) {

        return 0;
    }

    if (vae_c_init_he_uniform_f32(
            sentinel,
            8,
            UINT64_C(1),
            UINT64_C(1),
            0) != VAE_C_INIT_INVALID_FAN ||
        memcmp(sentinel, original, sizeof(sentinel)) != 0 ||
        vae_c_init_he_uniform_f32(
            NULL,
            1,
            UINT64_C(1),
            UINT64_C(1),
            1) != VAE_C_INIT_INVALID_ARGUMENT ||
        vae_c_init_he_uniform_f32(
            NULL,
            0,
            UINT64_C(1),
            UINT64_C(1),
            1) != VAE_C_INIT_OK) {

        return 0;
    }

    if (vae_c_init_xavier_uniform_f32(
            sentinel,
            8,
            UINT64_C(1),
            UINT64_C(1),
            1,
            0) != VAE_C_INIT_INVALID_FAN ||
        memcmp(sentinel, original, sizeof(sentinel)) != 0 ||
        vae_c_init_xavier_uniform_f32(
            sentinel,
            8,
            UINT64_C(1),
            UINT64_C(1),
            SIZE_MAX,
            1) != VAE_C_INIT_FAN_SUM_OVERFLOW ||
        memcmp(sentinel, original, sizeof(sentinel)) != 0 ||
        vae_c_init_xavier_uniform_f32(
            NULL,
            1,
            UINT64_C(1),
            UINT64_C(1),
            1,
            1) != VAE_C_INIT_INVALID_ARGUMENT ||
        vae_c_init_xavier_uniform_f32(
            NULL,
            0,
            UINT64_C(1),
            UINT64_C(1),
            1,
            1) != VAE_C_INIT_OK) {

        return 0;
    }

    if (vae_c_rng_init(NULL, UINT64_C(0)) !=
            VAE_C_RNG_INVALID_ARGUMENT ||
        vae_c_rng_init(&rng, UINT64_C(0)) != VAE_C_RNG_OK ||
        vae_c_rng_next_u64(NULL, &next_sentinel) !=
            VAE_C_RNG_INVALID_ARGUMENT ||
        vae_c_rng_next_u64(&rng, NULL) !=
            VAE_C_RNG_INVALID_ARGUMENT ||
        rng.state != UINT64_C(0) ||
        vae_c_rng_fill_standard_normal_f32(NULL, NULL, 0) !=
            VAE_C_RNG_OK ||
        vae_c_rng_fill_standard_normal_f32(&rng, NULL, 1) !=
            VAE_C_RNG_INVALID_ARGUMENT ||
        rng.state != UINT64_C(0) ||
        vae_c_rng_fill_latent_train_epsilon_f32(
            UINT64_C(1),
            UINT64_C(1),
            UINT64_C(1),
            NULL,
            0) != VAE_C_RNG_OK ||
        vae_c_rng_fill_latent_validation_epsilon_f32(
            UINT64_C(1),
            UINT64_C(1),
            NULL,
            0) != VAE_C_RNG_OK) {

        return 0;
    }

    return 1;
}


static int test_status_names(void)
{
    size_t index;

    for (index = VAE_C_RNG_OK;
         index <= VAE_C_RNG_NONFINITE_RESULT;
         index++) {

        if (strcmp(
                vae_c_rng_status_name((VAECRngStatus)index),
                "unknown") == 0) {

            return 0;
        }
    }

    for (index = VAE_C_INIT_OK;
         index <= VAE_C_INIT_RNG_ERROR;
         index++) {

        if (strcmp(
                vae_c_init_status_name((VAECInitStatus)index),
                "unknown") == 0) {

            return 0;
        }
    }

    return
        strcmp(
            vae_c_rng_status_name((VAECRngStatus)999),
            "unknown") == 0 &&
        strcmp(
            vae_c_init_status_name((VAECInitStatus)999),
            "unknown") == 0;
}


int main(void)
{
    int failures = 0;

    failures += report_check(
        "splitmix64_known_answers",
        test_splitmix64_known_answers()
    );
    failures += report_check(
        "seed_derivation_known_answers",
        test_seed_derivation_known_answers()
    );
    failures += report_check(
        "uniform_contracts",
        test_uniform_contracts()
    );
    failures += report_check(
        "standard_normal_contracts",
        test_standard_normal_contracts()
    );
    failures += report_check(
        "event_and_batch_independence",
        test_event_and_batch_independence()
    );
    failures += report_check(
        "basic_initializers",
        test_basic_initializers()
    );
    failures += report_check(
        "uniform_initializers",
        test_uniform_initializers()
    );
    failures += report_check(
        "initialization_order_independence",
        test_initialization_order_independence()
    );
    failures += report_check(
        "invalid_arguments_and_strong_guarantee",
        test_invalid_arguments_and_strong_guarantee()
    );
    failures += report_check("status_names", test_status_names());

    if (failures != 0) {
        fprintf(stderr, "%d VAE C RNG/INIT TEST(S) FAILED\n", failures);
        return 1;
    }

    printf("ALL VAE C RNG/INIT TESTS PASSED\n");
    return 0;
}
