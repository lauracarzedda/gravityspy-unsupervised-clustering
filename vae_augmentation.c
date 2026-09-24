#include "vae_augmentation.h"

#include <stdio.h>


/*
   Seed sostitutivo usato se viene passato seed = 0.

   Il generatore xorshift32 non può lavorare
   con uno stato permanentemente uguale a zero.
*/
#define GS_AUGMENT_FALLBACK_SEED 0xA341316Cu

/*
   Domain tag congelato per augmentation_stateless_v1.
   0x4753415547563121 codifica "GSAUGV1!" in ASCII big-endian ed e'
   distinto dai domini del generatore latente VAE.
*/
#define GS_AUGMENT_STATELESS_V1_DOMAIN_TAG \
    UINT64_C(0x4753415547563121)


static uint64_t gs_augment_stateless_mix64(uint64_t value)
{
    value += UINT64_C(0x9E3779B97F4A7C15);
    value =
        (value ^ (value >> 30)) *
        UINT64_C(0xBF58476D1CE4E5B9);
    value =
        (value ^ (value >> 27)) *
        UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31);
}


static uint64_t gs_augment_stateless_value(
    uint64_t base_seed,
    uint64_t epoch,
    uint64_t manifest_index,
    GSAugmentMode mode,
    uint64_t stream_index
)
{
    uint64_t hash;

    hash = gs_augment_stateless_mix64(
        GS_AUGMENT_STATELESS_V1_DOMAIN_TAG
    );
    hash = gs_augment_stateless_mix64(hash ^ base_seed);
    hash = gs_augment_stateless_mix64(hash ^ epoch);
    hash = gs_augment_stateless_mix64(hash ^ manifest_index);
    hash = gs_augment_stateless_mix64(hash ^ (uint64_t)mode);
    return gs_augment_stateless_mix64(hash ^ stream_index);
}


const char *gs_augment_mode_name(
    GSAugmentMode mode
)
{
    switch (mode) {
        case GS_AUGMENT_MODE_LEGACY_TEXT:
            return "legacy_text";

        case GS_AUGMENT_MODE_SYMMETRIC_FIGURE:
            return "symmetric_figure";

        default:
            return "unknown";
    }
}


int gs_augment_mode_crop_range(
    GSAugmentMode mode,
    int *min_crop_left,
    int *max_crop_left
)
{
    if (min_crop_left == NULL ||
        max_crop_left == NULL) {

        printf("Errore: puntatore intervallo crop NULL.\n");
        return 1;
    }

    switch (mode) {
        case GS_AUGMENT_MODE_LEGACY_TEXT:
            *min_crop_left = GS_SAKAI_MIN_CROP_LEFT;
            *max_crop_left = GS_SAKAI_MAX_CROP_LEFT;
            return 0;

        case GS_AUGMENT_MODE_SYMMETRIC_FIGURE:
            *min_crop_left = GS_EVENT_MIN_CROP_LEFT;
            *max_crop_left = GS_EVENT_MAX_CROP_LEFT;
            return 0;

        default:
            printf("Errore: modalità augmentation non valida: %d\n",
                   (int)mode);
            return 1;
    }
}


int gs_augment_crop_left_for_event_mode(
    uint64_t base_seed,
    uint64_t epoch,
    uint64_t manifest_index,
    GSAugmentMode mode,
    int *crop_left
)
{
    int min_crop_left;
    int max_crop_left;
    int candidate;
    uint64_t range;
    uint64_t threshold;
    uint64_t stream_index;
    uint64_t random_value;

    if (crop_left == NULL) {
        return 1;
    }

    switch (mode) {
        case GS_AUGMENT_MODE_LEGACY_TEXT:
            min_crop_left = GS_SAKAI_MIN_CROP_LEFT;
            max_crop_left = GS_SAKAI_MAX_CROP_LEFT;
            break;

        case GS_AUGMENT_MODE_SYMMETRIC_FIGURE:
            min_crop_left = GS_EVENT_MIN_CROP_LEFT;
            max_crop_left = GS_EVENT_MAX_CROP_LEFT;
            break;

        default:
            return 1;
    }

    if (min_crop_left < 0 || max_crop_left < min_crop_left) {
        return 1;
    }

    range =
        (uint64_t)(unsigned int)(max_crop_left - min_crop_left) +
        UINT64_C(1);
    if (range == UINT64_C(0)) {
        return 1;
    }

    /*
       L'aritmetica unsigned rende (0 - range) definito modulo 2^64.
       I valori minori di threshold vengono scartati, lasciando un
       numero di risultati accettati esattamente divisibile per range.
    */
    threshold = (UINT64_C(0) - range) % range;
    stream_index = UINT64_C(0);

    for (;;) {
        random_value = gs_augment_stateless_value(
            base_seed,
            epoch,
            manifest_index,
            mode,
            stream_index
        );

        if (random_value >= threshold) {
            break;
        }

        if (stream_index == UINT64_MAX) {
            return 1;
        }
        stream_index++;
    }

    candidate =
        min_crop_left +
        (int)(random_value % range);

    if (candidate < min_crop_left ||
        candidate > max_crop_left ||
        !gs_event_crop_left_is_valid(candidate)) {

        return 1;
    }

    *crop_left = candidate;
    return 0;
}


void gs_augment_rng_seed(
    GSAugmentRng *rng,
    uint32_t seed
)
{
    if (rng == NULL) {
        return;
    }

    if (seed == 0u) {
        rng->state = GS_AUGMENT_FALLBACK_SEED;
    } else {
        rng->state = seed;
    }
}


uint32_t gs_augment_rng_next(
    GSAugmentRng *rng
)
{
    uint32_t x;

    if (rng == NULL) {
        return 0u;
    }

    if (rng->state == 0u) {
        rng->state = GS_AUGMENT_FALLBACK_SEED;
    }

    x = rng->state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    rng->state = x;

    return x;
}


int gs_augment_random_crop_left_mode(
    GSAugmentRng *rng,
    GSAugmentMode mode,
    int *crop_left
)
{
    int min_crop_left;
    int max_crop_left;

    uint32_t random_value;
    uint32_t range;

    if (rng == NULL) {
        printf("Errore: rng NULL.\n");
        return 1;
    }

    if (crop_left == NULL) {
        printf("Errore: crop_left NULL.\n");
        return 1;
    }

    if (gs_augment_mode_crop_range(
            mode,
            &min_crop_left,
            &max_crop_left
        ) != 0) {

        return 1;
    }

    range = (uint32_t)(
        max_crop_left -
        min_crop_left +
        1
    );

    if (range == 0u) {
        printf("Errore: intervallo augmentation vuoto.\n");
        return 1;
    }

    random_value = gs_augment_rng_next(rng);

    *crop_left =
        min_crop_left +
        (int)(random_value % range);

    if (*crop_left < min_crop_left ||
        *crop_left > max_crop_left) {

        printf("Errore: crop fuori dall'intervallo della modalità.\n");
        printf("modalità   = %s\n", gs_augment_mode_name(mode));
        printf("crop_left  = %d\n", *crop_left);
        printf("intervallo = %d-%d\n",
               min_crop_left,
               max_crop_left);

        return 1;
    }

    if (!gs_event_crop_left_is_valid(*crop_left)) {
        printf("Errore: il generatore ha prodotto "
               "un crop geometricamente non valido.\n");
        printf("crop_left = %d\n", *crop_left);
        return 1;
    }

    return 0;
}


int gs_augment_random_crop_left(
    GSAugmentRng *rng,
    int *crop_left
)
{
    return gs_augment_random_crop_left_mode(
        rng,
        GS_AUGMENT_MODE_LEGACY_TEXT,
        crop_left
    );
}


int gs_event_load_random_crop_float01_mode(
    const GSEventRecord *record,
    GSAugmentRng *rng,
    GSAugmentMode mode,
    int *used_crop_left,
    float *out_f32
)
{
    int crop_left;
    int rc;

    if (record == NULL) {
        printf("Errore: record evento NULL.\n");
        return 1;
    }

    if (rng == NULL) {
        printf("Errore: rng NULL.\n");
        return 1;
    }

    if (used_crop_left == NULL) {
        printf("Errore: used_crop_left NULL.\n");
        return 1;
    }

    if (out_f32 == NULL) {
        printf("Errore: buffer float NULL.\n");
        return 1;
    }

    rc = gs_augment_random_crop_left_mode(
        rng,
        mode,
        &crop_left
    );

    if (rc != 0) {
        return 1;
    }

    rc = gs_event_load_crop_float01(
        record,
        crop_left,
        out_f32
    );

    if (rc != 0) {
        return 1;
    }

    *used_crop_left = crop_left;

    return 0;
}


int gs_event_load_random_crop_float01(
    const GSEventRecord *record,
    GSAugmentRng *rng,
    int *used_crop_left,
    float *out_f32
)
{
    return gs_event_load_random_crop_float01_mode(
        record,
        rng,
        GS_AUGMENT_MODE_LEGACY_TEXT,
        used_crop_left,
        out_f32
    );
}
