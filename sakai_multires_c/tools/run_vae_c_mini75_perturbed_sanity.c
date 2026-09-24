#include "vae_augmentation.h"
#include "vae_c_adam.h"
#include "vae_c_config.h"
#include "vae_c_model_forward.h"
#include "vae_c_model_train.h"
#include "vae_event_dataset.h"
#include "vae_event_loader.h"
#include "vae_multires_batch.h"
#include "vae_multires_event_loader.h"
#include "vae_saved_split.h"
#include "vae_tensor_io.h"
#include "vae_tensor_manifest.h"
#include "vae_tensor_spec.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif


enum {
    SANITY_PATH_CAPACITY = 4096,
    SANITY_SUBSET_SIZE = 16,
    SANITY_CHANNELS = 4,
    SANITY_SIDE = 75,
    SANITY_STEPS = 80,
    SANITY_CENTRAL_EVAL_EVERY = 10,
    SANITY_CROP_COUNT = 49
};

#define SANITY_PROFILE_ID "mini75_smoke_v1"
#define SANITY_OUTPUT_BASENAME \
    "mini75_smoke_v1_subset16_symmetric_figure_steps80_" \
    "init123_latent456_aug789"
#define SANITY_INIT_SEED UINT64_C(123)
#define SANITY_LATENT_BASE_SEED UINT64_C(456)
#define SANITY_AUGMENTATION_SEED UINT32_C(789)


static const uint64_t sanity_manifest_indices[SANITY_SUBSET_SIZE] = {
    UINT64_C(7663), UINT64_C(2606), UINT64_C(7841), UINT64_C(4067),
    UINT64_C(634),  UINT64_C(5033), UINT64_C(2410), UINT64_C(1046),
    UINT64_C(7023), UINT64_C(973),  UINT64_C(2244), UINT64_C(5554),
    UINT64_C(1452), UINT64_C(3739), UINT64_C(4937), UINT64_C(4224)
};


typedef struct {
    VAECProfileCatalog catalog;
    GSEventManifest event_manifest;
    GSSavedSplit train_split;
    GSTensorManifest tensor_manifest;
    GSMultiresEventLoader event_loader;
    GSMultiresBatch batch;
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECAdam optimizer;

    float *central_inputs;
    size_t central_input_bytes;
    float *parameter_snapshot;
    size_t parameter_snapshot_count;
    int selected_event_indices[SANITY_SUBSET_SIZE];

    FILE *run_config_file;
    FILE *metrics_file;
    FILE *augmentation_file;

    char output_parent[SANITY_PATH_CAPACITY];
    char final_output[SANITY_PATH_CAPACITY];
    char partial_output[SANITY_PATH_CAPACITY];
    bool partial_created;
} SanityRun;


static int sanity_size_multiply(
    size_t left,
    size_t right,
    size_t *result
)
{
    if (result == NULL || (left != 0 && right > SIZE_MAX / left)) {
        return 1;
    }
    *result = left * right;
    return 0;
}


static int sanity_path_join(
    const char *directory,
    const char *name,
    char *path,
    size_t capacity
)
{
    size_t directory_length;
    size_t name_length;
    size_t needed;
    size_t position = 0;
    bool separator;

    if (directory == NULL || name == NULL || path == NULL || capacity == 0) {
        return 1;
    }
    directory_length = strlen(directory);
    name_length = strlen(name);
    separator = directory_length != 0 &&
        directory[directory_length - 1] != '/' &&
        directory[directory_length - 1] != '\\';

    if (directory_length > SIZE_MAX - name_length) {
        return 1;
    }
    needed = directory_length + name_length;
    if (separator) {
        if (needed == SIZE_MAX) {
            return 1;
        }
        needed++;
    }
    if (needed >= capacity) {
        return 1;
    }
    if (directory_length != 0) {
        memcpy(path, directory, directory_length);
        position = directory_length;
    }
    if (separator) {
        path[position++] = '/';
    }
    if (name_length != 0) {
        memcpy(path + position, name, name_length);
        position += name_length;
    }
    path[position] = '\0';
    return 0;
}


static int sanity_path_append(
    const char *path,
    const char *suffix,
    char *output,
    size_t capacity
)
{
    size_t path_length;
    size_t suffix_length;

    if (path == NULL || suffix == NULL || output == NULL || capacity == 0) {
        return 1;
    }
    path_length = strlen(path);
    suffix_length = strlen(suffix);
    if (path_length > SIZE_MAX - suffix_length ||
        path_length + suffix_length >= capacity) {

        return 1;
    }
    memcpy(output, path, path_length);
    memcpy(output + path_length, suffix, suffix_length + 1);
    return 0;
}


static int sanity_code_commit_is_valid(const char *code_commit)
{
    size_t index;

    if (code_commit == NULL || strlen(code_commit) != 40) {
        return 0;
    }
    for (index = 0; index < 40; index++) {
        unsigned char character = (unsigned char)code_commit[index];

        if (!((character >= (unsigned char)'0' &&
               character <= (unsigned char)'9') ||
              (character >= (unsigned char)'a' &&
               character <= (unsigned char)'f') ||
              (character >= (unsigned char)'A' &&
               character <= (unsigned char)'F'))) {

            return 0;
        }
    }
    return 1;
}


static int sanity_path_state(const char *path, bool *is_directory)
{
    struct stat information;

    if (path == NULL) {
        return -1;
    }
    errno = 0;
    if (stat(path, &information) == 0) {
        if (is_directory != NULL) {
#ifdef _WIN32
            *is_directory = (information.st_mode & _S_IFDIR) != 0;
#else
            *is_directory = S_ISDIR(information.st_mode);
#endif
        }
        return 1;
    }
    if (errno == ENOENT) {
        if (is_directory != NULL) {
            *is_directory = false;
        }
        return 0;
    }
    return -1;
}


static int sanity_ensure_directory(const char *path)
{
    bool is_directory = false;
    int state = sanity_path_state(path, &is_directory);

    if (state == 1) {
        return is_directory ? 0 : 1;
    }
    if (state != 0) {
        return 1;
    }
#ifdef _WIN32
    return _mkdir(path) == 0 ? 0 : 1;
#else
    return mkdir(path, 0777) == 0 ? 0 : 1;
#endif
}


static int sanity_remove_directory(const char *path)
{
#ifdef _WIN32
    return _rmdir(path);
#else
    return rmdir(path);
#endif
}


static int sanity_finish_file(FILE **file)
{
    FILE *stream;
    int failed = 0;

    if (file == NULL || *file == NULL) {
        return 1;
    }
    stream = *file;
    *file = NULL;
    if (fflush(stream) != 0 || ferror(stream)) {
        failed = 1;
    }
    if (fclose(stream) != 0) {
        failed = 1;
    }
    return failed;
}


static int sanity_open_output(
    const char *directory,
    const char *name,
    const char *mode,
    FILE **file
)
{
    char path[SANITY_PATH_CAPACITY];

    if (file == NULL) {
        return 1;
    }
    *file = NULL;
    if (sanity_path_join(directory, name, path, sizeof(path)) != 0) {
        return 1;
    }
    *file = fopen(path, mode);
    return *file == NULL ? 1 : 0;
}


static void sanity_remove_partial_outputs(SanityRun *run)
{
    static const char *names[] = {
        "run_config.txt",
        "selected_events.csv",
        "metrics.csv",
        "augmentation.csv"
    };
    size_t index;

    if (run == NULL || !run->partial_created) {
        return;
    }
    for (index = 0; index < sizeof(names) / sizeof(names[0]); index++) {
        char path[SANITY_PATH_CAPACITY];

        if (sanity_path_join(
                run->partial_output,
                names[index],
                path,
                sizeof(path)) == 0) {

            (void)remove(path);
        }
    }
    (void)sanity_remove_directory(run->partial_output);
    run->partial_created = false;
}


static void sanity_run_init(SanityRun *run)
{
    if (run == NULL) {
        return;
    }
    memset(run, 0, sizeof(*run));
    vae_c_profile_catalog_init(&run->catalog);
    vae_c_model_init(&run->model);
    vae_c_model_workspace_init(&run->workspace);
    vae_c_adam_init(&run->optimizer);
}


static void sanity_run_free(SanityRun *run)
{
    if (run == NULL) {
        return;
    }
    if (run->augmentation_file != NULL) {
        (void)fclose(run->augmentation_file);
        run->augmentation_file = NULL;
    }
    if (run->metrics_file != NULL) {
        (void)fclose(run->metrics_file);
        run->metrics_file = NULL;
    }
    if (run->run_config_file != NULL) {
        (void)fclose(run->run_config_file);
        run->run_config_file = NULL;
    }
    sanity_remove_partial_outputs(run);
    free(run->parameter_snapshot);
    free(run->central_inputs);
    vae_c_adam_free(&run->optimizer);
    vae_c_model_workspace_free(&run->workspace);
    vae_c_model_free(&run->model);
    gs_multires_batch_free(&run->batch);
    gs_multires_event_loader_free(&run->event_loader);
    gs_tensor_manifest_free(&run->tensor_manifest);
    gs_saved_split_free(&run->train_split);
    gs_event_manifest_free(&run->event_manifest);
    vae_c_profile_catalog_free(&run->catalog);
    memset(run, 0, sizeof(*run));
}


static int sanity_profile_is_frozen(const VAECProfile *profile)
{
    return profile != NULL &&
        strcmp(profile->profile_id, SANITY_PROFILE_ID) == 0 &&
        strcmp(profile->role, "engineering_smoke") == 0 &&
        profile->input_side == SANITY_SIDE &&
        profile->conv_depth == 3 &&
        profile->channels[0] == 8 &&
        profile->channels[1] == 16 &&
        profile->channels[2] == 32 &&
        profile->channels[3] == 0 &&
        profile->latent_dim == 32 &&
        profile->bottleneck_side == 5 &&
        profile->expected_trainable_parameters_bn1d == UINT64_C(41572) &&
        profile->expected_forward_mac_per_sample == UINT64_C(5620896) &&
        strcmp(profile->profile_status, "primary_smoke") == 0;
}


static int sanity_source_path_is_expected(const char *path)
{
    static const char forward_prefix[] = "out272_gray/";
    static const char backward_prefix[] = "out272_gray\\";

    return path != NULL &&
        (strncmp(path, forward_prefix, sizeof(forward_prefix) - 1) == 0 ||
         strncmp(path, backward_prefix, sizeof(backward_prefix) - 1) == 0);
}


static int sanity_load_contract(
    const char *repository_root,
    SanityRun *run,
    const VAECProfile **profile
)
{
    static const char *profile_relative =
        "sakai_multires_c/configs/vae_c_profiles_v1.csv";
    static const char *event_manifest_relative = "manifest_events_272.csv";
    static const char *train_split_relative =
        "split_train_indices_seed123.csv";
    static const char *tensor_manifest_relative =
        "manifest_tensors_75_central.csv";
    char profile_path[SANITY_PATH_CAPACITY];
    char event_manifest_path[SANITY_PATH_CAPACITY];
    char train_split_path[SANITY_PATH_CAPACITY];
    char tensor_manifest_path[SANITY_PATH_CAPACITY];
    char error_message[512];
    GSTensorSpec expected_spec;
    size_t sample;
    int min_crop;
    int max_crop;

    if (repository_root == NULL || run == NULL || profile == NULL ||
        sanity_path_join(repository_root, profile_relative,
                         profile_path, sizeof(profile_path)) != 0 ||
        sanity_path_join(repository_root, event_manifest_relative,
                         event_manifest_path,
                         sizeof(event_manifest_path)) != 0 ||
        sanity_path_join(repository_root, train_split_relative,
                         train_split_path, sizeof(train_split_path)) != 0 ||
        sanity_path_join(repository_root, tensor_manifest_relative,
                         tensor_manifest_path,
                         sizeof(tensor_manifest_path)) != 0) {

        fprintf(stderr, "Errore: percorso input non valido.\n");
        return 1;
    }

    error_message[0] = '\0';
    if (vae_c_profile_catalog_load_csv(
            profile_path,
            &run->catalog,
            error_message,
            sizeof(error_message)) != VAE_C_CONFIG_OK) {

        fprintf(stderr, "Errore catalogo profili: %s\n", error_message);
        return 1;
    }
    *profile = vae_c_profile_catalog_find(&run->catalog, SANITY_PROFILE_ID);
    if (!sanity_profile_is_frozen(*profile)) {
        fprintf(stderr, "Errore: profilo mini75 smoke non congelato.\n");
        return 1;
    }

    if (gs_event_manifest_load(event_manifest_path, &run->event_manifest) != 0 ||
        run->event_manifest.count != 8583 ||
        gs_saved_split_load(
            train_split_path,
            "train",
            &run->event_manifest,
            &run->train_split) != 0 ||
        run->train_split.count != 6866 ||
        gs_tensor_spec_init(&expected_spec, SANITY_SIDE) != 0 ||
        gs_tensor_manifest_load(
            tensor_manifest_path,
            &expected_spec,
            &run->tensor_manifest) != 0 ||
        run->tensor_manifest.count != run->event_manifest.count ||
        strcmp(run->tensor_manifest.extra_column_name, "origin") != 0 ||
        gs_augment_mode_crop_range(
            GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
            &min_crop,
            &max_crop) != 0 ||
        min_crop != 0 || max_crop != 48) {

        fprintf(stderr, "Errore: manifest, split o augmentation non congelati.\n");
        return 1;
    }

    for (sample = 0; sample < SANITY_SUBSET_SIZE; sample++) {
        uint64_t frozen_index = sanity_manifest_indices[sample];
        size_t manifest_index = run->train_split.indices[sample];
        const GSEventRecord *record;
        size_t view;

        if (frozen_index > SIZE_MAX ||
            manifest_index != (size_t)frozen_index) {

            fprintf(stderr,
                    "Errore: subset train diverso alla posizione %lu.\n",
                    (unsigned long)sample);
            return 1;
        }
        record = gs_event_manifest_get(&run->event_manifest, manifest_index);
        if (record == NULL) {
            return 1;
        }
        for (view = 0; view < GS_EVENT_VIEW_COUNT; view++) {
            if (!sanity_source_path_is_expected(record->image_paths[view])) {
                fprintf(stderr,
                        "Errore: sorgente evento fuori da out272_gray.\n");
                return 1;
            }
        }
    }
    return 0;
}


static int sanity_allocate(
    const VAECProfile *profile,
    SanityRun *run
)
{
    VAECOpStatus status;

    status = vae_c_model_allocate(
        &run->model, profile, SANITY_SUBSET_SIZE
    );
    if (status == VAE_C_OP_OK) {
        status = vae_c_model_initialize_parameters_v1(
            &run->model, SANITY_INIT_SEED
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_model_workspace_allocate(
            &run->model, SANITY_SUBSET_SIZE, &run->workspace
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_adam_configure_v1(&run->optimizer);
    }
    if (status != VAE_C_OP_OK) {
        fprintf(stderr, "Errore setup modello: %s\n",
                vae_c_op_status_name(status));
        return 1;
    }
    if (run->optimizer.learning_rate != VAE_C_ADAM_LEARNING_RATE_V1 ||
        run->optimizer.learning_rate != 5.0e-4 ||
        run->workspace.input.shape.rank != 4 ||
        run->workspace.input.shape.dims[0] != SANITY_SUBSET_SIZE ||
        run->workspace.input.shape.dims[1] != SANITY_CHANNELS ||
        run->workspace.input.shape.dims[2] != SANITY_SIDE ||
        run->workspace.input.shape.dims[3] != SANITY_SIDE ||
        gs_multires_event_loader_init(
            &run->event_loader, SANITY_SIDE) != 0 ||
        gs_multires_batch_alloc(
            &run->batch,
            &run->tensor_manifest.spec,
            SANITY_SUBSET_SIZE) != 0) {

        fprintf(stderr, "Errore: modello, Adam o loader non congelati.\n");
        return 1;
    }
    return 0;
}


static int sanity_load_central_inputs(
    const char *repository_root,
    SanityRun *run
)
{
    size_t values_per_event = run->tensor_manifest.spec.values_per_event;
    size_t total_values;
    size_t sample;

    if (values_per_event != (size_t)SANITY_CHANNELS *
            SANITY_SIDE * SANITY_SIDE ||
        sanity_size_multiply(
            SANITY_SUBSET_SIZE,
            values_per_event,
            &total_values) != 0 ||
        total_values != run->workspace.input.shape.count ||
        sanity_size_multiply(
            total_values,
            sizeof(float),
            &run->central_input_bytes) != 0) {

        return 1;
    }
    run->central_inputs = (float *)malloc(run->central_input_bytes);
    if (run->central_inputs == NULL) {
        return 1;
    }

    for (sample = 0; sample < SANITY_SUBSET_SIZE; sample++) {
        size_t manifest_index = run->train_split.indices[sample];
        const GSEventRecord *event_record = gs_event_manifest_get(
            &run->event_manifest, manifest_index
        );
        const GSTensorManifestRecord *tensor_record = gs_tensor_manifest_get(
            &run->tensor_manifest, manifest_index
        );
        char tensor_path[SANITY_PATH_CAPACITY];
        float *destination =
            run->central_inputs + sample * values_per_event;
        size_t value;

        if (event_record == NULL || tensor_record == NULL ||
            event_record->event_index != tensor_record->event_index ||
            strcmp(event_record->event_id, tensor_record->event_id) != 0 ||
            tensor_record->channels != SANITY_CHANNELS ||
            tensor_record->height != SANITY_SIDE ||
            tensor_record->width != SANITY_SIDE ||
            strcmp(tensor_record->dtype, "uint8") != 0 ||
            strcmp(tensor_record->layout, "channel_major") != 0 ||
            strcmp(tensor_record->channel_order, "0.5;1.0;2.0;4.0") != 0 ||
            strcmp(tensor_record->extra_value,
                   "derived_from_224_central") != 0 ||
            sanity_path_join(
                repository_root,
                tensor_record->tensor_path,
                tensor_path,
                sizeof(tensor_path)) != 0 ||
            gs_tensor_read_float01(
                tensor_path,
                &run->tensor_manifest.spec,
                destination) != 0) {

            fprintf(stderr,
                    "Errore lettura centrale alla posizione %lu.\n",
                    (unsigned long)sample);
            return 1;
        }
        for (value = 0; value < values_per_event; value++) {
            if (!isfinite(destination[value]) ||
                destination[value] < 0.0f ||
                destination[value] > 1.0f) {

                fprintf(stderr, "Errore: tensore centrale non valido.\n");
                return 1;
            }
        }
        run->selected_event_indices[sample] = event_record->event_index;
    }
    return 0;
}


static int sanity_snapshot_parameters(SanityRun *run)
{
    size_t value_count;
    size_t bytes;
    size_t parameter_index;
    size_t offset = 0;

    if (run->model.trainable_value_count > SIZE_MAX) {
        return 1;
    }
    value_count = (size_t)run->model.trainable_value_count;
    if (value_count == 0 ||
        sanity_size_multiply(value_count, sizeof(float), &bytes) != 0) {

        return 1;
    }
    run->parameter_snapshot = (float *)malloc(bytes);
    if (run->parameter_snapshot == NULL) {
        return 1;
    }
    for (parameter_index = 0;
         parameter_index < run->model.parameter_count;
         parameter_index++) {

        const VAECParameter *parameter =
            run->model.parameters[parameter_index];
        size_t count;

        if (parameter == NULL || parameter->values == NULL) {
            return 1;
        }
        count = parameter->shape.count;
        if (offset > value_count || count > value_count - offset) {
            return 1;
        }
        memcpy(run->parameter_snapshot + offset,
               parameter->values,
               count * sizeof(float));
        offset += count;
    }
    if (offset != value_count) {
        return 1;
    }
    run->parameter_snapshot_count = value_count;
    return 0;
}


static int sanity_parameters_changed_and_finite(const SanityRun *run)
{
    size_t parameter_index;
    size_t offset = 0;
    bool changed = false;

    if (run == NULL || run->parameter_snapshot == NULL) {
        return 0;
    }
    for (parameter_index = 0;
         parameter_index < run->model.parameter_count;
         parameter_index++) {

        const VAECParameter *parameter =
            run->model.parameters[parameter_index];
        size_t value_index;

        if (parameter == NULL || parameter->values == NULL ||
            offset > run->parameter_snapshot_count ||
            parameter->shape.count >
                run->parameter_snapshot_count - offset) {

            return 0;
        }
        for (value_index = 0;
             value_index < parameter->shape.count;
             value_index++) {

            float value = parameter->values[value_index];

            if (!isfinite(value)) {
                return 0;
            }
            if (memcmp(&value,
                       &run->parameter_snapshot[offset + value_index],
                       sizeof(value)) != 0) {

                changed = true;
            }
        }
        offset += parameter->shape.count;
    }
    return offset == run->parameter_snapshot_count && changed;
}


static int sanity_loss_is_valid(const VAECVAELossStats *stats)
{
    return stats != NULL &&
        stats->batch_size == SANITY_SUBSET_SIZE &&
        stats->beta_kl == 1.0f &&
        stats->reconstruction.batch_size == SANITY_SUBSET_SIZE &&
        stats->reconstruction.values_per_sample ==
            (size_t)SANITY_CHANNELS * SANITY_SIDE * SANITY_SIDE &&
        stats->kl.batch_size == SANITY_SUBSET_SIZE &&
        stats->kl.latent_dim == 32 &&
        isfinite(stats->reconstruction.total_sum) &&
        isfinite(stats->reconstruction.mean_per_sample) &&
        isfinite(stats->reconstruction.mean_per_value) &&
        isfinite(stats->kl.total_sum) &&
        isfinite(stats->kl.mean_per_sample) &&
        isfinite(stats->kl.mean_per_dimension) &&
        isfinite(stats->total_sum) &&
        isfinite(stats->mean_per_sample);
}


static int sanity_copy_central_and_eval(
    SanityRun *run,
    VAECVAELossStats *stats
)
{
    VAECOpStatus status;

    memcpy(run->workspace.input.data,
           run->central_inputs,
           run->central_input_bytes);
    vae_c_vae_loss_stats_init(stats);
    status = vae_c_model_vae_forward(
        &run->model,
        VAE_C_BATCH_NORM_MODE_EVAL,
        VAE_C_LATENT_MODE_MEAN,
        &run->workspace,
        stats
    );
    if (status != VAE_C_OP_OK) {
        fprintf(stderr, "Errore CENTRAL TRAIN-SUBSET MEAN+EVAL: %s\n",
                vae_c_op_status_name(status));
        return 1;
    }
    if (!sanity_loss_is_valid(stats) ||
        memcmp(run->workspace.input.data,
               run->central_inputs,
               run->central_input_bytes) != 0) {

        fprintf(stderr,
                "Errore: CENTRAL TRAIN-SUBSET EVAL non finita o input mutato.\n");
        return 1;
    }
    return 0;
}


static size_t sanity_distinct_crop_count(const size_t *crop_counts)
{
    size_t crop;
    size_t distinct = 0;

    for (crop = 0; crop < SANITY_CROP_COUNT; crop++) {
        if (crop_counts[crop] != 0) {
            distinct++;
        }
    }
    return distinct;
}


static int sanity_load_perturbed_batch(
    SanityRun *run,
    GSAugmentRng *augmentation_rng,
    GSAugmentRng *replay_rng,
    size_t step,
    FILE *augmentation_file,
    size_t *crop_counts,
    size_t *noncentral_count,
    size_t *different_count
)
{
    size_t values_per_event;
    size_t sample;

    if (run == NULL || augmentation_rng == NULL || replay_rng == NULL ||
        crop_counts == NULL || noncentral_count == NULL ||
        different_count == NULL) {

        return 1;
    }
    values_per_event = run->tensor_manifest.spec.values_per_event;
    if (gs_multires_batch_load_train_mode(
            &run->event_manifest,
            &run->train_split,
            &run->event_loader,
            0,
            SANITY_SUBSET_SIZE,
            augmentation_rng,
            GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
            &run->batch) != 0 ||
        run->batch.count != SANITY_SUBSET_SIZE) {

        fprintf(stderr, "Errore: caricamento batch perturbato fallito.\n");
        return 1;
    }

    for (sample = 0; sample < SANITY_SUBSET_SIZE; sample++) {
        int expected_crop;
        int crop = run->batch.crop_lefts[sample];
        int shift;
        const float *perturbed = gs_multires_batch_item_const_ptr(
            &run->batch, sample
        );
        const float *central =
            run->central_inputs + sample * values_per_event;
        double absolute_sum = 0.0;
        double mean_absolute_difference;
        size_t value;

        if (gs_augment_random_crop_left_mode(
                replay_rng,
                GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
                &expected_crop) != 0 ||
            crop != expected_crop ||
            crop < GS_EVENT_MIN_CROP_LEFT ||
            crop > GS_EVENT_MAX_CROP_LEFT ||
            run->batch.manifest_indices[sample] !=
                (size_t)sanity_manifest_indices[sample] ||
            perturbed == NULL) {

            fprintf(stderr,
                    "Errore: crop o manifest index non riproducibile "
                    "allo step %lu, posizione %lu.\n",
                    (unsigned long)step,
                    (unsigned long)sample);
            return 1;
        }
        shift = crop - GS_SAKAI_CENTRAL_CROP_LEFT;
        if (shift < -24 || shift > 24) {
            return 1;
        }
        for (value = 0; value < values_per_event; value++) {
            double difference;

            if (!isfinite(perturbed[value]) ||
                perturbed[value] < 0.0f ||
                perturbed[value] > 1.0f) {

                fprintf(stderr, "Errore: input perturbato non valido.\n");
                return 1;
            }
            difference = fabs(
                (double)perturbed[value] - (double)central[value]
            );
            absolute_sum += difference;
        }
        mean_absolute_difference =
            absolute_sum / (double)values_per_event;
        if (!isfinite(mean_absolute_difference)) {
            return 1;
        }

        crop_counts[crop]++;
        if (crop != GS_SAKAI_CENTRAL_CROP_LEFT) {
            (*noncentral_count)++;
        }
        if (mean_absolute_difference > 0.0) {
            (*different_count)++;
        }
        if (augmentation_file != NULL &&
            (fprintf(
                augmentation_file,
                "%lu,%lu,%" PRIu64 ",%d,%d,%s,%.17g\n",
                (unsigned long)(step + 1),
                (unsigned long)sample,
                sanity_manifest_indices[sample],
                crop,
                shift,
                crop == GS_SAKAI_CENTRAL_CROP_LEFT ? "true" : "false",
                mean_absolute_difference) < 0)) {

            return 1;
        }
    }
    if (augmentation_rng->state != replay_rng->state) {
        fprintf(stderr,
                "Errore: stato RNG replay diverso dopo il batch perturbato.\n");
        return 1;
    }
    if (augmentation_file != NULL &&
        (fflush(augmentation_file) != 0 || ferror(augmentation_file))) {

        return 1;
    }
    return 0;
}


static int sanity_write_csv_text(FILE *file, const char *text)
{
    const unsigned char *cursor = (const unsigned char *)text;

    if (file == NULL || text == NULL || fputc('"', file) == EOF) {
        return 1;
    }
    while (*cursor != 0) {
        if (*cursor == '"' && fputc('"', file) == EOF) {
            return 1;
        }
        if (fputc(*cursor, file) == EOF) {
            return 1;
        }
        cursor++;
    }
    return fputc('"', file) == EOF ? 1 : 0;
}


static int sanity_write_selected_events(SanityRun *run)
{
    FILE *file = NULL;
    size_t sample;

    if (sanity_open_output(
            run->partial_output,
            "selected_events.csv",
            "wb",
            &file) != 0 ||
        fprintf(file,
                "split_position,manifest_index,event_index,event_id,"
                "central_tensor_path\n") < 0) {

        if (file != NULL) {
            (void)fclose(file);
        }
        return 1;
    }
    for (sample = 0; sample < SANITY_SUBSET_SIZE; sample++) {
        size_t manifest_index = run->train_split.indices[sample];
        const GSEventRecord *event_record = gs_event_manifest_get(
            &run->event_manifest, manifest_index
        );
        const GSTensorManifestRecord *tensor_record = gs_tensor_manifest_get(
            &run->tensor_manifest, manifest_index
        );

        if (event_record == NULL || tensor_record == NULL ||
            fprintf(file, "%lu,%" PRIu64 ",%d,",
                    (unsigned long)sample,
                    sanity_manifest_indices[sample],
                    event_record->event_index) < 0 ||
            sanity_write_csv_text(file, event_record->event_id) != 0 ||
            fputc(',', file) == EOF ||
            sanity_write_csv_text(file, tensor_record->tensor_path) != 0 ||
            fputc('\n', file) == EOF) {

            (void)fclose(file);
            return 1;
        }
    }
    return sanity_finish_file(&file);
}


static int sanity_write_run_config_start(
    SanityRun *run,
    const char *code_commit
)
{
    size_t index;

    if (sanity_open_output(
            run->partial_output,
            "run_config.txt",
            "wb",
            &run->run_config_file) != 0) {

        return 1;
    }
    if (fprintf(
            run->run_config_file,
            "profile=%s\n"
            "subset_policy=first_16_train_split_seed123\n"
            "batch_size=%d\n"
            "steps=%d\n"
            "shuffle=false\n"
            "init_seed=%" PRIu64 "\n"
            "latent_base_seed=%" PRIu64 "\n"
            "augmentation_seed=%" PRIu32 "\n"
            "augmentation_mode=symmetric_figure\n"
            "augmentation_crop_left_min=0\n"
            "augmentation_crop_left_max=48\n"
            "augmentation_shift_min=-24\n"
            "augmentation_shift_max=24\n"
            "augmentation_rng_is_sequential=true\n"
            "augmentation_order_or_batch_independent=false\n"
            "augmentation_reproducibility_scope=fixed_seed_order_call_count\n"
            "per_event_per_epoch_rng_required_before_shuffled_training=true\n"
            "central_eval_every=%d\n"
            "central_eval_name=CENTRAL TRAIN-SUBSET EVAL\n"
            "learning_rate=%.17g\n"
            "train_mode=STOCHASTIC_TRAIN\n"
            "diagnostic_mode=MEAN_EVAL\n"
            "source_manifest=manifest_events_272.csv\n"
            "source_tensor_root=out272_gray\n"
            "central_manifest=manifest_tensors_75_central.csv\n"
            "central_tensor_root=tensors_75_central_uint8\n"
            "labels_used=false\n"
            "pca=false\n"
            "checkpoint=false\n"
            "code_commit=%s\n"
            "manifest_indices=",
            SANITY_PROFILE_ID,
            SANITY_SUBSET_SIZE,
            SANITY_STEPS,
            SANITY_INIT_SEED,
            SANITY_LATENT_BASE_SEED,
            SANITY_AUGMENTATION_SEED,
            SANITY_CENTRAL_EVAL_EVERY,
            run->optimizer.learning_rate,
            code_commit) < 0) {

        return 1;
    }
    for (index = 0; index < SANITY_SUBSET_SIZE; index++) {
        if (fprintf(run->run_config_file,
                    "%s%" PRIu64,
                    index == 0 ? "" : ",",
                    sanity_manifest_indices[index]) < 0) {

            return 1;
        }
    }
    return fputc('\n', run->run_config_file) == EOF ? 1 : 0;
}


static int sanity_write_output_headers(SanityRun *run)
{
    if (sanity_open_output(
            run->partial_output,
            "metrics.csv",
            "wb",
            &run->metrics_file) != 0 ||
        fprintf(
            run->metrics_file,
            "phase,step,optimizer_step,total_sum,total_mean_per_sample,"
            "reconstruction_total_sum,reconstruction_mean_per_sample,"
            "reconstruction_mean_per_value,kl_total_sum,"
            "kl_mean_per_sample,kl_mean_per_dimension\n") < 0 ||
        sanity_open_output(
            run->partial_output,
            "augmentation.csv",
            "wb",
            &run->augmentation_file) != 0 ||
        fprintf(
            run->augmentation_file,
            "step,split_position,manifest_index,crop_left,shift,is_central,"
            "mean_abs_diff_from_central\n") < 0) {

        return 1;
    }
    return 0;
}


static int sanity_write_metric(
    SanityRun *run,
    const char *phase,
    size_t step,
    const VAECVAELossStats *stats
)
{
    if (!sanity_loss_is_valid(stats) ||
        fprintf(
            run->metrics_file,
            "%s,%lu,%" PRIu64 ",%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g\n",
            phase,
            (unsigned long)step,
            run->optimizer.step,
            stats->total_sum,
            stats->mean_per_sample,
            stats->reconstruction.total_sum,
            stats->reconstruction.mean_per_sample,
            stats->reconstruction.mean_per_value,
            stats->kl.total_sum,
            stats->kl.mean_per_sample,
            stats->kl.mean_per_dimension) < 0 ||
        fflush(run->metrics_file) != 0 ||
        ferror(run->metrics_file)) {

        return 1;
    }
    return 0;
}


static int sanity_prepare_output(
    const char *repository_root,
    const char *code_commit,
    SanityRun *run
)
{
    char results[SANITY_PATH_CAPACITY];
    char vae_c[SANITY_PATH_CAPACITY];
    bool ignored_directory_kind;

    if (sanity_path_join(repository_root, "sakai_multires_c/results",
                         results, sizeof(results)) != 0 ||
        sanity_path_join(results, "vae_c", vae_c, sizeof(vae_c)) != 0 ||
        sanity_path_join(vae_c, "sanity",
                         run->output_parent,
                         sizeof(run->output_parent)) != 0 ||
        sanity_path_join(run->output_parent, SANITY_OUTPUT_BASENAME,
                         run->final_output,
                         sizeof(run->final_output)) != 0 ||
        sanity_path_append(run->final_output, ".partial",
                           run->partial_output,
                           sizeof(run->partial_output)) != 0) {

        return 1;
    }
    if (sanity_path_state(run->final_output, &ignored_directory_kind) != 0 ||
        sanity_path_state(run->partial_output, &ignored_directory_kind) != 0) {

        fprintf(stderr, "Errore: output finale o partial gia' esistente.\n");
        return 1;
    }
    if (sanity_ensure_directory(results) != 0 ||
        sanity_ensure_directory(vae_c) != 0 ||
        sanity_ensure_directory(run->output_parent) != 0 ||
        sanity_ensure_directory(run->partial_output) != 0) {

        fprintf(stderr, "Errore: impossibile creare la directory partial.\n");
        return 1;
    }
    run->partial_created = true;
    if (sanity_write_run_config_start(run, code_commit) != 0 ||
        sanity_write_selected_events(run) != 0 ||
        sanity_write_output_headers(run) != 0) {

        return 1;
    }
    return 0;
}


static int sanity_finish_output(
    SanityRun *run,
    const VAECVAELossStats *initial_eval,
    const VAECVAELossStats *best_eval,
    const VAECVAELossStats *final_eval,
    size_t distinct_crops,
    size_t noncentral_count,
    size_t different_count
)
{
    bool ignored_directory_kind;

    if (fprintf(
            run->run_config_file,
            "completed_steps=%" PRIu64 "\n"
            "parameters_changed=true\n"
            "all_losses_finite=true\n"
            "augmentation_reproducible_for_fixed_call_sequence=true\n"
            "distinct_crops=%lu\n"
            "noncentral_examples=%lu\n"
            "examples_different_from_central=%lu\n"
            "initial_central_mean_total=%.17g\n"
            "initial_central_mean_reconstruction=%.17g\n"
            "initial_central_mean_kl=%.17g\n"
            "best_central_mean_total=%.17g\n"
            "best_central_mean_reconstruction=%.17g\n"
            "best_central_mean_kl=%.17g\n"
            "final_central_mean_total=%.17g\n"
            "final_central_mean_reconstruction=%.17g\n"
            "final_central_mean_kl=%.17g\n",
            run->optimizer.step,
            (unsigned long)distinct_crops,
            (unsigned long)noncentral_count,
            (unsigned long)different_count,
            initial_eval->mean_per_sample,
            initial_eval->reconstruction.mean_per_sample,
            initial_eval->kl.mean_per_sample,
            best_eval->mean_per_sample,
            best_eval->reconstruction.mean_per_sample,
            best_eval->kl.mean_per_sample,
            final_eval->mean_per_sample,
            final_eval->reconstruction.mean_per_sample,
            final_eval->kl.mean_per_sample) < 0 ||
        sanity_finish_file(&run->augmentation_file) != 0 ||
        sanity_finish_file(&run->metrics_file) != 0 ||
        sanity_finish_file(&run->run_config_file) != 0) {

        return 1;
    }
    if (sanity_path_state(run->final_output, &ignored_directory_kind) != 0 ||
        rename(run->partial_output, run->final_output) != 0) {

        fprintf(stderr, "Errore: finalizzazione atomica output fallita.\n");
        return 1;
    }
    run->partial_created = false;
    return 0;
}


static int sanity_dry_run(SanityRun *run)
{
    GSAugmentRng augmentation_rng;
    GSAugmentRng replay_rng;
    size_t crop_counts[SANITY_CROP_COUNT] = {0};
    size_t noncentral_count = 0;
    size_t different_count = 0;
    size_t distinct_crops;

    gs_augment_rng_seed(&augmentation_rng, SANITY_AUGMENTATION_SEED);
    gs_augment_rng_seed(&replay_rng, SANITY_AUGMENTATION_SEED);
    if (sanity_load_perturbed_batch(
            run,
            &augmentation_rng,
            &replay_rng,
            0,
            NULL,
            crop_counts,
            &noncentral_count,
            &different_count) != 0) {

        return 1;
    }
    distinct_crops = sanity_distinct_crop_count(crop_counts);
    if (noncentral_count == 0 || different_count == 0 ||
        distinct_crops < 2) {

        fprintf(stderr,
                "Errore: il dry-run non ha dimostrato perturbazioni effettive.\n");
        return 1;
    }

    printf("DRY_RUN=true\n");
    printf("AUGMENTATION_REPRODUCIBLE_FOR_FIXED_CALL_SEQUENCE=true\n");
    printf("DRY_RUN_NONCENTRAL_EXAMPLES=%lu\n",
           (unsigned long)noncentral_count);
    printf("DRY_RUN_DISTINCT_CROPS=%lu\n",
           (unsigned long)distinct_crops);
    printf("DRY_RUN_EXAMPLES_DIFFERENT_FROM_CENTRAL=%lu\n",
           (unsigned long)different_count);
    printf("TRAIN_STEPS_EXECUTED=0\n");
    printf("OUTPUT_CREATED=false\n");
    printf("PERTURBED MINI75 SANITY DRY RUN PASSED\n");
    return 0;
}


static int sanity_run(
    const char *repository_root,
    const char *code_commit,
    bool dry_run
)
{
    SanityRun run;
    const VAECProfile *profile = NULL;
    VAECVAELossStats initial_eval;
    VAECVAELossStats best_eval;
    VAECVAELossStats final_eval;
    GSAugmentRng augmentation_rng;
    GSAugmentRng replay_rng;
    size_t crop_counts[SANITY_CROP_COUNT] = {0};
    size_t noncentral_count = 0;
    size_t different_count = 0;
    size_t step;
    int result = 1;

    sanity_run_init(&run);
    vae_c_vae_loss_stats_init(&initial_eval);
    vae_c_vae_loss_stats_init(&best_eval);
    vae_c_vae_loss_stats_init(&final_eval);

    if (sanity_load_contract(repository_root, &run, &profile) != 0 ||
        sanity_allocate(profile, &run) != 0 ||
        sanity_load_central_inputs(repository_root, &run) != 0 ||
        sanity_copy_central_and_eval(&run, &initial_eval) != 0) {

        goto cleanup;
    }
    best_eval = initial_eval;
    final_eval = initial_eval;

    printf("PROFILE=%s\n", SANITY_PROFILE_ID);
    printf("SUBSET_SIZE=%d\n", SANITY_SUBSET_SIZE);
    printf("BATCH_SIZE=%d\n", SANITY_SUBSET_SIZE);
    printf("STEPS=%d\n", SANITY_STEPS);
    printf("CODE_COMMIT=%s\n", code_commit);
    printf("AUGMENTATION_MODE=symmetric_figure\n");
    printf("AUGMENTATION_SEED=%" PRIu32 "\n",
           SANITY_AUGMENTATION_SEED);
    printf("AUGMENTATION_RNG_IS_SEQUENTIAL=true\n");
    printf("AUGMENTATION_ORDER_OR_BATCH_INDEPENDENT=false\n");
    printf("CENTRAL_EVAL_NAME=CENTRAL TRAIN-SUBSET EVAL\n");
    printf("INITIAL_CENTRAL_MEAN_TOTAL=%.17g\n",
           initial_eval.mean_per_sample);
    printf("INITIAL_CENTRAL_MEAN_RECONSTRUCTION=%.17g\n",
           initial_eval.reconstruction.mean_per_sample);
    printf("INITIAL_CENTRAL_MEAN_KL=%.17g\n",
           initial_eval.kl.mean_per_sample);

    if (dry_run) {
        if (sanity_dry_run(&run) == 0) {
            result = 0;
        }
        goto cleanup;
    }

    if (sanity_snapshot_parameters(&run) != 0 ||
        sanity_prepare_output(repository_root, code_commit, &run) != 0 ||
        sanity_write_metric(
            &run,
            "central_train_subset_mean_eval_initial",
            0,
            &initial_eval) != 0) {

        goto cleanup;
    }

    gs_augment_rng_seed(&augmentation_rng, SANITY_AUGMENTATION_SEED);
    gs_augment_rng_seed(&replay_rng, SANITY_AUGMENTATION_SEED);
    for (step = 0; step < SANITY_STEPS; step++) {
        VAECVAELossStats train_stats;
        VAECOpStatus status;
        uint64_t expected_optimizer_step = (uint64_t)step + UINT64_C(1);

        if (sanity_load_perturbed_batch(
                &run,
                &augmentation_rng,
                &replay_rng,
                step,
                run.augmentation_file,
                crop_counts,
                &noncentral_count,
                &different_count) != 0) {

            goto cleanup;
        }
        memcpy(run.workspace.input.data,
               run.batch.x,
               run.central_input_bytes);

        vae_c_vae_loss_stats_init(&train_stats);
        status = vae_c_model_train_step(
            &run.model,
            &run.workspace,
            &run.optimizer,
            SANITY_LATENT_BASE_SEED,
            (uint64_t)step,
            sanity_manifest_indices,
            SANITY_SUBSET_SIZE,
            &train_stats
        );
        if (status != VAE_C_OP_OK ||
            run.optimizer.step != expected_optimizer_step ||
            !sanity_loss_is_valid(&train_stats) ||
            memcmp(run.workspace.input.data,
                   run.batch.x,
                   run.central_input_bytes) != 0 ||
            sanity_write_metric(
                &run,
                "train_stochastic_pre_update",
                step + 1,
                &train_stats) != 0) {

            fprintf(stderr, "Errore train step %lu: %s\n",
                    (unsigned long)(step + 1),
                    vae_c_op_status_name(status));
            goto cleanup;
        }

        if (expected_optimizer_step % SANITY_CENTRAL_EVAL_EVERY == 0) {
            VAECVAELossStats evaluation;

            vae_c_vae_loss_stats_init(&evaluation);
            if (sanity_copy_central_and_eval(&run, &evaluation) != 0 ||
                sanity_write_metric(
                    &run,
                    "central_train_subset_mean_eval_post_update",
                    step + 1,
                    &evaluation) != 0) {

                goto cleanup;
            }
            if (evaluation.mean_per_sample < best_eval.mean_per_sample) {
                best_eval = evaluation;
            }
            final_eval = evaluation;
        }
    }

    {
        size_t distinct_crops = sanity_distinct_crop_count(crop_counts);

        if (run.optimizer.step != SANITY_STEPS ||
            noncentral_count == 0 || different_count == 0 ||
            distinct_crops < 2 ||
            !sanity_parameters_changed_and_finite(&run) ||
            sanity_finish_output(
                &run,
                &initial_eval,
                &best_eval,
                &final_eval,
                distinct_crops,
                noncentral_count,
                different_count) != 0) {

            fprintf(stderr, "Errore: contratto finale sanity non rispettato.\n");
            goto cleanup;
        }

        printf("TRAIN_STEPS_EXECUTED=%" PRIu64 "\n", run.optimizer.step);
        printf("PARAMETERS_CHANGED=true\n");
        printf("ALL_LOSSES_FINITE=true\n");
        printf("AUGMENTATION_REPRODUCIBLE_FOR_FIXED_CALL_SEQUENCE=true\n");
        printf("DISTINCT_CROPS=%lu\n", (unsigned long)distinct_crops);
        printf("NONCENTRAL_EXAMPLES=%lu\n",
               (unsigned long)noncentral_count);
        printf("EXAMPLES_DIFFERENT_FROM_CENTRAL=%lu\n",
               (unsigned long)different_count);
        printf("FINAL_CENTRAL_MEAN_TOTAL=%.17g\n",
               final_eval.mean_per_sample);
        printf("OUTPUT_DIRECTORY=%s\n", run.final_output);
        printf("PERTURBED MINI75 SANITY RUN COMPLETED\n");
    }
    result = 0;

cleanup:
    sanity_run_free(&run);
    return result;
}


static void sanity_print_usage(const char *program)
{
    printf(
        "Uso: %s [--dry-run] --code-commit HASH [repository_root]\n"
        "\n"
        "HASH deve contenere esattamente 40 caratteri esadecimali.\n"
        "Senza --dry-run esegue 80 step mini75 subset16 con "
        "symmetric_figure e output atomico sotto results/vae_c/sanity.\n"
        "--dry-run valida profilo, subset, centrali e un batch perturbato; "
        "esegue zero train step e non crea output.\n"
        "L'RNG augmentation e' sequenziale: la riproducibilita' richiede "
        "identico seed, ordine e numero di chiamate.\n",
        program
    );
}


int main(int argc, char **argv)
{
    const char *repository_root = ".";
    const char *code_commit = NULL;
    bool dry_run = false;
    bool repository_root_seen = false;
    int argument;

    for (argument = 1; argument < argc; argument++) {
        if (strcmp(argv[argument], "--help") == 0 ||
            strcmp(argv[argument], "-h") == 0) {

            sanity_print_usage(argv[0]);
            return 0;
        }
    }
    for (argument = 1; argument < argc; argument++) {
        if (strcmp(argv[argument], "--dry-run") == 0) {
            if (dry_run) {
                sanity_print_usage(argv[0]);
                return 2;
            }
            dry_run = true;
        } else if (strcmp(argv[argument], "--code-commit") == 0) {
            if (code_commit != NULL || argument + 1 >= argc) {
                sanity_print_usage(argv[0]);
                return 2;
            }
            code_commit = argv[++argument];
        } else if (argv[argument][0] == '-' || repository_root_seen) {
            sanity_print_usage(argv[0]);
            return 2;
        } else {
            repository_root = argv[argument];
            repository_root_seen = true;
        }
    }
    if (!sanity_code_commit_is_valid(code_commit)) {
        fprintf(stderr,
                "Errore: --code-commit richiede esattamente "
                "40 caratteri esadecimali.\n");
        sanity_print_usage(argv[0]);
        return 2;
    }

    return sanity_run(repository_root, code_commit, dry_run);
}
