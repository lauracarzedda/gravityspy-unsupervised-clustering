#include "vae_c_adam.h"
#include "vae_c_config.h"
#include "vae_c_model_forward.h"
#include "vae_c_model_train.h"
#include "vae_event_dataset.h"
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
    OVERFIT_PATH_CAPACITY = 4096,
    OVERFIT_SUBSET_SIZE = 16,
    OVERFIT_CHANNELS = 4,
    OVERFIT_SIDE = 75,
    OVERFIT_EPOCHS = 200,
    OVERFIT_MEAN_EVAL_EVERY = 10,
    OVERFIT_PREVIEW_EVENTS = 2
};

#define OVERFIT_PROFILE_ID "mini75_smoke_v1"
#define OVERFIT_OUTPUT_BASENAME \
    "mini75_smoke_v1_subset16_init123_latent456"
#define OVERFIT_INIT_SEED UINT64_C(123)
#define OVERFIT_LATENT_BASE_SEED UINT64_C(456)


static const uint64_t overfit_manifest_indices[OVERFIT_SUBSET_SIZE] = {
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
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECAdam optimizer;

    float *input_snapshot;
    float *parameter_snapshot;
    size_t parameter_snapshot_count;

    int selected_event_indices[OVERFIT_SUBSET_SIZE];

    FILE *run_config_file;
    FILE *metrics_file;

    char output_parent[OVERFIT_PATH_CAPACITY];
    char final_output[OVERFIT_PATH_CAPACITY];
    char partial_output[OVERFIT_PATH_CAPACITY];
    bool partial_created;
} OverfitRun;


static int overfit_size_multiply(
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


static int overfit_path_join(
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


static int overfit_path_append(
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


static int overfit_code_commit_is_valid(const char *code_commit)
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


static int overfit_path_state(const char *path, bool *is_directory)
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


static int overfit_ensure_directory(const char *path)
{
    bool is_directory = false;
    int state = overfit_path_state(path, &is_directory);

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


static int overfit_remove_directory(const char *path)
{
#ifdef _WIN32
    return _rmdir(path);
#else
    return rmdir(path);
#endif
}


static int overfit_finish_file(FILE **file)
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


static int overfit_open_output(
    const char *directory,
    const char *name,
    const char *mode,
    FILE **file
)
{
    char path[OVERFIT_PATH_CAPACITY];

    if (file == NULL) {
        return 1;
    }
    *file = NULL;
    if (overfit_path_join(directory, name, path, sizeof(path)) != 0) {
        return 1;
    }
    *file = fopen(path, mode);
    return *file == NULL ? 1 : 0;
}


static int overfit_preview_name(
    const char *prefix,
    int event_index,
    size_t channel,
    char *name,
    size_t capacity
)
{
    int written;

    if (prefix == NULL || name == NULL || capacity == 0 ||
        event_index < 1 || channel >= OVERFIT_CHANNELS) {

        return 1;
    }
    written = snprintf(
        name,
        capacity,
        "%s_event_%06d_ch%lu.pgm",
        prefix,
        event_index,
        (unsigned long)channel
    );
    return written < 0 || (size_t)written >= capacity ? 1 : 0;
}


static void overfit_remove_partial_outputs(OverfitRun *run)
{
    static const char *preview_prefixes[] = {
        "target_initial",
        "reconstruction_mean_initial",
        "reconstruction_mean_final"
    };
    static const char *text_files[] = {
        "run_config.txt",
        "selected_events.csv",
        "metrics.csv"
    };
    size_t index;
    size_t sample;
    size_t channel;

    if (run == NULL || !run->partial_created) {
        return;
    }
    for (index = 0; index < sizeof(text_files) / sizeof(text_files[0]); index++) {
        char path[OVERFIT_PATH_CAPACITY];

        if (overfit_path_join(
                run->partial_output,
                text_files[index],
                path,
                sizeof(path)) == 0) {

            (void)remove(path);
        }
    }
    for (index = 0;
         index < sizeof(preview_prefixes) / sizeof(preview_prefixes[0]);
         index++) {

        for (sample = 0; sample < OVERFIT_PREVIEW_EVENTS; sample++) {
            for (channel = 0; channel < OVERFIT_CHANNELS; channel++) {
                char name[128];
                char path[OVERFIT_PATH_CAPACITY];

                if (overfit_preview_name(
                        preview_prefixes[index],
                        run->selected_event_indices[sample],
                        channel,
                        name,
                        sizeof(name)) == 0 &&
                    overfit_path_join(
                        run->partial_output,
                        name,
                        path,
                        sizeof(path)) == 0) {

                    (void)remove(path);
                }
            }
        }
    }
    (void)overfit_remove_directory(run->partial_output);
    run->partial_created = false;
}


static void overfit_run_init(OverfitRun *run)
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


static void overfit_run_free(OverfitRun *run)
{
    if (run == NULL) {
        return;
    }
    if (run->metrics_file != NULL) {
        (void)fclose(run->metrics_file);
        run->metrics_file = NULL;
    }
    if (run->run_config_file != NULL) {
        (void)fclose(run->run_config_file);
        run->run_config_file = NULL;
    }
    overfit_remove_partial_outputs(run);
    free(run->parameter_snapshot);
    free(run->input_snapshot);
    vae_c_adam_free(&run->optimizer);
    vae_c_model_workspace_free(&run->workspace);
    vae_c_model_free(&run->model);
    gs_tensor_manifest_free(&run->tensor_manifest);
    gs_saved_split_free(&run->train_split);
    gs_event_manifest_free(&run->event_manifest);
    vae_c_profile_catalog_free(&run->catalog);
    memset(run, 0, sizeof(*run));
}


static int overfit_profile_is_frozen(const VAECProfile *profile)
{
    return profile != NULL &&
        strcmp(profile->profile_id, OVERFIT_PROFILE_ID) == 0 &&
        strcmp(profile->role, "engineering_smoke") == 0 &&
        profile->input_side == OVERFIT_SIDE &&
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


static int overfit_load_contract(
    const char *repository_root,
    OverfitRun *run,
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
    char profile_path[OVERFIT_PATH_CAPACITY];
    char event_manifest_path[OVERFIT_PATH_CAPACITY];
    char train_split_path[OVERFIT_PATH_CAPACITY];
    char tensor_manifest_path[OVERFIT_PATH_CAPACITY];
    char error_message[512];
    GSTensorSpec expected_spec;

    if (repository_root == NULL || run == NULL || profile == NULL ||
        overfit_path_join(repository_root, profile_relative,
                          profile_path, sizeof(profile_path)) != 0 ||
        overfit_path_join(repository_root, event_manifest_relative,
                          event_manifest_path,
                          sizeof(event_manifest_path)) != 0 ||
        overfit_path_join(repository_root, train_split_relative,
                          train_split_path, sizeof(train_split_path)) != 0 ||
        overfit_path_join(repository_root, tensor_manifest_relative,
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
    *profile = vae_c_profile_catalog_find(&run->catalog, OVERFIT_PROFILE_ID);
    if (!overfit_profile_is_frozen(*profile)) {
        fprintf(stderr, "Errore: profilo mini75 smoke diverso dal contratto congelato.\n");
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
        gs_tensor_spec_init(&expected_spec, OVERFIT_SIDE) != 0 ||
        gs_tensor_manifest_load(
            tensor_manifest_path,
            &expected_spec,
            &run->tensor_manifest) != 0 ||
        run->tensor_manifest.count != run->event_manifest.count ||
        strcmp(run->tensor_manifest.extra_column_name, "origin") != 0) {

        fprintf(stderr, "Errore: manifest o split non rispettano il contratto.\n");
        return 1;
    }
    return 0;
}


static int overfit_allocate_model(
    const VAECProfile *profile,
    OverfitRun *run
)
{
    VAECOpStatus status;

    status = vae_c_model_allocate(
        &run->model, profile, OVERFIT_SUBSET_SIZE
    );
    if (status == VAE_C_OP_OK) {
        status = vae_c_model_initialize_parameters_v1(
            &run->model, OVERFIT_INIT_SEED
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_model_workspace_allocate(
            &run->model, OVERFIT_SUBSET_SIZE, &run->workspace
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
        run->workspace.input.shape.dims[0] != OVERFIT_SUBSET_SIZE ||
        run->workspace.input.shape.dims[1] != OVERFIT_CHANNELS ||
        run->workspace.input.shape.dims[2] != OVERFIT_SIDE ||
        run->workspace.input.shape.dims[3] != OVERFIT_SIDE) {

        fprintf(stderr, "Errore: modello, Adam o workspace non congelati.\n");
        return 1;
    }
    return 0;
}


static int overfit_load_inputs(
    const char *repository_root,
    OverfitRun *run
)
{
    size_t values_per_event = run->tensor_manifest.spec.values_per_event;
    size_t input_bytes;
    size_t sample;

    if (values_per_event != (size_t)OVERFIT_CHANNELS *
            OVERFIT_SIDE * OVERFIT_SIDE ||
        overfit_size_multiply(
            run->workspace.input.shape.count,
            sizeof(float),
            &input_bytes) != 0) {

        return 1;
    }

    for (sample = 0; sample < OVERFIT_SUBSET_SIZE; sample++) {
        uint64_t frozen_index = overfit_manifest_indices[sample];
        size_t manifest_index = run->train_split.indices[sample];
        const GSEventRecord *event_record;
        const GSTensorManifestRecord *tensor_record;
        char tensor_path[OVERFIT_PATH_CAPACITY];
        float *destination;

        if (frozen_index > SIZE_MAX ||
            manifest_index != (size_t)frozen_index) {

            fprintf(stderr, "Errore: subset train diverso alla posizione %lu.\n",
                    (unsigned long)sample);
            return 1;
        }
        event_record = gs_event_manifest_get(
            &run->event_manifest, manifest_index
        );
        tensor_record = gs_tensor_manifest_get(
            &run->tensor_manifest, manifest_index
        );
        if (event_record == NULL || tensor_record == NULL ||
            event_record->event_index != tensor_record->event_index ||
            strcmp(event_record->event_id, tensor_record->event_id) != 0 ||
            tensor_record->channels != OVERFIT_CHANNELS ||
            tensor_record->height != OVERFIT_SIDE ||
            tensor_record->width != OVERFIT_SIDE ||
            strcmp(tensor_record->dtype, "uint8") != 0 ||
            strcmp(tensor_record->layout, "channel_major") != 0 ||
            strcmp(tensor_record->channel_order, "0.5;1.0;2.0;4.0") != 0 ||
            strcmp(tensor_record->extra_value,
                   "derived_from_224_central") != 0 ||
            overfit_path_join(
                repository_root,
                tensor_record->tensor_path,
                tensor_path,
                sizeof(tensor_path)) != 0) {

            fprintf(stderr, "Errore: record centrale incoerente alla posizione %lu.\n",
                    (unsigned long)sample);
            return 1;
        }
        destination = run->workspace.input.data + sample * values_per_event;
        if (gs_tensor_read_float01(
                tensor_path,
                &run->tensor_manifest.spec,
                destination) != 0) {

            fprintf(stderr, "Errore lettura tensore centrale: %s\n", tensor_path);
            return 1;
        }
        run->selected_event_indices[sample] = event_record->event_index;
    }

    run->input_snapshot = (float *)malloc(input_bytes);
    if (run->input_snapshot == NULL) {
        return 1;
    }
    memcpy(run->input_snapshot, run->workspace.input.data, input_bytes);
    return 0;
}


static int overfit_snapshot_parameters(OverfitRun *run)
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
        overfit_size_multiply(value_count, sizeof(float), &bytes) != 0) {

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


static int overfit_input_is_unchanged(const OverfitRun *run)
{
    size_t bytes;

    return run != NULL && run->input_snapshot != NULL &&
        overfit_size_multiply(
            run->workspace.input.shape.count,
            sizeof(float),
            &bytes) == 0 &&
        memcmp(run->input_snapshot, run->workspace.input.data, bytes) == 0;
}


static int overfit_parameters_changed_and_finite(const OverfitRun *run)
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


static int overfit_loss_is_valid(const VAECVAELossStats *stats)
{
    return stats != NULL &&
        stats->batch_size == OVERFIT_SUBSET_SIZE &&
        stats->beta_kl == 1.0f &&
        stats->reconstruction.batch_size == OVERFIT_SUBSET_SIZE &&
        stats->reconstruction.values_per_sample ==
            (size_t)OVERFIT_CHANNELS * OVERFIT_SIDE * OVERFIT_SIDE &&
        stats->kl.batch_size == OVERFIT_SUBSET_SIZE &&
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


static int overfit_mean_eval(
    OverfitRun *run,
    VAECVAELossStats *stats
)
{
    VAECOpStatus status;

    vae_c_vae_loss_stats_init(stats);
    status = vae_c_model_vae_forward(
        &run->model,
        VAE_C_BATCH_NORM_MODE_EVAL,
        VAE_C_LATENT_MODE_MEAN,
        &run->workspace,
        stats
    );
    if (status != VAE_C_OP_OK) {
        fprintf(stderr, "Errore MEAN+EVAL: %s\n",
                vae_c_op_status_name(status));
        return 1;
    }
    if (!overfit_loss_is_valid(stats) ||
        !overfit_input_is_unchanged(run)) {

        fprintf(stderr, "Errore: output MEAN+EVAL non finito o input mutato.\n");
        return 1;
    }
    return 0;
}


static float overfit_sigmoid(float logit)
{
    if (logit >= 0.0f) {
        float exponential = expf(-logit);
        return 1.0f / (1.0f + exponential);
    }
    {
        float exponential = expf(logit);
        return exponential / (1.0f + exponential);
    }
}


static unsigned char overfit_probability_to_u8(float value)
{
    float scaled;

    if (value <= 0.0f) {
        return 0;
    }
    if (value >= 1.0f) {
        return 255;
    }
    scaled = value * 255.0f + 0.5f;
    return (unsigned char)scaled;
}


static int overfit_write_pgm(
    const char *directory,
    const char *prefix,
    int event_index,
    size_t channel,
    const float *values,
    bool values_are_logits
)
{
    unsigned char pixels[OVERFIT_SIDE * OVERFIT_SIDE];
    char name[128];
    FILE *file = NULL;
    size_t index;

    if (values == NULL ||
        overfit_preview_name(
            prefix, event_index, channel, name, sizeof(name)) != 0 ||
        overfit_open_output(directory, name, "wb", &file) != 0) {

        return 1;
    }
    for (index = 0; index < OVERFIT_SIDE * OVERFIT_SIDE; index++) {
        float value = values[index];

        if (!isfinite(value)) {
            (void)fclose(file);
            return 1;
        }
        if (values_are_logits) {
            value = overfit_sigmoid(value);
        }
        pixels[index] = overfit_probability_to_u8(value);
    }
    if (fprintf(file, "P5\n%d %d\n255\n", OVERFIT_SIDE, OVERFIT_SIDE) < 0 ||
        fwrite(pixels, 1, sizeof(pixels), file) != sizeof(pixels) ||
        overfit_finish_file(&file) != 0) {

        if (file != NULL) {
            (void)fclose(file);
        }
        return 1;
    }
    return 0;
}


static int overfit_write_preview_set(
    const OverfitRun *run,
    const char *prefix,
    const float *batch_values,
    bool values_are_logits
)
{
    size_t sample;
    size_t channel;
    size_t values_per_channel = OVERFIT_SIDE * OVERFIT_SIDE;
    size_t values_per_event = OVERFIT_CHANNELS * values_per_channel;

    for (sample = 0; sample < OVERFIT_PREVIEW_EVENTS; sample++) {
        for (channel = 0; channel < OVERFIT_CHANNELS; channel++) {
            const float *values = batch_values +
                sample * values_per_event + channel * values_per_channel;

            if (overfit_write_pgm(
                    run->partial_output,
                    prefix,
                    run->selected_event_indices[sample],
                    channel,
                    values,
                    values_are_logits) != 0) {

                return 1;
            }
        }
    }
    return 0;
}


static int overfit_write_csv_text(FILE *file, const char *text)
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


static int overfit_write_selected_events(OverfitRun *run)
{
    FILE *file = NULL;
    size_t sample;

    if (overfit_open_output(
            run->partial_output,
            "selected_events.csv",
            "wb",
            &file) != 0 ||
        fprintf(file,
                "split_pos,manifest_index,event_index,event_id,tensor_path\n") < 0) {

        if (file != NULL) {
            (void)fclose(file);
        }
        return 1;
    }
    for (sample = 0; sample < OVERFIT_SUBSET_SIZE; sample++) {
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
                    overfit_manifest_indices[sample],
                    event_record->event_index) < 0 ||
            overfit_write_csv_text(file, event_record->event_id) != 0 ||
            fputc(',', file) == EOF ||
            overfit_write_csv_text(file, tensor_record->tensor_path) != 0 ||
            fputc('\n', file) == EOF) {

            (void)fclose(file);
            return 1;
        }
    }
    return overfit_finish_file(&file);
}


static int overfit_write_run_config_start(
    OverfitRun *run,
    const char *code_commit
)
{
    size_t index;

    if (overfit_open_output(
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
            "init_seed=%" PRIu64 "\n"
            "latent_base_seed=%" PRIu64 "\n"
            "epochs=%d\n"
            "mean_eval_every=%d\n"
            "learning_rate=%.17g\n"
            "train_mode=STOCHASTIC_TRAIN\n"
            "diagnostic_mode=MEAN_EVAL\n"
            "input_manifest=manifest_tensors_75_central.csv\n"
            "input_tensor_root=tensors_75_central_uint8\n"
            "augmentation=false\n"
            "dynamic_crop=false\n"
            "labels_used=false\n"
            "checkpoint=false\n"
            "code_commit=%s\n"
            "manifest_indices=",
            OVERFIT_PROFILE_ID,
            OVERFIT_SUBSET_SIZE,
            OVERFIT_INIT_SEED,
            OVERFIT_LATENT_BASE_SEED,
            OVERFIT_EPOCHS,
            OVERFIT_MEAN_EVAL_EVERY,
            run->optimizer.learning_rate,
            code_commit) < 0) {

        return 1;
    }
    for (index = 0; index < OVERFIT_SUBSET_SIZE; index++) {
        if (fprintf(run->run_config_file,
                    "%s%" PRIu64,
                    index == 0 ? "" : ",",
                    overfit_manifest_indices[index]) < 0) {

            return 1;
        }
    }
    return fputc('\n', run->run_config_file) == EOF ? 1 : 0;
}


static int overfit_write_metrics_header(OverfitRun *run)
{
    if (overfit_open_output(
            run->partial_output,
            "metrics.csv",
            "wb",
            &run->metrics_file) != 0) {

        return 1;
    }
    return fprintf(
        run->metrics_file,
        "phase,epoch,optimizer_step,total_sum,total_mean_per_sample,"
        "reconstruction_total_sum,reconstruction_mean_per_sample,"
        "reconstruction_mean_per_value,kl_total_sum,"
        "kl_mean_per_sample,kl_mean_per_dimension\n"
    ) < 0 ? 1 : 0;
}


static int overfit_write_metric(
    OverfitRun *run,
    const char *phase,
    int epoch,
    const VAECVAELossStats *stats
)
{
    if (!overfit_loss_is_valid(stats) ||
        fprintf(
            run->metrics_file,
            "%s,%d,%" PRIu64 ",%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g\n",
            phase,
            epoch,
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


static int overfit_prepare_output(
    const char *repository_root,
    const char *code_commit,
    OverfitRun *run
)
{
    char results[OVERFIT_PATH_CAPACITY];
    char vae_c[OVERFIT_PATH_CAPACITY];
    bool ignored_directory_kind;

    if (overfit_path_join(repository_root, "sakai_multires_c/results",
                          results, sizeof(results)) != 0 ||
        overfit_path_join(results, "vae_c", vae_c, sizeof(vae_c)) != 0 ||
        overfit_path_join(vae_c, "overfit",
                          run->output_parent,
                          sizeof(run->output_parent)) != 0 ||
        overfit_path_join(run->output_parent, OVERFIT_OUTPUT_BASENAME,
                          run->final_output,
                          sizeof(run->final_output)) != 0 ||
        overfit_path_append(run->final_output, ".partial",
                            run->partial_output,
                            sizeof(run->partial_output)) != 0) {

        return 1;
    }
    if (overfit_path_state(run->final_output, &ignored_directory_kind) != 0 ||
        overfit_path_state(run->partial_output, &ignored_directory_kind) != 0) {

        fprintf(stderr, "Errore: output finale o partial gia' esistente.\n");
        return 1;
    }
    if (overfit_ensure_directory(results) != 0 ||
        overfit_ensure_directory(vae_c) != 0 ||
        overfit_ensure_directory(run->output_parent) != 0 ||
        overfit_ensure_directory(run->partial_output) != 0) {

        fprintf(stderr, "Errore: impossibile creare la directory partial.\n");
        return 1;
    }
    run->partial_created = true;
    if (overfit_write_run_config_start(run, code_commit) != 0 ||
        overfit_write_selected_events(run) != 0 ||
        overfit_write_metrics_header(run) != 0) {

        return 1;
    }
    return 0;
}


static int overfit_finish_output(
    OverfitRun *run,
    const VAECVAELossStats *initial_eval,
    const VAECVAELossStats *final_eval
)
{
    bool ignored_directory_kind;

    if (fprintf(
            run->run_config_file,
            "completed_steps=%" PRIu64 "\n"
            "parameters_changed=true\n"
            "input_unchanged=true\n"
            "all_losses_finite=true\n"
            "initial_mean_total=%.17g\n"
            "initial_mean_reconstruction=%.17g\n"
            "initial_mean_kl=%.17g\n"
            "final_mean_total=%.17g\n"
            "final_mean_reconstruction=%.17g\n"
            "final_mean_kl=%.17g\n",
            run->optimizer.step,
            initial_eval->mean_per_sample,
            initial_eval->reconstruction.mean_per_sample,
            initial_eval->kl.mean_per_sample,
            final_eval->mean_per_sample,
            final_eval->reconstruction.mean_per_sample,
            final_eval->kl.mean_per_sample) < 0 ||
        overfit_finish_file(&run->metrics_file) != 0 ||
        overfit_finish_file(&run->run_config_file) != 0) {

        return 1;
    }

    if (overfit_path_state(run->final_output, &ignored_directory_kind) != 0 ||
        rename(run->partial_output, run->final_output) != 0) {

        fprintf(stderr, "Errore: finalizzazione atomica output fallita.\n");
        return 1;
    }
    run->partial_created = false;
    return 0;
}


static int overfit_run(
    const char *repository_root,
    const char *code_commit,
    bool dry_run
)
{
    OverfitRun run;
    const VAECProfile *profile = NULL;
    VAECVAELossStats initial_eval;
    VAECVAELossStats final_eval;
    int result = 1;
    size_t epoch;

    overfit_run_init(&run);
    vae_c_vae_loss_stats_init(&initial_eval);
    vae_c_vae_loss_stats_init(&final_eval);

    if (overfit_load_contract(repository_root, &run, &profile) != 0 ||
        overfit_allocate_model(profile, &run) != 0 ||
        overfit_load_inputs(repository_root, &run) != 0 ||
        overfit_snapshot_parameters(&run) != 0 ||
        overfit_mean_eval(&run, &initial_eval) != 0) {

        goto cleanup;
    }

    printf("PROFILE=%s\n", OVERFIT_PROFILE_ID);
    printf("SUBSET_SIZE=%d\n", OVERFIT_SUBSET_SIZE);
    printf("BATCH_SIZE=%d\n", OVERFIT_SUBSET_SIZE);
    printf("CODE_COMMIT=%s\n", code_commit);
    printf("INPUT_CONTRACT=central_uint8_to_float01\n");
    printf("INITIAL_MEAN_TOTAL=%.17g\n", initial_eval.mean_per_sample);
    printf("INITIAL_MEAN_RECONSTRUCTION=%.17g\n",
           initial_eval.reconstruction.mean_per_sample);
    printf("INITIAL_MEAN_KL=%.17g\n", initial_eval.kl.mean_per_sample);

    if (dry_run) {
        printf("DRY_RUN=true\n");
        printf("TRAIN_STEPS_EXECUTED=0\n");
        printf("OUTPUT_CREATED=false\n");
        printf("OVERFIT MINI75 DRY RUN PASSED\n");
        result = 0;
        goto cleanup;
    }

    if (overfit_prepare_output(repository_root, code_commit, &run) != 0 ||
        overfit_write_metric(
            &run, "mean_eval_initial", -1, &initial_eval) != 0 ||
        overfit_write_preview_set(
            &run,
            "target_initial",
            run.workspace.input.data,
            false) != 0 ||
        overfit_write_preview_set(
            &run,
            "reconstruction_mean_initial",
            run.workspace.logits.data,
            true) != 0) {

        goto cleanup;
    }

    for (epoch = 0; epoch < OVERFIT_EPOCHS; epoch++) {
        VAECVAELossStats train_stats;
        VAECOpStatus status;
        uint64_t expected_step = (uint64_t)epoch + UINT64_C(1);

        vae_c_vae_loss_stats_init(&train_stats);
        status = vae_c_model_train_step(
            &run.model,
            &run.workspace,
            &run.optimizer,
            OVERFIT_LATENT_BASE_SEED,
            (uint64_t)epoch,
            overfit_manifest_indices,
            OVERFIT_SUBSET_SIZE,
            &train_stats
        );
        if (status != VAE_C_OP_OK ||
            run.optimizer.step != expected_step ||
            !overfit_loss_is_valid(&train_stats) ||
            !overfit_input_is_unchanged(&run) ||
            overfit_write_metric(
                &run, "train_pre_update", (int)epoch, &train_stats) != 0) {

            fprintf(stderr, "Errore train epoch %lu: %s\n",
                    (unsigned long)epoch,
                    vae_c_op_status_name(status));
            goto cleanup;
        }

        if (expected_step % OVERFIT_MEAN_EVAL_EVERY == 0) {
            VAECVAELossStats evaluation;

            vae_c_vae_loss_stats_init(&evaluation);
            if (overfit_mean_eval(&run, &evaluation) != 0 ||
                overfit_write_metric(
                    &run,
                    "mean_eval_post_update",
                    (int)epoch,
                    &evaluation) != 0) {

                goto cleanup;
            }
            final_eval = evaluation;
        }
    }

    if (run.optimizer.step != OVERFIT_EPOCHS ||
        !overfit_input_is_unchanged(&run) ||
        !overfit_parameters_changed_and_finite(&run) ||
        !overfit_loss_is_valid(&final_eval) ||
        vae_c_model_validate(&run.model) != VAE_C_OP_OK ||
        overfit_write_preview_set(
            &run,
            "reconstruction_mean_final",
            run.workspace.logits.data,
            true) != 0 ||
        overfit_finish_output(&run, &initial_eval, &final_eval) != 0) {

        fprintf(stderr, "Errore: validazione finale overfit fallita.\n");
        goto cleanup;
    }

    printf("COMPLETED_STEPS=%" PRIu64 "\n", run.optimizer.step);
    printf("PARAMETERS_CHANGED=true\n");
    printf("INPUT_UNCHANGED=true\n");
    printf("ALL_LOSSES_FINITE=true\n");
    printf("FINAL_MEAN_TOTAL=%.17g\n", final_eval.mean_per_sample);
    printf("FINAL_MEAN_RECONSTRUCTION=%.17g\n",
           final_eval.reconstruction.mean_per_sample);
    printf("FINAL_MEAN_KL=%.17g\n", final_eval.kl.mean_per_sample);
    printf("OUTPUT_DIRECTORY=%s\n", run.final_output);
    printf("OVERFIT MINI75 RUN COMPLETED\n");
    result = 0;

cleanup:
    overfit_run_free(&run);
    return result;
}


static void overfit_print_usage(const char *program)
{
    printf(
        "Uso: %s [--dry-run] --code-commit HASH [repository_root]\n"
        "\n"
        "HASH deve contenere esattamente 40 caratteri esadecimali.\n"
        "Senza --dry-run esegue il contratto congelato mini75 subset16: "
        "200 epoche e output atomico sotto results/vae_c/overfit.\n"
        "--dry-run valida profilo, split, 16 tensori centrali, modello e "
        "MEAN+EVAL iniziale; non esegue train step e non crea output.\n",
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

            overfit_print_usage(argv[0]);
            return 0;
        }
    }
    for (argument = 1; argument < argc; argument++) {
        if (strcmp(argv[argument], "--dry-run") == 0) {
            if (dry_run) {
                overfit_print_usage(argv[0]);
                return 2;
            }
            dry_run = true;
        } else if (strcmp(argv[argument], "--code-commit") == 0) {
            if (code_commit != NULL || argument + 1 >= argc) {
                overfit_print_usage(argv[0]);
                return 2;
            }
            code_commit = argv[++argument];
        } else if (argv[argument][0] == '-' || repository_root_seen) {
            overfit_print_usage(argv[0]);
            return 2;
        } else {
            repository_root = argv[argument];
            repository_root_seen = true;
        }
    }
    if (!overfit_code_commit_is_valid(code_commit)) {
        fprintf(stderr,
                "Errore: --code-commit richiede esattamente 40 caratteri esadecimali.\n");
        overfit_print_usage(argv[0]);
        return 2;
    }

    return overfit_run(repository_root, code_commit, dry_run);
}
