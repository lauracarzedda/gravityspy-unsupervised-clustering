#include "vae_loader.h"
#include "vae_dataset.h"
#include "vae_batch.h"
#include "vae_shuffle.h"
#include "vae_split.h"

#include <stdio.h>
#include <stdlib.h>
#include <float.h>


static int run_loader_epoch_preview(
    const char *name,
    const GSManifest *manifest,
    const size_t *indices,
    size_t index_count,
    size_t batch_size,
    size_t max_batches
)
{
    GSBatch batch;
    size_t start_pos;
    size_t loaded_events;
    size_t batch_counter;

    float global_min;
    float global_max;
    double global_sum;
    size_t global_values;

    int rc;

    if (name == NULL || manifest == NULL || indices == NULL) {
        printf("Errore: argomenti NULL in run_loader_epoch_preview.\n");
        return 1;
    }

    if (index_count == 0) {
        printf("Errore: index_count uguale a zero per %s.\n", name);
        return 1;
    }

    if (batch_size == 0) {
        printf("Errore: batch_size uguale a zero.\n");
        return 1;
    }

    rc = gs_batch_alloc(&batch, batch_size);
    if (rc != 0) {
        return 1;
    }

    start_pos = 0;
    loaded_events = 0;
    batch_counter = 0;

    global_min = FLT_MAX;
    global_max = -FLT_MAX;
    global_sum = 0.0;
    global_values = 0;

    printf("=== Dry-run loader: %s ===\n", name);
    printf("Eventi disponibili: %lu\n", (unsigned long)index_count);
    printf("Batch size:         %lu\n", (unsigned long)batch_size);

    if (max_batches == 0) {
        printf("Max batches:        tutti\n\n");
    } else {
        printf("Max batches:        %lu\n\n", (unsigned long)max_batches);
    }

    while (start_pos < index_count) {
        size_t remaining;
        size_t current_batch_size;
        GSBatchStats stats;

        remaining = index_count - start_pos;

        if (remaining < batch_size) {
            current_batch_size = remaining;
        } else {
            current_batch_size = batch_size;
        }

        rc = gs_batch_load_from_indices(
            manifest,
            indices,
            index_count,
            start_pos,
            current_batch_size,
            &batch
        );

        if (rc != 0) {
            gs_batch_free(&batch);
            return 1;
        }

        rc = gs_batch_compute_stats(&batch, &stats);
        if (rc != 0) {
            gs_batch_free(&batch);
            return 1;
        }

        if (stats.min_value < global_min) {
            global_min = stats.min_value;
        }

        if (stats.max_value > global_max) {
            global_max = stats.max_value;
        }

        global_sum += stats.mean_value * (double)(batch.count * GS_TENSOR_SIZE);
        global_values += batch.count * GS_TENSOR_SIZE;

        /*
           Stampa solo i primi batch, così il terminale non esplode.
        */
        if (batch_counter < 3) {
            printf("Batch %lu:\n", (unsigned long)batch_counter);
            printf("  start_pos = %lu\n", (unsigned long)start_pos);
            printf("  count     = %lu\n", (unsigned long)batch.count);
            printf("  shape     = %lu x %d x %d x %d\n",
                   (unsigned long)batch.count,
                   GS_CHANNELS,
                   GS_HEIGHT,
                   GS_WIDTH);
            printf("  min/max/mean = %.6f / %.6f / %.6f\n",
                   stats.min_value,
                   stats.max_value,
                   stats.mean_value);

            printf("  primo evento del batch: label=%s, event_id=%s\n",
                   gs_batch_label(&batch, 0),
                   gs_batch_event_id(&batch, 0));

            printf("\n");
        }

        loaded_events += batch.count;
        start_pos += batch.count;
        batch_counter++;

        if (max_batches > 0 && batch_counter >= max_batches) {
            break;
        }
    }

    printf("Riassunto dry-run %s:\n", name);
    printf("  batch processati = %lu\n", (unsigned long)batch_counter);
    printf("  eventi caricati  = %lu\n", (unsigned long)loaded_events);

    if (max_batches == 0) {
        printf("  eventi attesi    = %lu\n", (unsigned long)index_count);
    } else {
        printf("  nota: test limitato a max_batches=%lu\n", (unsigned long)max_batches);
    }

    if (global_values > 0) {
        printf("  statistiche globali sui batch letti:\n");
        printf("    min  = %.6f\n", global_min);
        printf("    max  = %.6f\n", global_max);
        printf("    mean = %.6f\n", global_sum / (double)global_values);
    }

    printf("\n");

    gs_batch_free(&batch);
    return 0;
}


int main(int argc, char **argv)
{
    const char *manifest_path;
    double train_rate;
    size_t batch_size;
    unsigned int seed;
    size_t max_batches;

    GSManifest manifest;
    GSSplit split;

    int rc;

    /*
       Uso:
       test_epoch.exe manifest.csv train_rate batch_size seed max_batches

       Esempio leggero:
       test_epoch.exe manifest_tensors_224_central.csv 0.8 4 123 3

       Se max_batches = 0, percorre tutto train e tutta validation.
    */
    manifest_path = "manifest_tensors_224_central.csv";
    train_rate = 0.8;
    batch_size = 4;
    seed = 123;
    max_batches = 3;

    if (argc >= 2) {
        manifest_path = argv[1];
    }

    if (argc >= 3) {
        train_rate = atof(argv[2]);
    }

    if (argc >= 4) {
        batch_size = (size_t)strtoul(argv[3], NULL, 10);
    }

    if (argc >= 5) {
        seed = (unsigned int)strtoul(argv[4], NULL, 10);
    }

    if (argc >= 6) {
        max_batches = (size_t)strtoul(argv[5], NULL, 10);
    }

    printf("Manifest:    %s\n", manifest_path);
    printf("Train rate:  %.3f\n", train_rate);
    printf("Batch size:  %lu\n", (unsigned long)batch_size);
    printf("Seed:        %u\n", seed);
    printf("Max batches: %lu\n\n", (unsigned long)max_batches);

    rc = gs_manifest_load(manifest_path, &manifest);
    if (rc != 0) {
        return 1;
    }

    rc = gs_split_train_val(&manifest, train_rate, seed, &split);
    if (rc != 0) {
        gs_manifest_free(&manifest);
        return 1;
    }

    printf("Split creato:\n");
    printf("  train events      = %lu\n", (unsigned long)split.train_count);
    printf("  validation events = %lu\n\n", (unsigned long)split.val_count);

    rc = run_loader_epoch_preview(
        "TRAIN",
        &manifest,
        split.train_indices,
        split.train_count,
        batch_size,
        max_batches
    );

    if (rc != 0) {
        gs_split_free(&split);
        gs_manifest_free(&manifest);
        return 1;
    }

    rc = run_loader_epoch_preview(
        "VALIDATION",
        &manifest,
        split.val_indices,
        split.val_count,
        batch_size,
        max_batches
    );

    if (rc != 0) {
        gs_split_free(&split);
        gs_manifest_free(&manifest);
        return 1;
    }

    gs_split_free(&split);
    gs_manifest_free(&manifest);

    printf("Dry-run completato correttamente.\n");
    printf("Il futuro VAE usera' batch.x, non le label.\n");

    return 0;
}