#include "vae_split.h"
#include "vae_shuffle.h"

#include <stdio.h>
#include <stdlib.h>


static void gs_split_zero(GSSplit *split)
{
    if (split == NULL) {
        return;
    }

    split->train_indices = NULL;
    split->train_count = 0;

    split->val_indices = NULL;
    split->val_count = 0;

    split->total_count = 0;
    split->train_rate = 0.0;
    split->seed = 0;
}


void gs_split_free(GSSplit *split)
{
    if (split == NULL) {
        return;
    }

    free(split->train_indices);
    free(split->val_indices);

    gs_split_zero(split);
}


int gs_split_train_val(
    const GSManifest *manifest,
    double train_rate,
    unsigned int seed,
    GSSplit *split
)
{
    size_t *all_indices;
    size_t total;
    size_t train_count;
    size_t val_count;
    size_t i;
    int rc;

    if (manifest == NULL) {
        printf("Errore: manifest NULL.\n");
        return 1;
    }

    if (split == NULL) {
        printf("Errore: split NULL.\n");
        return 1;
    }

    if (manifest->count == 0) {
        printf("Errore: manifest vuoto.\n");
        return 1;
    }

    if (train_rate <= 0.0 || train_rate >= 1.0) {
        printf("Errore: train_rate deve essere tra 0 e 1.\n");
        printf("train_rate = %.6f\n", train_rate);
        return 1;
    }

    gs_split_zero(split);

    total = manifest->count;

    /*
       Per 8583 eventi e train_rate = 0.8:
       train_count = 6866
       val_count   = 1717
    */
    train_count = (size_t)((double)total * train_rate);

    if (train_count == 0) {
        train_count = 1;
    }

    if (train_count >= total) {
        train_count = total - 1;
    }

    val_count = total - train_count;

    all_indices = NULL;

    rc = gs_indices_alloc(&all_indices, total);
    if (rc != 0) {
        return 1;
    }

    gs_indices_fill_sequential(all_indices, total);
    gs_indices_shuffle(all_indices, total, seed);

    split->train_indices = (size_t *)malloc(train_count * sizeof(size_t));
    split->val_indices = (size_t *)malloc(val_count * sizeof(size_t));

    if (split->train_indices == NULL || split->val_indices == NULL) {
        printf("Errore: memoria insufficiente per split train/validation.\n");

        free(all_indices);
        gs_split_free(split);
        return 1;
    }

    for (i = 0; i < train_count; i++) {
        split->train_indices[i] = all_indices[i];
    }

    for (i = 0; i < val_count; i++) {
        split->val_indices[i] = all_indices[train_count + i];
    }

    split->train_count = train_count;
    split->val_count = val_count;
    split->total_count = total;
    split->train_rate = train_rate;
    split->seed = seed;

    free(all_indices);

    return 0;
}


void gs_split_print_summary(
    const GSManifest *manifest,
    const GSSplit *split,
    size_t preview_count
)
{
    size_t i;

    if (manifest == NULL) {
        printf("Manifest NULL.\n");
        return;
    }

    if (split == NULL) {
        printf("Split NULL.\n");
        return;
    }

    printf("Split train/validation creato correttamente.\n");
    printf("Totale eventi:      %lu\n", (unsigned long)split->total_count);
    printf("Train rate:         %.3f\n", split->train_rate);
    printf("Seed shuffle:       %u\n", split->seed);
    printf("Train events:       %lu\n", (unsigned long)split->train_count);
    printf("Validation events:  %lu\n", (unsigned long)split->val_count);

    printf("\nPrimi eventi del train set, solo per controllo:\n");

    for (i = 0; i < preview_count && i < split->train_count; i++) {
        size_t manifest_index;
        const GSManifestRecord *record;

        manifest_index = split->train_indices[i];
        record = gs_manifest_get(manifest, manifest_index);

        if (record != NULL) {
            printf("  train[%lu] -> manifest_index=%lu, event_index=%d, label=%s, event_id=%s\n",
                   (unsigned long)i,
                   (unsigned long)manifest_index,
                   record->event_index,
                   record->label,
                   record->event_id);
        }
    }

    printf("\nPrimi eventi del validation set, solo per controllo:\n");

    for (i = 0; i < preview_count && i < split->val_count; i++) {
        size_t manifest_index;
        const GSManifestRecord *record;

        manifest_index = split->val_indices[i];
        record = gs_manifest_get(manifest, manifest_index);

        if (record != NULL) {
            printf("  val[%lu] -> manifest_index=%lu, event_index=%d, label=%s, event_id=%s\n",
                   (unsigned long)i,
                   (unsigned long)manifest_index,
                   record->event_index,
                   record->label,
                   record->event_id);
        }
    }
}


int gs_split_save_indices_csv(
    const char *csv_path,
    const char *split_name,
    const GSManifest *manifest,
    const size_t *indices,
    size_t count
)
{
    FILE *fp;
    size_t i;

    if (csv_path == NULL) {
        printf("Errore: csv_path NULL.\n");
        return 1;
    }

    if (split_name == NULL) {
        printf("Errore: split_name NULL.\n");
        return 1;
    }

    if (manifest == NULL) {
        printf("Errore: manifest NULL.\n");
        return 1;
    }

    if (indices == NULL) {
        printf("Errore: indices NULL.\n");
        return 1;
    }

    fp = fopen(csv_path, "w");
    if (fp == NULL) {
        printf("Errore: impossibile creare il file CSV:\n%s\n", csv_path);
        return 1;
    }

    /*
       Non salviamo la label.
       La label resta nel manifest originale e sarà usata solo per valutazione.
    */
    fprintf(fp, "split_name,split_pos,manifest_index,event_index,event_id,tensor_path\n");

    for (i = 0; i < count; i++) {
        size_t manifest_index;
        const GSManifestRecord *record;

        manifest_index = indices[i];

        if (manifest_index >= manifest->count) {
            printf("Errore: indice manifest non valido durante salvataggio split.\n");
            printf("manifest_index = %lu\n", (unsigned long)manifest_index);
            printf("manifest count = %lu\n", (unsigned long)manifest->count);

            fclose(fp);
            return 1;
        }

        record = gs_manifest_get(manifest, manifest_index);

        if (record == NULL) {
            printf("Errore: record NULL durante salvataggio split.\n");
            fclose(fp);
            return 1;
        }

        fprintf(
            fp,
            "%s,%lu,%lu,%d,\"%s\",\"%s\"\n",
            split_name,
            (unsigned long)i,
            (unsigned long)manifest_index,
            record->event_index,
            record->event_id,
            record->tensor_path
        );
    }

    fclose(fp);

    return 0;
}


int gs_split_save_train_val_csv(
    const GSManifest *manifest,
    const GSSplit *split,
    const char *train_csv_path,
    const char *val_csv_path
)
{
    int rc;

    if (manifest == NULL) {
        printf("Errore: manifest NULL.\n");
        return 1;
    }

    if (split == NULL) {
        printf("Errore: split NULL.\n");
        return 1;
    }

    if (train_csv_path == NULL) {
        printf("Errore: train_csv_path NULL.\n");
        return 1;
    }

    if (val_csv_path == NULL) {
        printf("Errore: val_csv_path NULL.\n");
        return 1;
    }

    rc = gs_split_save_indices_csv(
        train_csv_path,
        "train",
        manifest,
        split->train_indices,
        split->train_count
    );

    if (rc != 0) {
        return 1;
    }

    rc = gs_split_save_indices_csv(
        val_csv_path,
        "validation",
        manifest,
        split->val_indices,
        split->val_count
    );

    if (rc != 0) {
        return 1;
    }

    return 0;
}