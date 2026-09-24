#include "vae_dataset.h"
#include "vae_split.h"

#include <stdio.h>
#include <stdlib.h>


static size_t count_csv_data_rows(const char *csv_path)
{
    FILE *fp;
    char line[4096];
    size_t lines;

    fp = fopen(csv_path, "r");
    if (fp == NULL) {
        return 0;
    }

    lines = 0;

    while (fgets(line, sizeof(line), fp) != NULL) {
        lines++;
    }

    fclose(fp);

    if (lines == 0) {
        return 0;
    }

    /*
       Sottrae la riga header.
    */
    return lines - 1;
}


int main(int argc, char **argv)
{
    const char *manifest_path;
    double train_rate;
    unsigned int seed;

    char train_csv_path[256];
    char val_csv_path[256];

    GSManifest manifest;
    GSSplit split;

    size_t train_rows;
    size_t val_rows;

    int rc;

    /*
       Uso:
       test_save_split.exe manifest.csv train_rate seed

       Esempio:
       test_save_split.exe manifest_tensors_224_central.csv 0.8 123
    */
    manifest_path = "manifest_tensors_224_central.csv";
    train_rate = 0.8;
    seed = 123;

    if (argc >= 2) {
        manifest_path = argv[1];
    }

    if (argc >= 3) {
        train_rate = atof(argv[2]);
    }

    if (argc >= 4) {
        seed = (unsigned int)strtoul(argv[3], NULL, 10);
    }

    snprintf(
        train_csv_path,
        sizeof(train_csv_path),
        "split_train_indices_seed%u.csv",
        seed
    );

    snprintf(
        val_csv_path,
        sizeof(val_csv_path),
        "split_val_indices_seed%u.csv",
        seed
    );

    printf("Manifest:   %s\n", manifest_path);
    printf("Train rate: %.3f\n", train_rate);
    printf("Seed:       %u\n", seed);
    printf("Train CSV:  %s\n", train_csv_path);
    printf("Val CSV:    %s\n\n", val_csv_path);

    rc = gs_manifest_load(manifest_path, &manifest);
    if (rc != 0) {
        return 1;
    }

    printf("Manifest caricato correttamente.\n");
    printf("Numero eventi: %lu\n\n", (unsigned long)manifest.count);

    rc = gs_split_train_val(&manifest, train_rate, seed, &split);
    if (rc != 0) {
        gs_manifest_free(&manifest);
        return 1;
    }

    printf("Split creato:\n");
    printf("  train events      = %lu\n", (unsigned long)split.train_count);
    printf("  validation events = %lu\n\n", (unsigned long)split.val_count);

    rc = gs_split_save_train_val_csv(
        &manifest,
        &split,
        train_csv_path,
        val_csv_path
    );

    if (rc != 0) {
        gs_split_free(&split);
        gs_manifest_free(&manifest);
        return 1;
    }

    printf("File split salvati correttamente.\n");

    train_rows = count_csv_data_rows(train_csv_path);
    val_rows = count_csv_data_rows(val_csv_path);

    printf("Controllo righe dati:\n");
    printf("  %s -> %lu righe dati\n",
           train_csv_path,
           (unsigned long)train_rows);

    printf("  %s -> %lu righe dati\n",
           val_csv_path,
           (unsigned long)val_rows);

    if (train_rows != split.train_count) {
        printf("Errore: righe train diverse da train_count.\n");
        gs_split_free(&split);
        gs_manifest_free(&manifest);
        return 1;
    }

    if (val_rows != split.val_count) {
        printf("Errore: righe validation diverse da val_count.\n");
        gs_split_free(&split);
        gs_manifest_free(&manifest);
        return 1;
    }

    printf("\nOK: split salvato e verificato.\n");
    printf("Le label NON sono state salvate nei file di split.\n");
    printf("Il futuro VAE usera' gli indici per caricare batch.x.\n");

    gs_split_free(&split);
    gs_manifest_free(&manifest);

    return 0;
}