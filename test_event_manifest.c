#include "vae_event_dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main(int argc, char **argv)
{
    const char *manifest_path;

    GSEventManifest manifest;

    size_t i;
    size_t order_errors;
    size_t empty_field_errors;

    size_t checked_files;
    size_t missing_files;

    int rc;

    /*
       Uso:

       test_event_manifest.exe manifest_events_272.csv
    */
    manifest_path = "manifest_events_272.csv";

    if (argc >= 2) {
        manifest_path = argv[1];
    }

    printf("Manifest eventi: %s\n\n", manifest_path);

    rc = gs_event_manifest_load(manifest_path, &manifest);

    if (rc != 0) {
        return 1;
    }

    printf("Manifest eventi caricato correttamente.\n");
    printf("Numero eventi: %lu\n\n",
           (unsigned long)manifest.count);

    /*
       Il dataset ufficiale preparato contiene 8583 eventi.
    */
    if (manifest.count != 8583) {
        printf("Errore: numero eventi inatteso.\n");
        printf("Attesi: 8583\n");
        printf("Trovati: %lu\n",
               (unsigned long)manifest.count);

        gs_event_manifest_free(&manifest);
        return 1;
    }

    printf("Primi 3 record:\n\n");

    for (i = 0; i < 3 && i < manifest.count; i++) {
        printf("---- Record %lu ----\n",
               (unsigned long)i);

        gs_event_manifest_print_record(
            &manifest.records[i]
        );

        printf("\n");
    }

    /*
       Controllo 1:
       event_index deve seguire l'ordine 1,2,3,...
    */
    order_errors = 0;

    /*
       Controllo 2:
       label, event_id e percorsi non devono essere vuoti.
    */
    empty_field_errors = 0;

    for (i = 0; i < manifest.count; i++) {
        const GSEventRecord *record;
        size_t view;

        record = gs_event_manifest_get(&manifest, i);

        if (record == NULL) {
            printf("Errore: record NULL all'indice %lu.\n",
                   (unsigned long)i);

            gs_event_manifest_free(&manifest);
            return 1;
        }

        if (record->event_index != (int)(i + 1)) {
            if (order_errors < 10) {
                printf(
                    "Ordine errato: posizione=%lu, event_index=%d\n",
                    (unsigned long)i,
                    record->event_index
                );
            }

            order_errors++;
        }

        if (strlen(record->label) == 0 ||
            strlen(record->event_id) == 0) {

            empty_field_errors++;
        }

        for (view = 0;
             view < GS_EVENT_VIEW_COUNT;
             view++) {

            if (strlen(record->image_paths[view]) == 0) {
                empty_field_errors++;
            }
        }
    }

    printf("Controlli struttura:\n");
    printf("  errori ordine event_index = %lu\n",
           (unsigned long)order_errors);

    printf("  campi vuoti               = %lu\n",
           (unsigned long)empty_field_errors);

    if (order_errors != 0 || empty_field_errors != 0) {
        printf("Errore: manifest non coerente.\n");

        gs_event_manifest_free(&manifest);
        return 1;
    }

    /*
       Controlla l'esistenza di tutte le 34332 PNG.

       Il programma apre e chiude i file soltanto per verificarli:
       non legge ancora i pixel e non modifica le immagini.
    */
    checked_files = 0;
    missing_files = 0;

    rc = gs_event_manifest_check_files(
        &manifest,
        &checked_files,
        &missing_files,
        10
    );

    if (rc != 0) {
        gs_event_manifest_free(&manifest);
        return 1;
    }

    printf("\nControllo percorsi PNG:\n");
    printf("  file controllati = %lu\n",
           (unsigned long)checked_files);

    printf("  file mancanti    = %lu\n",
           (unsigned long)missing_files);

    if (checked_files != manifest.count * GS_EVENT_VIEW_COUNT) {
        printf("Errore: numero file controllati inatteso.\n");

        gs_event_manifest_free(&manifest);
        return 1;
    }

    if (missing_files != 0) {
        printf("Errore: alcuni file PNG non sono stati trovati.\n");

        gs_event_manifest_free(&manifest);
        return 1;
    }

    printf("\nOK: parser del manifest eventi completato.\n");
    printf("Eventi validi: %lu\n",
           (unsigned long)manifest.count);

    printf("Viste PNG valide: %lu\n",
           (unsigned long)checked_files);

    printf("Ordine canali: 0.5 s, 1.0 s, 2.0 s, 4.0 s\n");
    printf("In questo test non e' stato ancora applicato alcun crop.\n");

    gs_event_manifest_free(&manifest);

    return 0;
}