#include "vae_event_dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>


/*
   Inizializza il manifest in uno stato vuoto e sicuro.
*/
static void gs_event_manifest_zero(GSEventManifest *manifest)
{
    if (manifest == NULL) {
        return;
    }

    manifest->records = NULL;
    manifest->count = 0;
}


/*
   Copia una stringa garantendo sempre il terminatore '\0'.

   Ritorna:
   0 = copia riuscita
   1 = stringa troppo lunga o argomenti non validi
*/
static int gs_copy_string_checked(
    char *dst,
    size_t dst_size,
    const char *src
)
{
    size_t len;

    if (dst == NULL || dst_size == 0 || src == NULL) {
        return 1;
    }

    len = strlen(src);

    if (len >= dst_size) {
        return 1;
    }

    memcpy(dst, src, len + 1);

    return 0;
}


/*
   Rimuove un eventuale BOM UTF-8 all'inizio del CSV.
   Serve a rendere il parser un po' più robusto.
*/
static void gs_strip_utf8_bom(char *line)
{
    unsigned char *p;

    if (line == NULL) {
        return;
    }

    p = (unsigned char *)line;

    if (p[0] == 0xEF &&
        p[1] == 0xBB &&
        p[2] == 0xBF) {

        memmove(line, line + 3, strlen(line + 3) + 1);
    }
}


/*
   Divide una riga CSV in campi.

   Gestisce sia campi normali:

       1

   sia campi tra virgolette:

       "1080Lines"
       "out272_gray/img_000001.png"

   La funzione modifica la riga in-place:
   sostituisce separatori e virgolette finali con '\0'.
*/
static int gs_csv_split_line(
    char *line,
    char **fields,
    int max_fields
)
{
    char *p;
    int count;

    if (line == NULL || fields == NULL || max_fields <= 0) {
        return 0;
    }

    p = line;
    count = 0;

    while (*p != '\0' && count < max_fields) {

        if (*p == '\r' || *p == '\n') {
            break;
        }

        /*
           Campo racchiuso tra virgolette.
        */
        if (*p == '"') {
            p++;

            fields[count] = p;
            count++;

            while (*p != '\0') {

                if (*p == '"') {
                    *p = '\0';
                    p++;

                    if (*p == ',') {
                        p++;
                    }

                    break;
                }

                p++;
            }
        }

        /*
           Campo non racchiuso tra virgolette.
        */
        else {
            fields[count] = p;
            count++;

            while (*p != '\0' &&
                   *p != ',' &&
                   *p != '\r' &&
                   *p != '\n') {

                p++;
            }

            if (*p == ',') {
                *p = '\0';
                p++;
            } else {
                *p = '\0';
            }
        }
    }

    return count;
}


/*
   Verifica che l'header abbia esattamente le sette colonne attese.
*/
static int gs_validate_event_header(char *line)
{
    const char *expected[7] = {
        "event_index",
        "label",
        "event_id",
        "img_0_5_272",
        "img_1_0_272",
        "img_2_0_272",
        "img_4_0_272"
    };

    char *fields[8];
    int field_count;
    int i;

    gs_strip_utf8_bom(line);

    field_count = gs_csv_split_line(line, fields, 8);

    if (field_count != 7) {
        printf("Errore: header manifest eventi non valido.\n");
        printf("Colonne trovate: %d\n", field_count);
        printf("Colonne attese: 7\n");
        return 1;
    }

    for (i = 0; i < 7; i++) {
        if (strcmp(fields[i], expected[i]) != 0) {
            printf("Errore: colonna %d non valida.\n", i);
            printf("Trovata: %s\n", fields[i]);
            printf("Attesa:  %s\n", expected[i]);
            return 1;
        }
    }

    return 0;
}


void gs_event_manifest_free(GSEventManifest *manifest)
{
    if (manifest == NULL) {
        return;
    }

    free(manifest->records);

    gs_event_manifest_zero(manifest);
}


static int gs_event_manifest_load_impl(
    const char *csv_path,
    GSEventManifest *manifest,
    int label_free
)
{
    FILE *fp;

    char line[4096];
    char *fields[8];

    size_t capacity;
    size_t line_number;

    if (csv_path == NULL) {
        printf("Errore: csv_path NULL.\n");
        return 1;
    }

    if (manifest == NULL) {
        printf("Errore: manifest eventi NULL.\n");
        return 1;
    }

    gs_event_manifest_zero(manifest);

    fp = fopen(csv_path, "r");

    if (fp == NULL) {
        printf("Errore: impossibile aprire il manifest eventi:\n");
        printf("%s\n", csv_path);
        return 1;
    }

    /*
       Legge e controlla l'header.
    */
    if (fgets(line, sizeof(line), fp) == NULL) {
        printf("Errore: manifest eventi vuoto.\n");
        fclose(fp);
        return 1;
    }

    if (gs_validate_event_header(line) != 0) {
        fclose(fp);
        return 1;
    }

    /*
       Allocazione iniziale.
       Se servirà più spazio useremo realloc.
    */
    capacity = 1024;

    manifest->records = (GSEventRecord *)malloc(
        capacity * sizeof(GSEventRecord)
    );

    if (manifest->records == NULL) {
        printf("Errore: memoria insufficiente per il manifest eventi.\n");
        fclose(fp);
        return 1;
    }

    line_number = 1;

    while (fgets(line, sizeof(line), fp) != NULL) {
        GSEventRecord record;

        int field_count;
        char *endptr;
        long parsed_index;

        line_number++;

        /*
           Salta eventuali righe vuote.
        */
        if (line[0] == '\0' ||
            line[0] == '\n' ||
            line[0] == '\r') {

            continue;
        }

        memset(&record, 0, sizeof(record));

        field_count = gs_csv_split_line(line, fields, 8);

        if (field_count != 7) {
            printf("Errore CSV alla riga %lu.\n",
                   (unsigned long)line_number);

            printf("Campi trovati: %d\n", field_count);
            printf("Campi attesi: 7\n");

            gs_event_manifest_free(manifest);
            fclose(fp);
            return 1;
        }

        /*
           Converte event_index da testo a intero.
        */
        endptr = NULL;
        parsed_index = strtol(fields[0], &endptr, 10);

        if (endptr == fields[0] ||
            *endptr != '\0' ||
            parsed_index < 1 ||
            parsed_index > INT_MAX) {

            printf("Errore: event_index non valido alla riga %lu.\n",
                   (unsigned long)line_number);

            printf("Valore trovato: %s\n", fields[0]);

            gs_event_manifest_free(manifest);
            fclose(fp);
            return 1;
        }

        record.event_index = (int)parsed_index;

        /* In label-free inference the label column is never dereferenced. */
        if (!label_free && gs_copy_string_checked(
                record.label,
                sizeof(record.label),
                fields[1]
            ) != 0) {

            printf("Errore: label troppo lunga alla riga %lu.\n",
                   (unsigned long)line_number);

            gs_event_manifest_free(manifest);
            fclose(fp);
            return 1;
        }

        if (gs_copy_string_checked(
                record.event_id,
                sizeof(record.event_id),
                fields[2]
            ) != 0) {

            printf("Errore: event_id troppo lungo alla riga %lu.\n",
                   (unsigned long)line_number);

            gs_event_manifest_free(manifest);
            fclose(fp);
            return 1;
        }

        /*
           Copia i quattro percorsi nell'ordine temporale corretto.
        */
        if (gs_copy_string_checked(
                record.image_paths[0],
                sizeof(record.image_paths[0]),
                fields[3]
            ) != 0 ||

            gs_copy_string_checked(
                record.image_paths[1],
                sizeof(record.image_paths[1]),
                fields[4]
            ) != 0 ||

            gs_copy_string_checked(
                record.image_paths[2],
                sizeof(record.image_paths[2]),
                fields[5]
            ) != 0 ||

            gs_copy_string_checked(
                record.image_paths[3],
                sizeof(record.image_paths[3]),
                fields[6]
            ) != 0) {

            printf("Errore: percorso immagine troppo lungo alla riga %lu.\n",
                   (unsigned long)line_number);

            gs_event_manifest_free(manifest);
            fclose(fp);
            return 1;
        }

        /*
           Aumenta l'array se la capacità è esaurita.
        */
        if (manifest->count == capacity) {
            GSEventRecord *new_records;

            capacity *= 2;

            new_records = (GSEventRecord *)realloc(
                manifest->records,
                capacity * sizeof(GSEventRecord)
            );

            if (new_records == NULL) {
                printf("Errore: memoria insufficiente durante realloc.\n");

                gs_event_manifest_free(manifest);
                fclose(fp);
                return 1;
            }

            manifest->records = new_records;
        }

        manifest->records[manifest->count] = record;
        manifest->count++;
    }

    fclose(fp);

    if (manifest->count == 0) {
        printf("Errore: nessun evento trovato nel manifest.\n");
        gs_event_manifest_free(manifest);
        return 1;
    }

    return 0;
}


const GSEventRecord *gs_event_manifest_get(
    const GSEventManifest *manifest,
    size_t index
)
{
    if (manifest == NULL) {
        return NULL;
    }

    if (index >= manifest->count) {
        return NULL;
    }

    return &manifest->records[index];
}


const char *gs_event_view_name(size_t view_index)
{
    if (view_index == 0) return "0.5 s";
    if (view_index == 1) return "1.0 s";
    if (view_index == 2) return "2.0 s";
    if (view_index == 3) return "4.0 s";

    return "vista sconosciuta";
}


const char *gs_event_image_path(
    const GSEventRecord *record,
    size_t view_index
)
{
    if (record == NULL) {
        return NULL;
    }

    if (view_index >= GS_EVENT_VIEW_COUNT) {
        return NULL;
    }

    return record->image_paths[view_index];
}


void gs_event_manifest_print_record(
    const GSEventRecord *record
)
{
    size_t view;

    if (record == NULL) {
        printf("Record evento NULL.\n");
        return;
    }

    printf("event_index: %d\n", record->event_index);
    printf("label:       %s\n", record->label);
    printf("event_id:    %s\n", record->event_id);

    for (view = 0; view < GS_EVENT_VIEW_COUNT; view++) {
        printf("vista %lu (%s): %s\n",
               (unsigned long)view,
               gs_event_view_name(view),
               record->image_paths[view]);
    }
}


int gs_event_manifest_check_files(
    const GSEventManifest *manifest,
    size_t *checked_files,
    size_t *missing_files,
    size_t max_missing_to_print
)
{
    size_t checked;
    size_t missing;
    size_t event_index;
    size_t view_index;

    if (manifest == NULL ||
        checked_files == NULL ||
        missing_files == NULL) {

        printf("Errore: argomenti NULL nel controllo file.\n");
        return 1;
    }

    checked = 0;
    missing = 0;

    for (event_index = 0;
         event_index < manifest->count;
         event_index++) {

        const GSEventRecord *record;

        record = &manifest->records[event_index];

        for (view_index = 0;
             view_index < GS_EVENT_VIEW_COUNT;
             view_index++) {

            FILE *fp;
            const char *path;

            path = record->image_paths[view_index];

            checked++;

            fp = fopen(path, "rb");

            if (fp == NULL) {
                if (missing < max_missing_to_print) {
                    printf(
                        "File mancante: evento=%d, vista=%s, path=%s\n",
                        record->event_index,
                        gs_event_view_name(view_index),
                        path
                    );
                }

                missing++;
            } else {
                fclose(fp);
            }
        }
    }

    *checked_files = checked;
    *missing_files = missing;

    return 0;
}


int gs_event_manifest_load(
    const char *csv_path,
    GSEventManifest *manifest
)
{
    return gs_event_manifest_load_impl(csv_path, manifest, 0);
}


int gs_event_manifest_load_label_free(
    const char *csv_path,
    GSEventManifest *manifest
)
{
    return gs_event_manifest_load_impl(csv_path, manifest, 1);
}
