#include "vae_saved_split.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>


static void gs_saved_split_zero(GSSavedSplit *split)
{
    if (split == NULL) {
        return;
    }

    split->indices = NULL;
    split->count = 0;
    split->name[0] = '\0';
}


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
   Parser CSV semplice.

   Gestisce anche campi tra virgolette:
   "H1_BqniZ6oLFl"
   "tensors_224_central_uint8/event_007664.bin"
*/
static int gs_saved_split_csv_line(
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
        } else {
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


static int gs_saved_split_validate_header(char *line)
{
    const char *expected[6] = {
        "split_name",
        "split_pos",
        "manifest_index",
        "event_index",
        "event_id",
        "tensor_path"
    };

    char *fields[7];
    int count;
    int i;

    gs_strip_utf8_bom(line);

    count = gs_saved_split_csv_line(line, fields, 7);

    if (count != 6) {
        printf("Errore: header split non valido.\n");
        printf("Colonne trovate: %d\n", count);
        printf("Colonne attese: 6\n");
        return 1;
    }

    for (i = 0; i < 6; i++) {
        if (strcmp(fields[i], expected[i]) != 0) {
            printf("Errore: colonna split %d non valida.\n", i);
            printf("Trovata: %s\n", fields[i]);
            printf("Attesa:  %s\n", expected[i]);
            return 1;
        }
    }

    return 0;
}


static int gs_parse_size_value(
    const char *text,
    size_t *value
)
{
    char *endptr;
    unsigned long parsed;

    if (text == NULL || value == NULL || text[0] == '\0') {
        return 1;
    }

    if (text[0] == '-') {
        return 1;
    }

    endptr = NULL;
    parsed = strtoul(text, &endptr, 10);

    if (endptr == text || *endptr != '\0') {
        return 1;
    }

    *value = (size_t)parsed;

    return 0;
}


static int gs_parse_event_index(
    const char *text,
    int *value
)
{
    char *endptr;
    long parsed;

    if (text == NULL || value == NULL || text[0] == '\0') {
        return 1;
    }

    endptr = NULL;
    parsed = strtol(text, &endptr, 10);

    if (endptr == text ||
        *endptr != '\0' ||
        parsed < 1 ||
        parsed > INT_MAX) {

        return 1;
    }

    *value = (int)parsed;

    return 0;
}


void gs_saved_split_free(GSSavedSplit *split)
{
    if (split == NULL) {
        return;
    }

    free(split->indices);

    gs_saved_split_zero(split);
}


int gs_saved_split_load(
    const char *csv_path,
    const char *expected_split_name,
    const GSEventManifest *event_manifest,
    GSSavedSplit *split
)
{
    FILE *fp;

    char line[4096];
    char *fields[7];

    size_t capacity;
    size_t line_number;

    unsigned char *seen;

    if (csv_path == NULL) {
        printf("Errore: csv_path split NULL.\n");
        return 1;
    }

    if (expected_split_name == NULL) {
        printf("Errore: expected_split_name NULL.\n");
        return 1;
    }

    if (event_manifest == NULL) {
        printf("Errore: event_manifest NULL.\n");
        return 1;
    }

    if (split == NULL) {
        printf("Errore: split NULL.\n");
        return 1;
    }

    gs_saved_split_zero(split);

    fp = fopen(csv_path, "r");

    if (fp == NULL) {
        printf("Errore: impossibile aprire lo split:\n");
        printf("%s\n", csv_path);
        return 1;
    }

    if (fgets(line, sizeof(line), fp) == NULL) {
        printf("Errore: file split vuoto.\n");
        fclose(fp);
        return 1;
    }

    if (gs_saved_split_validate_header(line) != 0) {
        fclose(fp);
        return 1;
    }

    capacity = 1024;

    split->indices = (size_t *)malloc(
        capacity * sizeof(size_t)
    );

    seen = (unsigned char *)calloc(
        event_manifest->count,
        sizeof(unsigned char)
    );

    if (split->indices == NULL || seen == NULL) {
        printf("Errore: memoria insufficiente per lo split salvato.\n");

        free(seen);
        gs_saved_split_free(split);
        fclose(fp);

        return 1;
    }

    strncpy(
        split->name,
        expected_split_name,
        GS_SAVED_SPLIT_NAME_LEN - 1
    );

    split->name[GS_SAVED_SPLIT_NAME_LEN - 1] = '\0';

    line_number = 1;

    while (fgets(line, sizeof(line), fp) != NULL) {
        size_t split_pos;
        size_t manifest_index;

        int event_index;

        const GSEventRecord *event_record;

        int field_count;

        line_number++;

        if (line[0] == '\0' ||
            line[0] == '\n' ||
            line[0] == '\r') {

            continue;
        }

        field_count = gs_saved_split_csv_line(
            line,
            fields,
            7
        );

        if (field_count != 6) {
            printf("Errore CSV split alla riga %lu.\n",
                   (unsigned long)line_number);

            printf("Campi trovati: %d\n", field_count);

            free(seen);
            gs_saved_split_free(split);
            fclose(fp);

            return 1;
        }

        if (strcmp(fields[0], expected_split_name) != 0) {
            printf("Errore: split_name non valido alla riga %lu.\n",
                   (unsigned long)line_number);

            printf("Trovato: %s\n", fields[0]);
            printf("Atteso:  %s\n", expected_split_name);

            free(seen);
            gs_saved_split_free(split);
            fclose(fp);

            return 1;
        }

        if (gs_parse_size_value(fields[1], &split_pos) != 0 ||
            gs_parse_size_value(fields[2], &manifest_index) != 0 ||
            gs_parse_event_index(fields[3], &event_index) != 0) {

            printf("Errore: valore numerico non valido alla riga %lu.\n",
                   (unsigned long)line_number);

            free(seen);
            gs_saved_split_free(split);
            fclose(fp);

            return 1;
        }

        /*
           split_pos deve essere:
           0, 1, 2, 3, ...
        */
        if (split_pos != split->count) {
            printf("Errore: split_pos non sequenziale.\n");
            printf("Riga: %lu\n", (unsigned long)line_number);
            printf("Trovato: %lu\n", (unsigned long)split_pos);
            printf("Atteso:  %lu\n", (unsigned long)split->count);

            free(seen);
            gs_saved_split_free(split);
            fclose(fp);

            return 1;
        }

        if (manifest_index >= event_manifest->count) {
            printf("Errore: manifest_index fuori intervallo.\n");
            printf("Valore: %lu\n",
                   (unsigned long)manifest_index);

            free(seen);
            gs_saved_split_free(split);
            fclose(fp);

            return 1;
        }

        if (seen[manifest_index] != 0) {
            printf("Errore: manifest_index duplicato nello split.\n");
            printf("Valore duplicato: %lu\n",
                   (unsigned long)manifest_index);

            free(seen);
            gs_saved_split_free(split);
            fclose(fp);

            return 1;
        }

        event_record = gs_event_manifest_get(
            event_manifest,
            manifest_index
        );

        if (event_record == NULL) {
            printf("Errore: record evento NULL.\n");

            free(seen);
            gs_saved_split_free(split);
            fclose(fp);

            return 1;
        }

        /*
           Verifica che il CSV di split punti proprio
           allo stesso evento nel manifest delle PNG.
        */
        if (event_index != event_record->event_index) {
            printf("Errore: event_index non coerente.\n");
            printf("Split:    %d\n", event_index);
            printf("Manifest: %d\n", event_record->event_index);

            free(seen);
            gs_saved_split_free(split);
            fclose(fp);

            return 1;
        }

        if (strcmp(fields[4], event_record->event_id) != 0) {
            printf("Errore: event_id non coerente.\n");
            printf("Split:    %s\n", fields[4]);
            printf("Manifest: %s\n", event_record->event_id);

            free(seen);
            gs_saved_split_free(split);
            fclose(fp);

            return 1;
        }

        /*
           tensor_path resta nel CSV perché proviene
           dal dataset centrale, ma il loader dinamico
           userà invece le quattro PNG del manifest eventi.
        */
        if (fields[5][0] == '\0') {
            printf("Errore: tensor_path vuoto alla riga %lu.\n",
                   (unsigned long)line_number);

            free(seen);
            gs_saved_split_free(split);
            fclose(fp);

            return 1;
        }

        if (split->count == capacity) {
            size_t *new_indices;

            capacity *= 2;

            new_indices = (size_t *)realloc(
                split->indices,
                capacity * sizeof(size_t)
            );

            if (new_indices == NULL) {
                printf("Errore: memoria insufficiente durante realloc.\n");

                free(seen);
                gs_saved_split_free(split);
                fclose(fp);

                return 1;
            }

            split->indices = new_indices;
        }

        split->indices[split->count] = manifest_index;
        split->count++;

        seen[manifest_index] = 1;
    }

    free(seen);
    fclose(fp);

    if (split->count == 0) {
        printf("Errore: nessun indice trovato nello split.\n");
        gs_saved_split_free(split);
        return 1;
    }

    return 0;
}