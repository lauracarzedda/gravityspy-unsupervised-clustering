#include "vae_dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static void gs_copy_string(char *dst, size_t dst_size, const char *src)
{
    if (dst == NULL || dst_size == 0) {
        return;
    }

    if (src == NULL) {
        dst[0] = '\0';
        return;
    }

    strncpy(dst, src, dst_size - 1);
    dst[dst_size - 1] = '\0';
}


/*
   Parser CSV semplice ma sufficiente per il nostro manifest.

   Gestisce campi tra virgolette come:
   "1080Lines"
   "H1_09HE6k6EaS"
   "tensors_224_central_uint8/event_000001.bin"

   Modifica la riga in-place e mette i puntatori ai campi dentro fields.
*/
static int gs_csv_split_line(char *line, char **fields, int max_fields)
{
    int n = 0;
    char *p = line;

    while (*p != '\0' && n < max_fields) {

        if (*p == '\r' || *p == '\n') {
            break;
        }

        if (*p == '"') {
            p++;
            fields[n] = p;
            n++;

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
            fields[n] = p;
            n++;

            while (*p != '\0' && *p != ',' && *p != '\r' && *p != '\n') {
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

    return n;
}


void gs_manifest_free(GSManifest *manifest)
{
    if (manifest == NULL) {
        return;
    }

    free(manifest->records);
    manifest->records = NULL;
    manifest->count = 0;
}


int gs_manifest_load(const char *csv_path, GSManifest *manifest)
{
    FILE *fp;
    char line[4096];
    char *fields[16];

    size_t capacity;
    size_t line_number;

    if (csv_path == NULL) {
        printf("Errore: csv_path NULL.\n");
        return 1;
    }

    if (manifest == NULL) {
        printf("Errore: manifest NULL.\n");
        return 1;
    }

    manifest->records = NULL;
    manifest->count = 0;

    fp = fopen(csv_path, "r");
    if (fp == NULL) {
        printf("Errore: impossibile aprire il manifest:\n%s\n", csv_path);
        return 1;
    }

    /*
       Salta header.
    */
    if (fgets(line, sizeof(line), fp) == NULL) {
        printf("Errore: manifest vuoto.\n");
        fclose(fp);
        return 1;
    }

    capacity = 1024;
    manifest->records = (GSManifestRecord *)malloc(capacity * sizeof(GSManifestRecord));

    if (manifest->records == NULL) {
        printf("Errore: memoria insufficiente per il manifest.\n");
        fclose(fp);
        return 1;
    }

    line_number = 1;

    while (fgets(line, sizeof(line), fp) != NULL) {
        int nf;
        GSManifestRecord rec;

        line_number++;

        /*
           Salta eventuali righe vuote.
        */
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') {
            continue;
        }

        nf = gs_csv_split_line(line, fields, 16);

        if (nf < 11) {
            printf("Errore CSV alla riga %lu: campi trovati = %d, attesi = 11.\n",
                   (unsigned long)line_number,
                   nf);
            gs_manifest_free(manifest);
            fclose(fp);
            return 1;
        }

        rec.event_index = atoi(fields[0]);

        gs_copy_string(rec.label, GS_MAX_LABEL_LEN, fields[1]);
        gs_copy_string(rec.event_id, GS_MAX_EVENT_ID_LEN, fields[2]);
        gs_copy_string(rec.tensor_path, GS_MAX_PATH_LEN, fields[3]);

        rec.channels = atoi(fields[4]);
        rec.height = atoi(fields[5]);
        rec.width = atoi(fields[6]);

        gs_copy_string(rec.dtype, GS_MAX_DTYPE_LEN, fields[7]);
        gs_copy_string(rec.layout, GS_MAX_LAYOUT_LEN, fields[8]);
        gs_copy_string(rec.channel_order, GS_MAX_CHANNEL_ORDER_LEN, fields[9]);

        rec.crop_left = atoi(fields[10]);

        /*
           Controlli di coerenza sul formato atteso.
        */
        if (rec.channels != GS_CHANNELS ||
            rec.height != GS_HEIGHT ||
            rec.width != GS_WIDTH) {

            printf("Errore: shape non valida alla riga %lu.\n", (unsigned long)line_number);
            printf("Trovato: %d x %d x %d\n", rec.channels, rec.height, rec.width);
            printf("Atteso:  %d x %d x %d\n", GS_CHANNELS, GS_HEIGHT, GS_WIDTH);

            gs_manifest_free(manifest);
            fclose(fp);
            return 1;
        }

        if (strcmp(rec.dtype, "uint8") != 0) {
            printf("Errore: dtype non valido alla riga %lu: %s\n",
                   (unsigned long)line_number,
                   rec.dtype);

            gs_manifest_free(manifest);
            fclose(fp);
            return 1;
        }

        if (strcmp(rec.layout, "channel_major") != 0) {
            printf("Errore: layout non valido alla riga %lu: %s\n",
                   (unsigned long)line_number,
                   rec.layout);

            gs_manifest_free(manifest);
            fclose(fp);
            return 1;
        }

        if (manifest->count == capacity) {
            GSManifestRecord *new_records;

            capacity *= 2;

            new_records = (GSManifestRecord *)realloc(
                manifest->records,
                capacity * sizeof(GSManifestRecord)
            );

            if (new_records == NULL) {
                printf("Errore: memoria insufficiente durante realloc.\n");

                gs_manifest_free(manifest);
                fclose(fp);
                return 1;
            }

            manifest->records = new_records;
        }

        manifest->records[manifest->count] = rec;
        manifest->count++;
    }

    fclose(fp);

    return 0;
}


const GSManifestRecord *gs_manifest_get(const GSManifest *manifest, size_t index)
{
    if (manifest == NULL) {
        return NULL;
    }

    if (index >= manifest->count) {
        return NULL;
    }

    return &manifest->records[index];
}


void gs_manifest_print_record(const GSManifestRecord *record)
{
    if (record == NULL) {
        printf("Record NULL.\n");
        return;
    }

    printf("event_index:   %d\n", record->event_index);
    printf("label:         %s\n", record->label);
    printf("event_id:      %s\n", record->event_id);
    printf("tensor_path:   %s\n", record->tensor_path);
    printf("shape:         %d x %d x %d\n",
           record->channels,
           record->height,
           record->width);
    printf("dtype:         %s\n", record->dtype);
    printf("layout:        %s\n", record->layout);
    printf("channel_order: %s\n", record->channel_order);
    printf("crop_left:     %d\n", record->crop_left);
}


int gs_manifest_load_tensor(const GSManifest *manifest, size_t index, float *out)
{
    const GSManifestRecord *record;

    record = gs_manifest_get(manifest, index);

    if (record == NULL) {
        printf("Errore: indice manifest non valido.\n");
        return 1;
    }

    return gs_load_bin_u8_as_float01(record->tensor_path, out);
}