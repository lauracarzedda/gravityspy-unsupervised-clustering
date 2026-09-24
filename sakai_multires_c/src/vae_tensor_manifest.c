#include "vae_tensor_manifest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>


static void gs_tensor_manifest_zero(
    GSTensorManifest *manifest
)
{
    if (manifest == NULL) {
        return;
    }

    manifest->records = NULL;
    manifest->count = 0;

    manifest->extra_column_name[0] = '\0';

    manifest->spec.channels = 0;
    manifest->spec.height = 0;
    manifest->spec.width = 0;
    manifest->spec.pixels_per_channel = 0;
    manifest->spec.values_per_event = 0;
    manifest->spec.bytes_u8 = 0;
    manifest->spec.bytes_f32 = 0;
}


static int gs_copy_string_checked(
    char *destination,
    size_t destination_size,
    const char *source
)
{
    size_t length;

    if (destination == NULL ||
        destination_size == 0 ||
        source == NULL) {

        return 1;
    }

    length = strlen(source);

    if (length >= destination_size) {
        return 1;
    }

    memcpy(
        destination,
        source,
        length + 1
    );

    return 0;
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

        memmove(
            line,
            line + 3,
            strlen(line + 3) + 1
        );
    }
}


/*
   Parser CSV semplice, compatibile con campi tra virgolette.
*/
static int gs_tensor_csv_split(
    char *line,
    char **fields,
    int max_fields
)
{
    char *p;
    int count;

    if (line == NULL ||
        fields == NULL ||
        max_fields <= 0) {

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


static int gs_parse_positive_int(
    const char *text,
    int *value
)
{
    char *endptr;
    long parsed;

    if (text == NULL ||
        value == NULL ||
        text[0] == '\0') {

        return 1;
    }

    endptr = NULL;

    parsed = strtol(
        text,
        &endptr,
        10
    );

    if (endptr == text ||
        *endptr != '\0' ||
        parsed < 1 ||
        parsed > INT_MAX) {

        return 1;
    }

    *value = (int)parsed;

    return 0;
}


static int gs_validate_header(
    char *line,
    char *extra_column_name,
    size_t extra_name_size
)
{
    const char *expected[10] = {
        "event_index",
        "label",
        "event_id",
        "tensor_path",
        "channels",
        "height",
        "width",
        "dtype",
        "layout",
        "channel_order"
    };

    char *fields[12];

    int count;
    int i;

    gs_strip_utf8_bom(line);

    count = gs_tensor_csv_split(
        line,
        fields,
        12
    );

    if (count != 11) {
        printf("Errore: header manifest tensoriale non valido.\n");
        printf("Colonne trovate: %d\n", count);
        printf("Colonne attese: 11\n");

        return 1;
    }

    for (i = 0; i < 10; i++) {
        if (strcmp(fields[i], expected[i]) != 0) {
            printf("Errore: colonna %d non valida.\n", i);
            printf("Trovata: %s\n", fields[i]);
            printf("Attesa:  %s\n", expected[i]);

            return 1;
        }
    }

    if (strcmp(fields[10], "crop_left") != 0 &&
        strcmp(fields[10], "origin") != 0) {

        printf("Errore: ultima colonna non riconosciuta.\n");
        printf("Trovata: %s\n", fields[10]);

        return 1;
    }

    if (gs_copy_string_checked(
            extra_column_name,
            extra_name_size,
            fields[10]
        ) != 0) {

        return 1;
    }

    return 0;
}


void gs_tensor_manifest_free(
    GSTensorManifest *manifest
)
{
    if (manifest == NULL) {
        return;
    }

    free(manifest->records);

    gs_tensor_manifest_zero(manifest);
}


static int gs_tensor_manifest_load_impl(
    const char *csv_path,
    const GSTensorSpec *expected_spec,
    GSTensorManifest *manifest,
    int label_free
)
{
    FILE *fp;

    char line[4096];
    char *fields[12];

    size_t capacity;
    size_t line_number;

    if (csv_path == NULL) {
        printf("Errore: csv_path NULL.\n");
        return 1;
    }

    if (!gs_tensor_spec_is_valid(expected_spec)) {
        printf("Errore: expected_spec non valida.\n");
        return 1;
    }

    if (manifest == NULL) {
        printf("Errore: manifest NULL.\n");
        return 1;
    }

    gs_tensor_manifest_zero(manifest);

    fp = fopen(csv_path, "r");

    if (fp == NULL) {
        printf("Errore: impossibile aprire:\n%s\n",
               csv_path);

        return 1;
    }

    if (fgets(line, sizeof(line), fp) == NULL) {
        printf("Errore: manifest vuoto.\n");
        fclose(fp);

        return 1;
    }

    if (gs_validate_header(
            line,
            manifest->extra_column_name,
            sizeof(manifest->extra_column_name)
        ) != 0) {

        fclose(fp);
        return 1;
    }

    manifest->spec = *expected_spec;

    capacity = 1024;

    manifest->records =
        (GSTensorManifestRecord *)malloc(
            capacity *
            sizeof(GSTensorManifestRecord)
        );

    if (manifest->records == NULL) {
        printf("Errore: memoria insufficiente.\n");
        fclose(fp);

        return 1;
    }

    line_number = 1;

    while (fgets(line, sizeof(line), fp) != NULL) {
        GSTensorManifestRecord record;

        int field_count;

        line_number++;

        if (line[0] == '\0' ||
            line[0] == '\n' ||
            line[0] == '\r') {

            continue;
        }

        memset(
            &record,
            0,
            sizeof(record)
        );

        field_count = gs_tensor_csv_split(
            line,
            fields,
            12
        );

        if (field_count != 11) {
            printf("Errore CSV alla riga %lu.\n",
                   (unsigned long)line_number);

            gs_tensor_manifest_free(manifest);
            fclose(fp);

            return 1;
        }

        if (gs_parse_positive_int(
                fields[0],
                &record.event_index
            ) != 0 ||

            gs_parse_positive_int(
                fields[4],
                &record.channels
            ) != 0 ||

            gs_parse_positive_int(
                fields[5],
                &record.height
            ) != 0 ||

            gs_parse_positive_int(
                fields[6],
                &record.width
            ) != 0) {

            printf("Errore: campo numerico non valido alla riga %lu.\n",
                   (unsigned long)line_number);

            gs_tensor_manifest_free(manifest);
            fclose(fp);

            return 1;
        }

        if (record.channels != expected_spec->channels ||
            record.height != expected_spec->height ||
            record.width != expected_spec->width) {

            printf("Errore: shape non valida alla riga %lu.\n",
                   (unsigned long)line_number);

            printf("Trovata: %d x %d x %d\n",
                   record.channels,
                   record.height,
                   record.width);

            printf("Attesa:  %d x %d x %d\n",
                   expected_spec->channels,
                   expected_spec->height,
                   expected_spec->width);

            gs_tensor_manifest_free(manifest);
            fclose(fp);

            return 1;
        }

        if ((!label_free && gs_copy_string_checked(
                record.label,
                sizeof(record.label),
                fields[1]
            ) != 0) ||

            gs_copy_string_checked(
                record.event_id,
                sizeof(record.event_id),
                fields[2]
            ) != 0 ||

            gs_copy_string_checked(
                record.tensor_path,
                sizeof(record.tensor_path),
                fields[3]
            ) != 0 ||

            gs_copy_string_checked(
                record.dtype,
                sizeof(record.dtype),
                fields[7]
            ) != 0 ||

            gs_copy_string_checked(
                record.layout,
                sizeof(record.layout),
                fields[8]
            ) != 0 ||

            gs_copy_string_checked(
                record.channel_order,
                sizeof(record.channel_order),
                fields[9]
            ) != 0 ||

            gs_copy_string_checked(
                record.extra_value,
                sizeof(record.extra_value),
                fields[10]
            ) != 0) {

            printf("Errore: stringa troppo lunga alla riga %lu.\n",
                   (unsigned long)line_number);

            gs_tensor_manifest_free(manifest);
            fclose(fp);

            return 1;
        }

        if (strcmp(record.dtype, "uint8") != 0) {
            printf("Errore: dtype non valido alla riga %lu.\n",
                   (unsigned long)line_number);

            gs_tensor_manifest_free(manifest);
            fclose(fp);

            return 1;
        }

        if (strcmp(record.layout, "channel_major") != 0) {
            printf("Errore: layout non valido alla riga %lu.\n",
                   (unsigned long)line_number);

            gs_tensor_manifest_free(manifest);
            fclose(fp);

            return 1;
        }

        if (strcmp(
                record.channel_order,
                "0.5;1.0;2.0;4.0"
            ) != 0) {

            printf("Errore: ordine canali non valido alla riga %lu.\n",
                   (unsigned long)line_number);

            gs_tensor_manifest_free(manifest);
            fclose(fp);

            return 1;
        }

        if (manifest->count == capacity) {
            GSTensorManifestRecord *new_records;

            capacity *= 2;

            new_records =
                (GSTensorManifestRecord *)realloc(
                    manifest->records,
                    capacity *
                    sizeof(GSTensorManifestRecord)
                );

            if (new_records == NULL) {
                printf("Errore: realloc fallita.\n");

                gs_tensor_manifest_free(manifest);
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
        printf("Errore: nessun record trovato.\n");
        gs_tensor_manifest_free(manifest);

        return 1;
    }

    return 0;
}


const GSTensorManifestRecord *gs_tensor_manifest_get(
    const GSTensorManifest *manifest,
    size_t index
)
{
    if (manifest == NULL ||
        index >= manifest->count) {

        return NULL;
    }

    return &manifest->records[index];
}


int gs_tensor_manifest_load(
    const char *csv_path,
    const GSTensorSpec *expected_spec,
    GSTensorManifest *manifest
)
{
    return gs_tensor_manifest_load_impl(csv_path, expected_spec, manifest, 0);
}


int gs_tensor_manifest_load_label_free(
    const char *csv_path,
    const GSTensorSpec *expected_spec,
    GSTensorManifest *manifest
)
{
    return gs_tensor_manifest_load_impl(csv_path, expected_spec, manifest, 1);
}
