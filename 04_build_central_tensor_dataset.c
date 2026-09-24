#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

/*
 * Costruzione del dataset tensoriale centrale.
 *
 * Input:
 *   manifest_events_272.csv
 *   out272_gray/
 *
 * Per ogni evento:
 *   1. legge le quattro viste temporali 272x224:
 *      0.5 s, 1.0 s, 2.0 s, 4.0 s
 *
 *   2. considera le quattro viste come stack:
 *      4 x 224 x 272
 *
 *   3. applica il crop temporale centrale:
 *      4 x 224 x 272 -> 4 x 224 x 224
 *
 *   4. salva un file binario uint8:
 *      4 x 224 x 224 valori
 *
 * Questo dataset rappresenta la versione "Input Image" non perturbata.
 * Le immagini perturbate con shift casuale 0-24 px verranno generate
 * successivamente durante il training, partendo ancora da out272_gray.
 */

#define INPUT_W 272
#define INPUT_H 224

#define FINAL_W 224
#define FINAL_H 224

#define NUM_CHANNELS 4

#define CENTRAL_LEFT 24

#define MAX_LINE_LEN 8192
#define MAX_FIELDS 16
#define MAX_FIELD_LEN 4096
#define MAX_PATH_LEN 2048


void remove_newline_and_bom(char *s)
{
    int n;

    if (s == NULL) {
        return;
    }

    if ((unsigned char)s[0] == 0xEF &&
        (unsigned char)s[1] == 0xBB &&
        (unsigned char)s[2] == 0xBF) {
        memmove(s, s + 3, strlen(s + 3) + 1);
    }

    n = (int)strlen(s);

    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r')) {
        s[n - 1] = '\0';
        n--;
    }
}


int parse_csv_line(const char *line, char fields[MAX_FIELDS][MAX_FIELD_LEN])
{
    int i;
    int field_index;
    int pos;
    int in_quotes;
    char ch;

    field_index = 0;
    pos = 0;
    in_quotes = 0;

    for (i = 0; line[i] != '\0'; i++) {
        ch = line[i];

        if (ch == '"') {
            if (in_quotes && line[i + 1] == '"') {
                if (pos < MAX_FIELD_LEN - 1) {
                    fields[field_index][pos] = '"';
                    pos++;
                }

                i++;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (ch == ',' && !in_quotes) {
            fields[field_index][pos] = '\0';
            field_index++;

            if (field_index >= MAX_FIELDS) {
                return field_index;
            }

            pos = 0;
        } else if ((ch == '\n' || ch == '\r') && !in_quotes) {
            break;
        } else {
            if (pos < MAX_FIELD_LEN - 1) {
                fields[field_index][pos] = ch;
                pos++;
            }
        }
    }

    fields[field_index][pos] = '\0';

    return field_index + 1;
}


void csv_write_field(FILE *f, const char *s)
{
    const char *p;

    fputc('"', f);

    p = s;

    while (*p != '\0') {
        if (*p == '"') {
            fputc('"', f);
        }

        fputc(*p, f);
        p++;
    }

    fputc('"', f);
}


int load_gray_image_272(const char *path, unsigned char **out_image)
{
    int w;
    int h;
    int channels;
    unsigned char *img;

    img = stbi_load(path, &w, &h, &channels, 1);

    if (img == NULL) {
        printf("Errore: impossibile leggere immagine:\n%s\n", path);
        printf("Motivo STB: %s\n", stbi_failure_reason());
        return 0;
    }

    if (w != INPUT_W || h != INPUT_H) {
        printf("Errore: dimensione inattesa per:\n%s\n", path);
        printf("Attesa: %dx%d, trovata: %dx%d\n", INPUT_W, INPUT_H, w, h);
        stbi_image_free(img);
        return 0;
    }

    *out_image = img;

    return 1;
}


void crop_channel_to_tensor(
    const unsigned char *src,
    unsigned char *tensor,
    int channel_index
)
{
    int x;
    int y;
    int src_x;
    int src_y;
    int dst_index;

    /*
     * Layout del tensore salvato:
     *
     * canale 0: 0.5 s
     * canale 1: 1.0 s
     * canale 2: 2.0 s
     * canale 3: 4.0 s
     *
     * Ogni canale è salvato come matrice 224x224.
     */
    for (y = 0; y < FINAL_H; y++) {
        for (x = 0; x < FINAL_W; x++) {
            src_x = CENTRAL_LEFT + x;
            src_y = y;

            dst_index =
                channel_index * FINAL_H * FINAL_W
                + y * FINAL_W
                + x;

            tensor[dst_index] = src[src_y * INPUT_W + src_x];
        }
    }
}


int write_tensor_file(const char *path, const unsigned char *tensor, size_t tensor_size)
{
    FILE *f;
    size_t written;

    f = fopen(path, "wb");

    if (f == NULL) {
        return 0;
    }

    written = fwrite(tensor, 1, tensor_size, f);

    fclose(f);

    if (written != tensor_size) {
        return 0;
    }

    return 1;
}


int process_event(
    char fields[MAX_FIELDS][MAX_FIELD_LEN],
    FILE *manifest_out
)
{
    unsigned char *views[NUM_CHANNELS];
    unsigned char *tensor;

    char tensor_path[MAX_PATH_LEN];

    int event_index;
    int i;
    int ok;

    size_t tensor_size;

    /*
     * Manifest atteso:
     * 0 event_index
     * 1 label
     * 2 event_id
     * 3 img_0_5_272
     * 4 img_1_0_272
     * 5 img_2_0_272
     * 6 img_4_0_272
     */

    event_index = atoi(fields[0]);

    for (i = 0; i < NUM_CHANNELS; i++) {
        views[i] = NULL;
    }

    tensor_size = (size_t)NUM_CHANNELS * FINAL_H * FINAL_W;

    tensor = (unsigned char *)malloc(tensor_size);

    if (tensor == NULL) {
        printf("Errore memoria tensor per evento %s\n", fields[0]);
        return 0;
    }

    /*
     * Carica le quattro viste temporali.
     */
    for (i = 0; i < NUM_CHANNELS; i++) {
        if (!load_gray_image_272(fields[3 + i], &views[i])) {
            free(tensor);

            for (i = 0; i < NUM_CHANNELS; i++) {
                if (views[i] != NULL) {
                    stbi_image_free(views[i]);
                }
            }

            return 0;
        }
    }

    /*
     * Crop centrale identico su tutti i canali.
     */
    for (i = 0; i < NUM_CHANNELS; i++) {
        crop_channel_to_tensor(views[i], tensor, i);
    }

    sprintf(
        tensor_path,
        "tensors_224_central_uint8/event_%06d.bin",
        event_index
    );

    ok = write_tensor_file(tensor_path, tensor, tensor_size);

    if (!ok) {
        printf("Errore salvataggio tensor: %s\n", tensor_path);
    } else {
        fprintf(manifest_out, "%d,", event_index);

        csv_write_field(manifest_out, fields[1]);
        fputc(',', manifest_out);

        csv_write_field(manifest_out, fields[2]);
        fputc(',', manifest_out);

        csv_write_field(manifest_out, tensor_path);
        fputc(',', manifest_out);

        fprintf(
            manifest_out,
            "%d,%d,%d,uint8,channel_major,0.5;1.0;2.0;4.0,%d\n",
            NUM_CHANNELS,
            FINAL_H,
            FINAL_W,
            CENTRAL_LEFT
        );

        if (event_index == 1 || event_index % 500 == 0) {
            printf("OK event_%06d  label=%s  event_id=%s\n",
                   event_index,
                   fields[1],
                   fields[2]);
        }
    }

    for (i = 0; i < NUM_CHANNELS; i++) {
        if (views[i] != NULL) {
            stbi_image_free(views[i]);
        }
    }

    free(tensor);

    return ok;
}


int main(int argc, char **argv)
{
    const char *events_manifest_path;

    FILE *events_manifest;
    FILE *tensor_manifest;

    char line[MAX_LINE_LEN];
    char fields[MAX_FIELDS][MAX_FIELD_LEN];

    int row_count;
    int event_count;
    int success_count;
    int bad_rows;
    int n_fields;

    if (argc < 2) {
        printf("Uso:\n");
        printf("  04_build_central_tensor_dataset.exe manifest_events_272.csv\n");
        return 1;
    }

    events_manifest_path = argv[1];

    events_manifest = fopen(events_manifest_path, "r");

    if (events_manifest == NULL) {
        printf("Errore: impossibile aprire manifest eventi:\n%s\n", events_manifest_path);
        return 1;
    }

    tensor_manifest = fopen("manifest_tensors_224_central.csv", "w");

    if (tensor_manifest == NULL) {
        printf("Errore: impossibile creare manifest_tensors_224_central.csv\n");
        fclose(events_manifest);
        return 1;
    }

    fprintf(
        tensor_manifest,
        "event_index,label,event_id,tensor_path,channels,height,width,dtype,layout,channel_order,crop_left\n"
    );

    row_count = 0;
    event_count = 0;
    success_count = 0;
    bad_rows = 0;

    while (fgets(line, MAX_LINE_LEN, events_manifest) != NULL) {
        remove_newline_and_bom(line);

        if (strlen(line) == 0) {
            continue;
        }

        row_count++;

        /*
         * Salta intestazione.
         */
        if (row_count == 1) {
            continue;
        }

        n_fields = parse_csv_line(line, fields);

        if (n_fields < 7) {
            bad_rows++;
            continue;
        }

        event_count++;

        if (process_event(fields, tensor_manifest)) {
            success_count++;
        } else {
            bad_rows++;
        }
    }

    fclose(tensor_manifest);
    fclose(events_manifest);

    printf("\nCostruzione dataset tensoriale completata.\n");
    printf("Eventi letti: %d\n", event_count);
    printf("Tensor salvati: %d\n", success_count);
    printf("Righe problematiche: %d\n", bad_rows);
    printf("Manifest creato: manifest_tensors_224_central.csv\n");

    return 0;
}