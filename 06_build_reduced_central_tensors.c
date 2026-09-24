#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Costruzione delle versioni ridotte del dataset centrale.
 *
 * Input:
 *   manifest_tensors_224_central.csv
 *   tensors_224_central_uint8/
 *
 * Per ogni evento:
 *   1. legge il tensore centrale 4x224x224 uint8;
 *   2. ridimensiona ogni canale separatamente;
 *   3. salva:
 *        4x112x112 in tensors_112_central_uint8/
 *        4x75x75   in tensors_75_central_uint8/
 *
 * Queste versioni ridotte sono derivate dal tensore centrale già costruito
 * dopo stacking e crop temporale centrale.
 */

#define SRC_W 224
#define SRC_H 224

#define DST112_W 112
#define DST112_H 112

#define DST75_W 75
#define DST75_H 75

#define NUM_CHANNELS 4

#define SRC_TENSOR_SIZE (NUM_CHANNELS * SRC_W * SRC_H)
#define DST112_TENSOR_SIZE (NUM_CHANNELS * DST112_W * DST112_H)
#define DST75_TENSOR_SIZE (NUM_CHANNELS * DST75_W * DST75_H)

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


int read_binary_file(const char *path, unsigned char *buffer, size_t expected_size)
{
    FILE *f;
    size_t n_read;

    f = fopen(path, "rb");

    if (f == NULL) {
        return 0;
    }

    n_read = fread(buffer, 1, expected_size, f);

    fclose(f);

    if (n_read != expected_size) {
        return 0;
    }

    return 1;
}


int write_binary_file(const char *path, const unsigned char *buffer, size_t size)
{
    FILE *f;
    size_t n_written;

    f = fopen(path, "wb");

    if (f == NULL) {
        return 0;
    }

    n_written = fwrite(buffer, 1, size, f);

    fclose(f);

    if (n_written != size) {
        return 0;
    }

    return 1;
}


void resize_bilinear_gray_channel(
    const unsigned char *src,
    int src_w,
    int src_h,
    unsigned char *dst,
    int dst_w,
    int dst_h
)
{
    int x;
    int y;
    int x0;
    int x1;
    int y0;
    int y1;

    float gx;
    float gy;
    float dx;
    float dy;

    float p00;
    float p10;
    float p01;
    float p11;

    float top_value;
    float bottom_value;
    float value;

    for (y = 0; y < dst_h; y++) {
        if (dst_h > 1) {
            gy = ((float)y * (float)(src_h - 1)) / (float)(dst_h - 1);
        } else {
            gy = 0.0f;
        }

        y0 = (int)gy;
        y1 = y0 + 1;

        if (y1 >= src_h) {
            y1 = src_h - 1;
        }

        dy = gy - (float)y0;

        for (x = 0; x < dst_w; x++) {
            if (dst_w > 1) {
                gx = ((float)x * (float)(src_w - 1)) / (float)(dst_w - 1);
            } else {
                gx = 0.0f;
            }

            x0 = (int)gx;
            x1 = x0 + 1;

            if (x1 >= src_w) {
                x1 = src_w - 1;
            }

            dx = gx - (float)x0;

            p00 = (float)src[y0 * src_w + x0];
            p10 = (float)src[y0 * src_w + x1];
            p01 = (float)src[y1 * src_w + x0];
            p11 = (float)src[y1 * src_w + x1];

            top_value = p00 * (1.0f - dx) + p10 * dx;
            bottom_value = p01 * (1.0f - dx) + p11 * dx;
            value = top_value * (1.0f - dy) + bottom_value * dy;

            if (value < 0.0f) {
                value = 0.0f;
            }

            if (value > 255.0f) {
                value = 255.0f;
            }

            dst[y * dst_w + x] = (unsigned char)(value + 0.5f);
        }
    }
}


void resize_tensor(
    const unsigned char *src_tensor,
    unsigned char *dst_tensor,
    int dst_w,
    int dst_h
)
{
    int c;

    const unsigned char *src_channel;
    unsigned char *dst_channel;

    for (c = 0; c < NUM_CHANNELS; c++) {
        src_channel = src_tensor + c * SRC_W * SRC_H;
        dst_channel = dst_tensor + c * dst_w * dst_h;

        resize_bilinear_gray_channel(
            src_channel,
            SRC_W,
            SRC_H,
            dst_channel,
            dst_w,
            dst_h
        );
    }
}


int process_one_tensor(
    char fields[MAX_FIELDS][MAX_FIELD_LEN],
    FILE *manifest112,
    FILE *manifest75
)
{
    int event_index;
    int ok112;
    int ok75;

    char out112_path[MAX_PATH_LEN];
    char out75_path[MAX_PATH_LEN];

    unsigned char *src_tensor;
    unsigned char *tensor112;
    unsigned char *tensor75;

    /*
     * Manifest input atteso:
     * 0 event_index
     * 1 label
     * 2 event_id
     * 3 tensor_path
     * 4 channels
     * 5 height
     * 6 width
     * ...
     */

    event_index = atoi(fields[0]);

    src_tensor = (unsigned char *)malloc((size_t)SRC_TENSOR_SIZE);
    tensor112 = (unsigned char *)malloc((size_t)DST112_TENSOR_SIZE);
    tensor75 = (unsigned char *)malloc((size_t)DST75_TENSOR_SIZE);

    if (src_tensor == NULL || tensor112 == NULL || tensor75 == NULL) {
        printf("Errore memoria per evento %d\n", event_index);

        if (src_tensor != NULL) free(src_tensor);
        if (tensor112 != NULL) free(tensor112);
        if (tensor75 != NULL) free(tensor75);

        return 0;
    }

    if (!read_binary_file(fields[3], src_tensor, (size_t)SRC_TENSOR_SIZE)) {
        printf("Errore lettura tensor 224: %s\n", fields[3]);

        free(src_tensor);
        free(tensor112);
        free(tensor75);

        return 0;
    }

    resize_tensor(src_tensor, tensor112, DST112_W, DST112_H);
    resize_tensor(src_tensor, tensor75, DST75_W, DST75_H);

    sprintf(
        out112_path,
        "tensors_112_central_uint8/event_%06d.bin",
        event_index
    );

    sprintf(
        out75_path,
        "tensors_75_central_uint8/event_%06d.bin",
        event_index
    );

    ok112 = write_binary_file(
        out112_path,
        tensor112,
        (size_t)DST112_TENSOR_SIZE
    );

    ok75 = write_binary_file(
        out75_path,
        tensor75,
        (size_t)DST75_TENSOR_SIZE
    );

    if (!ok112 || !ok75) {
        printf("Errore salvataggio versioni ridotte evento %d\n", event_index);

        free(src_tensor);
        free(tensor112);
        free(tensor75);

        return 0;
    }

    fprintf(manifest112, "%d,", event_index);
    csv_write_field(manifest112, fields[1]);
    fputc(',', manifest112);
    csv_write_field(manifest112, fields[2]);
    fputc(',', manifest112);
    csv_write_field(manifest112, out112_path);
    fputc(',', manifest112);
    fprintf(
        manifest112,
        "%d,%d,%d,uint8,channel_major,0.5;1.0;2.0;4.0,derived_from_224_central\n",
        NUM_CHANNELS,
        DST112_H,
        DST112_W
    );

    fprintf(manifest75, "%d,", event_index);
    csv_write_field(manifest75, fields[1]);
    fputc(',', manifest75);
    csv_write_field(manifest75, fields[2]);
    fputc(',', manifest75);
    csv_write_field(manifest75, out75_path);
    fputc(',', manifest75);
    fprintf(
        manifest75,
        "%d,%d,%d,uint8,channel_major,0.5;1.0;2.0;4.0,derived_from_224_central\n",
        NUM_CHANNELS,
        DST75_H,
        DST75_W
    );

    if (event_index == 1 || event_index % 500 == 0) {
        printf("OK event_%06d  reduced 112 + 75\n", event_index);
    }

    free(src_tensor);
    free(tensor112);
    free(tensor75);

    return 1;
}


int main(int argc, char **argv)
{
    const char *manifest224_path;

    FILE *manifest224;
    FILE *manifest112;
    FILE *manifest75;

    char line[MAX_LINE_LEN];
    char fields[MAX_FIELDS][MAX_FIELD_LEN];

    int row_count;
    int event_count;
    int success_count;
    int bad_rows;
    int n_fields;

    if (argc < 2) {
        printf("Uso:\n");
        printf("  06_build_reduced_central_tensors.exe manifest_tensors_224_central.csv\n");
        return 1;
    }

    manifest224_path = argv[1];

    manifest224 = fopen(manifest224_path, "r");

    if (manifest224 == NULL) {
        printf("Errore: impossibile aprire manifest 224:\n%s\n", manifest224_path);
        return 1;
    }

    manifest112 = fopen("manifest_tensors_112_central.csv", "w");

    if (manifest112 == NULL) {
        printf("Errore: impossibile creare manifest_tensors_112_central.csv\n");
        fclose(manifest224);
        return 1;
    }

    manifest75 = fopen("manifest_tensors_75_central.csv", "w");

    if (manifest75 == NULL) {
        printf("Errore: impossibile creare manifest_tensors_75_central.csv\n");
        fclose(manifest112);
        fclose(manifest224);
        return 1;
    }

    fprintf(
        manifest112,
        "event_index,label,event_id,tensor_path,channels,height,width,dtype,layout,channel_order,origin\n"
    );

    fprintf(
        manifest75,
        "event_index,label,event_id,tensor_path,channels,height,width,dtype,layout,channel_order,origin\n"
    );

    row_count = 0;
    event_count = 0;
    success_count = 0;
    bad_rows = 0;

    while (fgets(line, MAX_LINE_LEN, manifest224) != NULL) {
        remove_newline_and_bom(line);

        if (strlen(line) == 0) {
            continue;
        }

        row_count++;

        if (row_count == 1) {
            continue;
        }

        n_fields = parse_csv_line(line, fields);

        if (n_fields < 4) {
            bad_rows++;
            continue;
        }

        event_count++;

        if (process_one_tensor(fields, manifest112, manifest75)) {
            success_count++;
        } else {
            bad_rows++;
        }
    }

    fclose(manifest75);
    fclose(manifest112);
    fclose(manifest224);

    printf("\nCostruzione versioni ridotte completata.\n");
    printf("Eventi letti: %d\n", event_count);
    printf("Eventi ridotti salvati: %d\n", success_count);
    printf("Righe problematiche: %d\n", bad_rows);
    printf("Manifest creati:\n");
    printf("manifest_tensors_112_central.csv\n");
    printf("manifest_tensors_75_central.csv\n");

    return 0;
}
