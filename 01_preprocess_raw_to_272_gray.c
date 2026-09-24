#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/*
 * Preprocessing ufficiale delle PNG grezze Gravity Spy.
 *
 * Per ogni PNG:
 * 1. Lettura immagine originale 800x600.
 * 2. Crop fisso dello spettrogramma con coordinate Zenodo.
 * 3. Resize bilineare a 272x224.
 * 4. Conversione RGB -> grayscale.
 * 5. Salvataggio dell'immagine 272x224 monocanale.
 *
 * Questa fase produce le singole viste temporali nella forma precedente
 * allo stacking. Successivamente, le quattro viste 0.5, 1.0, 2.0 e 4.0 s
 * dello stesso evento verranno impilate per costruire il tensore 4x224x272.
 */

/* Coordinate crop Zenodo: righe 66:532, colonne 105:671 */
#define CROP_TOP     66
#define CROP_BOTTOM 532
#define CROP_LEFT   105
#define CROP_RIGHT  671

#define CROP_W (CROP_RIGHT - CROP_LEFT)
#define CROP_H (CROP_BOTTOM - CROP_TOP)

/*
 * In C/Windows le dimensioni sono indicate come larghezza x altezza.
 * 272x224 corrisponde al 224x272 del paper, dove 224 è la direzione
 * in frequenza e 272 la direzione temporale.
 */
#define RESIZED_W 272
#define RESIZED_H 224

#define MAX_PATH_LEN 2048
#define MAX_FIELD_LEN 512

#ifdef _WIN32
#define NATIVE_PATH_SEPARATOR '\\'
#else
#define NATIVE_PATH_SEPARATOR '/'
#endif


int is_path_separator(char c)
{
    return c == '/' || c == '\\';
}


int has_windows_drive_prefix(const char *path)
{
    if (path == NULL || path[0] == '\0') {
        return 0;
    }

    return isalpha((unsigned char)path[0]) && path[1] == ':';
}


int is_absolute_path(const char *path)
{
    if (path == NULL || path[0] == '\0') {
        return 0;
    }

    if (is_path_separator(path[0])) {
        return 1;
    }

    return has_windows_drive_prefix(path) && is_path_separator(path[2]);
}


int is_valid_relative_path(const char *path)
{
    const char *component;
    const char *p;
    size_t component_len;
    unsigned char c;

    if (path == NULL || path[0] == '\0' || is_absolute_path(path)) {
        return 0;
    }

    component = path;
    p = path;

    for (;;) {
        c = (unsigned char)*p;

        if (c != '\0' && (c < 32 || c == 127 || c == ':')) {
            return 0;
        }

        if (c == '\0' || is_path_separator((char)c)) {
            component_len = (size_t)(p - component);

            if (component_len == 0) {
                return 0;
            }

            if ((component_len == 1 && component[0] == '.') ||
                (component_len == 2 && component[0] == '.' &&
                 component[1] == '.')) {
                return 0;
            }

            if (c == '\0') {
                break;
            }

            component = p + 1;
        }

        p++;
    }

    return 1;
}


int dirname_from_absolute_path(const char *path, char *out, size_t out_size)
{
    const char *p;
    const char *last_separator;
    size_t len;

    if (!is_absolute_path(path) || out == NULL || out_size == 0) {
        return 0;
    }

    last_separator = NULL;

    for (p = path; *p != '\0'; p++) {
        if (is_path_separator(*p)) {
            last_separator = p;
        }
    }

    if (last_separator == NULL) {
        return 0;
    }

    len = (size_t)(last_separator - path);

    if (last_separator == path ||
        (last_separator == path + 2 && has_windows_drive_prefix(path))) {
        len++;
    }

    if (len + 1 > out_size) {
        return 0;
    }

    memcpy(out, path, len);
    out[len] = '\0';

    return 1;
}


int join_path(
    char *out,
    size_t out_size,
    const char *base,
    const char *relative
)
{
    size_t base_len;
    size_t relative_len;
    size_t pos;
    size_t i;
    int add_separator;

    if (out == NULL || out_size == 0 || base == NULL || relative == NULL) {
        return 0;
    }

    base_len = strlen(base);
    relative_len = strlen(relative);

    if (base_len == 0 || relative_len == 0) {
        return 0;
    }

    add_separator = !is_path_separator(base[base_len - 1]);

    if (base_len >= out_size ||
        (size_t)add_separator >= out_size - base_len ||
        relative_len >= out_size - base_len - (size_t)add_separator) {
        return 0;
    }

    memcpy(out, base, base_len);
    pos = base_len;

    if (add_separator) {
        out[pos++] = NATIVE_PATH_SEPARATOR;
    }

    for (i = 0; i < relative_len; i++) {
        out[pos++] = is_path_separator(relative[i])
            ? NATIVE_PATH_SEPARATOR
            : relative[i];
    }

    out[pos] = '\0';

    return 1;
}


int join_manifest_path(
    char *out,
    size_t out_size,
    const char *directory,
    const char *filename
)
{
    size_t directory_len;
    size_t filename_len;
    size_t i;
    size_t pos;

    if (out == NULL || out_size == 0 ||
        directory == NULL || filename == NULL) {
        return 0;
    }

    directory_len = strlen(directory);
    filename_len = strlen(filename);

    if (directory_len == 0 || filename_len == 0 ||
        directory_len >= out_size ||
        out_size - directory_len < 2 ||
        filename_len > out_size - directory_len - 2) {
        return 0;
    }

    pos = 0;

    for (i = 0; i < directory_len; i++) {
        out[pos++] = is_path_separator(directory[i]) ? '/' : directory[i];
    }

    out[pos++] = '/';
    memcpy(out + pos, filename, filename_len);
    pos += filename_len;
    out[pos] = '\0';

    return 1;
}


int format_output_filename(char *out, size_t out_size, int index)
{
    int written;

    if (out == NULL || out_size == 0) {
        return 0;
    }

    written = snprintf(out, out_size, "img_%06d.png", index);

    return written >= 0 && (size_t)written < out_size;
}


int read_list_line(FILE *f, char *out, size_t out_size)
{
    int ch;
    size_t len;

    if (f == NULL || out == NULL || out_size == 0) {
        return -2;
    }

    len = 0;

    for (;;) {
        ch = fgetc(f);

        if (ch == EOF) {
            if (len == 0) {
                return 0;
            }

            break;
        }

        if (ch == '\n') {
            break;
        }

        if (ch == 0 || (ch < 32 && ch != '\r') || ch == 127) {
            out[0] = '\0';
            return -2;
        }

        if (len + 1 >= out_size) {
            do {
                ch = fgetc(f);
            } while (ch != EOF && ch != '\n');

            out[0] = '\0';
            return -1;
        }

        out[len++] = (char)ch;
    }

    if (len > 0 && out[len - 1] == '\r') {
        len--;
    }

    out[len] = '\0';

    return 1;
}


void remove_newline_and_bom(char *s)
{
    int n;

    if (s == NULL) {
        return;
    }

    n = (int)strlen(s);

    if (n >= 3 &&
        (unsigned char)s[0] == 0xEF &&
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


const char *basename_ptr(const char *path)
{
    const char *p1;
    const char *p2;
    const char *p;

    p1 = strrchr(path, '\\');
    p2 = strrchr(path, '/');

    if (p1 == NULL && p2 == NULL) {
        return path;
    }

    if (p1 == NULL) {
        return p2 + 1;
    }

    if (p2 == NULL) {
        return p1 + 1;
    }

    p = (p1 > p2) ? p1 : p2;

    return p + 1;
}


void extract_parent_folder(const char *path, char *out, int out_size)
{
    char temp[MAX_PATH_LEN];
    char *p1;
    char *p2;
    char *p;
    const char *parent;

    strncpy(temp, path, MAX_PATH_LEN - 1);
    temp[MAX_PATH_LEN - 1] = '\0';

    p1 = strrchr(temp, '\\');
    p2 = strrchr(temp, '/');

    if (p1 == NULL && p2 == NULL) {
        strncpy(out, "unknown", out_size - 1);
        out[out_size - 1] = '\0';
        return;
    }

    if (p1 == NULL) {
        p = p2;
    } else if (p2 == NULL) {
        p = p1;
    } else {
        p = (p1 > p2) ? p1 : p2;
    }

    *p = '\0';

    parent = basename_ptr(temp);

    strncpy(out, parent, out_size - 1);
    out[out_size - 1] = '\0';
}


void parse_event_and_duration(
    const char *path,
    char *event_id,
    int event_size,
    char *duration,
    int duration_size
)
{
    const char *base;
    char filename[MAX_FIELD_LEN];
    char *dot;
    char *marker;
    int len;

    base = basename_ptr(path);

    strncpy(filename, base, MAX_FIELD_LEN - 1);
    filename[MAX_FIELD_LEN - 1] = '\0';

    dot = strrchr(filename, '.');

    if (dot != NULL) {
        *dot = '\0';
    }

    marker = strstr(filename, "_spectrogram_");

    if (marker != NULL) {
        len = (int)(marker - filename);

        if (len >= event_size) {
            len = event_size - 1;
        }

        strncpy(event_id, filename, len);
        event_id[len] = '\0';

        strncpy(duration, marker + strlen("_spectrogram_"), duration_size - 1);
        duration[duration_size - 1] = '\0';
    } else {
        strncpy(event_id, filename, event_size - 1);
        event_id[event_size - 1] = '\0';

        strncpy(duration, "unknown", duration_size - 1);
        duration[duration_size - 1] = '\0';
    }
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


unsigned char *crop_rgb(const unsigned char *src, int src_w, int src_h)
{
    unsigned char *dst;
    int x;
    int y;
    int c;

    if (src_w < CROP_RIGHT || src_h < CROP_BOTTOM) {
        return NULL;
    }

    dst = (unsigned char *)malloc((size_t)CROP_W * CROP_H * 3);

    if (dst == NULL) {
        return NULL;
    }

    for (y = 0; y < CROP_H; y++) {
        for (x = 0; x < CROP_W; x++) {
            for (c = 0; c < 3; c++) {
                dst[(y * CROP_W + x) * 3 + c] =
                    src[((CROP_TOP + y) * src_w + (CROP_LEFT + x)) * 3 + c];
            }
        }
    }

    return dst;
}


unsigned char *resize_bilinear_rgb(
    const unsigned char *src,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h
)
{
    unsigned char *dst;
    int x;
    int y;
    int c;
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

    dst = (unsigned char *)malloc((size_t)dst_w * dst_h * 3);

    if (dst == NULL) {
        return NULL;
    }

    for (y = 0; y < dst_h; y++) {
        gy = ((float)y * (float)(src_h - 1)) / (float)(dst_h - 1);
        y0 = (int)gy;
        y1 = y0 + 1;

        if (y1 >= src_h) {
            y1 = src_h - 1;
        }

        dy = gy - (float)y0;

        for (x = 0; x < dst_w; x++) {
            gx = ((float)x * (float)(src_w - 1)) / (float)(dst_w - 1);
            x0 = (int)gx;
            x1 = x0 + 1;

            if (x1 >= src_w) {
                x1 = src_w - 1;
            }

            dx = gx - (float)x0;

            for (c = 0; c < 3; c++) {
                p00 = (float)src[(y0 * src_w + x0) * 3 + c];
                p10 = (float)src[(y0 * src_w + x1) * 3 + c];
                p01 = (float)src[(y1 * src_w + x0) * 3 + c];
                p11 = (float)src[(y1 * src_w + x1) * 3 + c];

                top_value = p00 * (1.0f - dx) + p10 * dx;
                bottom_value = p01 * (1.0f - dx) + p11 * dx;
                value = top_value * (1.0f - dy) + bottom_value * dy;

                if (value < 0.0f) {
                    value = 0.0f;
                }

                if (value > 255.0f) {
                    value = 255.0f;
                }

                dst[(y * dst_w + x) * 3 + c] = (unsigned char)(value + 0.5f);
            }
        }
    }

    return dst;
}


unsigned char *rgb_to_gray(const unsigned char *src, int w, int h)
{
    unsigned char *dst;
    int x;
    int y;
    unsigned char r;
    unsigned char g;
    unsigned char b;
    float gray;

    dst = (unsigned char *)malloc((size_t)w * h);

    if (dst == NULL) {
        return NULL;
    }

    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            r = src[(y * w + x) * 3 + 0];
            g = src[(y * w + x) * 3 + 1];
            b = src[(y * w + x) * 3 + 2];

            gray = 0.299f * (float)r
                 + 0.587f * (float)g
                 + 0.114f * (float)b;

            dst[y * w + x] = (unsigned char)(gray + 0.5f);
        }
    }

    return dst;
}


int process_one_png(
    const char *input_path,
    const char *input_relative,
    const char *out_dir_absolute,
    const char *out_dir_relative,
    int index,
    FILE *manifest
)
{
    int w;
    int h;
    int channels;
    unsigned char *image;
    unsigned char *cropped_rgb;
    unsigned char *resized_rgb;
    unsigned char *gray272;

    char output_filename[64];
    char output_path[MAX_PATH_LEN];
    char output_relative[MAX_PATH_LEN];
    char label[MAX_FIELD_LEN];
    char event_id[MAX_FIELD_LEN];
    char duration[MAX_FIELD_LEN];

    int ok;

    if (!format_output_filename(
            output_filename,
            sizeof(output_filename),
            index
        ) ||
        !join_path(
            output_path,
            sizeof(output_path),
            out_dir_absolute,
            output_filename
        ) ||
        !join_manifest_path(
            output_relative,
            sizeof(output_relative),
            out_dir_relative,
            output_filename
        )) {
        printf("Errore: overflow nella composizione del path di output.\n");
        return -1;
    }

    image = stbi_load(input_path, &w, &h, &channels, 3);

    if (image == NULL) {
        printf("SKIP: impossibile leggere: %s\n", input_path);
        return 0;
    }

    if (w != 800 || h != 600) {
        printf("ATTENZIONE: dimensione non attesa %dx%d: %s\n", w, h, input_path);
    }

    cropped_rgb = crop_rgb(image, w, h);

    if (cropped_rgb == NULL) {
        printf("SKIP: crop fallito: %s\n", input_path);
        stbi_image_free(image);
        return 0;
    }

    resized_rgb = resize_bilinear_rgb(
        cropped_rgb,
        CROP_W,
        CROP_H,
        RESIZED_W,
        RESIZED_H
    );

    if (resized_rgb == NULL) {
        printf("SKIP: resize fallito: %s\n", input_path);
        free(cropped_rgb);
        stbi_image_free(image);
        return 0;
    }

    gray272 = rgb_to_gray(resized_rgb, RESIZED_W, RESIZED_H);

    if (gray272 == NULL) {
        printf("SKIP: grayscale fallito: %s\n", input_path);
        free(resized_rgb);
        free(cropped_rgb);
        stbi_image_free(image);
        return 0;
    }

    ok = stbi_write_png(
        output_path,
        RESIZED_W,
        RESIZED_H,
        1,
        gray272,
        RESIZED_W
    );

    if (!ok) {
        printf("SKIP: salvataggio fallito: %s\n", input_path);
    } else {
        extract_parent_folder(input_relative, label, MAX_FIELD_LEN);

        parse_event_and_duration(
            input_relative,
            event_id,
            MAX_FIELD_LEN,
            duration,
            MAX_FIELD_LEN
        );

        fprintf(manifest, "%d,", index);

        csv_write_field(manifest, label);
        fputc(',', manifest);

        csv_write_field(manifest, event_id);
        fputc(',', manifest);

        csv_write_field(manifest, duration);
        fputc(',', manifest);

        csv_write_field(manifest, input_relative);
        fputc(',', manifest);

        csv_write_field(manifest, output_relative);
        fputc('\n', manifest);

        if (index == 1 || index % 500 == 0) {
            printf("OK %06d  label=%s  event=%s  duration=%s\n",
                   index, label, event_id, duration);
        }
    }

    free(gray272);
    free(resized_rgb);
    free(cropped_rgb);
    stbi_image_free(image);

    if (!ok) {
        return 0;
    }

    return 1;
}


int main(int argc, char **argv)
{
    FILE *list_file;
    FILE *manifest;
    char line[MAX_PATH_LEN];
    char run_root[MAX_PATH_LEN];
    char out_dir_absolute[MAX_PATH_LEN];
    char manifest_path[MAX_PATH_LEN];
    char input_path[MAX_PATH_LEN];

    const char *list_path_absolute;
    const char *data_root_absolute;
    const char *out_dir_relative;

    int max_images;
    int attempted;
    int success;
    int index;
    int line_number;
    int line_status;
    int process_status;
    int failed;

    if (argc < 4 || argc > 5) {
        printf("Uso:\n");
        printf("  01_preprocess_raw_to_272_gray.exe lista_png_abs data_root_abs out_dir_rel [max_images]\n\n");
        printf("Test su 20 immagini:\n");
        printf("  01_preprocess_raw_to_272_gray.exe C:\\run\\lista_png.txt D:\\dataset out272_gray 20\n\n");
        printf("Tutto il dataset:\n");
        printf("  01_preprocess_raw_to_272_gray.exe C:\\run\\lista_png.txt D:\\dataset out272_gray\n");
        return 1;
    }

    list_path_absolute = argv[1];
    data_root_absolute = argv[2];
    out_dir_relative = argv[3];

    if (!is_absolute_path(list_path_absolute)) {
        printf("Errore: lista_png_abs deve essere un path assoluto.\n");
        return 1;
    }

    if (!is_absolute_path(data_root_absolute)) {
        printf("Errore: data_root_abs deve essere un path assoluto.\n");
        return 1;
    }

    if (!is_valid_relative_path(out_dir_relative)) {
        printf("Errore: out_dir_rel deve essere un path relativo valido.\n");
        return 1;
    }

    if (!dirname_from_absolute_path(
            list_path_absolute,
            run_root,
            sizeof(run_root)
        )) {
        printf("Errore: impossibile ricavare la directory della lista.\n");
        return 1;
    }

    if (!join_path(
            out_dir_absolute,
            sizeof(out_dir_absolute),
            run_root,
            out_dir_relative
        ) ||
        !join_path(
            manifest_path,
            sizeof(manifest_path),
            run_root,
            "manifest_preprocessing_272_only.csv"
        )) {
        printf("Errore: overflow nella composizione dei path di run.\n");
        return 1;
    }

    max_images = 0;

    if (argc >= 5) {
        max_images = atoi(argv[4]);
    }

    list_file = fopen(list_path_absolute, "rb");

    if (list_file == NULL) {
        printf("Errore: impossibile aprire la lista: %s\n", list_path_absolute);
        return 1;
    }

    manifest = fopen(manifest_path, "w");

    if (manifest == NULL) {
        printf("Errore: impossibile creare il manifest: %s\n", manifest_path);
        fclose(list_file);
        return 1;
    }

    fprintf(
        manifest,
        "index,label,event_id,duration,input_png,out272_gray\n"
    );

    attempted = 0;
    success = 0;
    index = 1;
    line_number = 0;
    failed = 0;

    for (;;) {
        line_status = read_list_line(list_file, line, sizeof(line));

        if (line_status == 0) {
            break;
        }

        line_number++;

        if (line_status == -1) {
            printf("Errore: riga %d troncata o troppo lunga.\n", line_number);
            failed = 1;
            break;
        }

        if (line_status == -2) {
            printf("Errore: carattere di controllo alla riga %d.\n", line_number);
            failed = 1;
            break;
        }

        remove_newline_and_bom(line);

        if (strlen(line) == 0) {
            continue;
        }

        if (max_images > 0 && attempted >= max_images) {
            break;
        }

        if (!is_valid_relative_path(line)) {
            printf("Errore: path dataset-relative non valido alla riga %d: %s\n",
                   line_number, line);
            failed = 1;
            break;
        }

        if (!join_path(
                input_path,
                sizeof(input_path),
                data_root_absolute,
                line
            )) {
            printf("Errore: overflow nel path di input alla riga %d.\n",
                   line_number);
            failed = 1;
            break;
        }

        attempted++;

        process_status = process_one_png(
                input_path,
                line,
                out_dir_absolute,
                out_dir_relative,
                index,
                manifest
            );

        if (process_status < 0) {
            failed = 1;
            break;
        }

        if (process_status > 0) {
            success++;
            index++;
        }
    }

    fclose(list_file);
    fclose(manifest);

    printf("\nFine preprocessing.\n");
    printf("PNG tentate: %d\n", attempted);
    printf("PNG riuscite: %d\n", success);
    printf("Manifest creato: %s\n", manifest_path);

    if (max_images > 0) {
        printf("\nNota: era attivo il limite max_images = %d.\n", max_images);
        printf("Per processare tutto il dataset, rilancia senza questo parametro.\n");
    }

    return failed ? 1 : 0;
}
