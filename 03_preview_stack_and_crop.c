#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/*
 * Preview dello stacking, del crop centrale e della perturbazione temporale.
 *
 * Input:
 *   manifest_events_272.csv
 *
 * Il programma prende il primo evento completo, legge le quattro viste:
 *   0.5 s, 1.0 s, 2.0 s, 4.0 s
 *
 * Ogni vista è una immagine 272x224 in scala di grigi.
 *
 * Concettualmente costruisce:
 *   stack[4][224][272]
 *
 * Poi produce due versioni:
 *
 * 1. Input image non perturbata:
 *    crop centrale da 272x224 a 224x224.
 *
 * 2. Perturbed image:
 *    crop temporale con shift casuale compreso tra 0 e 24 pixel.
 *
 * Lo stesso shift viene applicato a tutte e quattro le viste temporali
 * dello stesso evento, per mantenere la coerenza del campione.
 */

#define INPUT_W 272
#define INPUT_H 224

#define FINAL_W 224
#define FINAL_H 224

#define NUM_CHANNELS 4

#define MIN_SHIFT 0
#define MAX_SHIFT 24

#define MAX_LINE_LEN 8192
#define MAX_FIELDS 16
#define MAX_FIELD_LEN 4096


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


unsigned char *crop_224_from_left(const unsigned char *src, int left)
{
    unsigned char *dst;
    int x;
    int y;
    int top;

    if (left < 0 || left + FINAL_W > INPUT_W) {
        return NULL;
    }

    dst = (unsigned char *)malloc((size_t)FINAL_W * FINAL_H);

    if (dst == NULL) {
        return NULL;
    }

    /*
     * L'altezza è già 224, quindi non viene tagliata in frequenza.
     * Il crop avviene solo nella direzione temporale, cioè in larghezza.
     */
    top = (INPUT_H - FINAL_H) / 2;

    for (y = 0; y < FINAL_H; y++) {
        for (x = 0; x < FINAL_W; x++) {
            dst[y * FINAL_W + x] =
                src[(top + y) * INPUT_W + (left + x)];
        }
    }

    return dst;
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


int main(int argc, char **argv)
{
    const char *manifest_path;
    FILE *manifest;

    char line[MAX_LINE_LEN];
    char fields[MAX_FIELDS][MAX_FIELD_LEN];

    int n_fields;

    unsigned char *stack[NUM_CHANNELS];
    unsigned char *central_crop[NUM_CHANNELS];
    unsigned char *perturbed_crop[NUM_CHANNELS];

    const char *duration_names[NUM_CHANNELS];

    char output_path[MAX_FIELD_LEN];

    int i;
    int ok;

    int central_left;
    int random_shift;

    if (argc < 2) {
        printf("Uso:\n");
        printf("  03_preview_stack_and_crop.exe manifest_events_272.csv [shift]\n\n");
        printf("Se shift non viene indicato, viene scelto casualmente tra 0 e 24.\n");
        printf("Esempio con shift fisso:\n");
        printf("  03_preview_stack_and_crop.exe manifest_events_272.csv 12\n");
        return 1;
    }

    manifest_path = argv[1];

    duration_names[0] = "0_5";
    duration_names[1] = "1_0";
    duration_names[2] = "2_0";
    duration_names[3] = "4_0";

    for (i = 0; i < NUM_CHANNELS; i++) {
        stack[i] = NULL;
        central_crop[i] = NULL;
        perturbed_crop[i] = NULL;
    }

    /*
     * Crop centrale:
     * 272 - 224 = 48 pixel.
     * Taglio 24 pixel a sinistra e 24 a destra.
     */
    central_left = (INPUT_W - FINAL_W) / 2;

    /*
     * Crop perturbato.
     * Se l'utente passa uno shift, viene usato quello.
     * Altrimenti viene estratto casualmente tra 0 e 24.
     */
    if (argc >= 3) {
        random_shift = atoi(argv[2]);

        if (random_shift < MIN_SHIFT || random_shift > MAX_SHIFT) {
            printf("Errore: lo shift deve essere compreso tra %d e %d.\n",
                   MIN_SHIFT, MAX_SHIFT);
            return 1;
        }
    } else {
        srand((unsigned int)time(NULL));
        random_shift = MIN_SHIFT + rand() % (MAX_SHIFT - MIN_SHIFT + 1);
    }

    manifest = fopen(manifest_path, "r");

    if (manifest == NULL) {
        printf("Errore: impossibile aprire il manifest:\n%s\n", manifest_path);
        return 1;
    }

    /*
     * Salta intestazione.
     */
    if (fgets(line, MAX_LINE_LEN, manifest) == NULL) {
        printf("Errore: manifest vuoto.\n");
        fclose(manifest);
        return 1;
    }

    /*
     * Legge il primo evento.
     */
    if (fgets(line, MAX_LINE_LEN, manifest) == NULL) {
        printf("Errore: nessun evento nel manifest.\n");
        fclose(manifest);
        return 1;
    }

    fclose(manifest);

    remove_newline_and_bom(line);

    n_fields = parse_csv_line(line, fields);

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
    if (n_fields < 7) {
        printf("Errore: riga evento non valida.\n");
        return 1;
    }

    printf("Evento selezionato:\n");
    printf("  event_index = %s\n", fields[0]);
    printf("  label       = %s\n", fields[1]);
    printf("  event_id    = %s\n", fields[2]);
    printf("\n");

    printf("Crop centrale:\n");
    printf("  left = %d\n", central_left);
    printf("  colonne usate = %d -> %d\n", central_left, central_left + FINAL_W - 1);
    printf("\n");

    printf("Crop perturbato:\n");
    printf("  shift casuale = %d\n", random_shift);
    printf("  colonne usate = %d -> %d\n", random_shift, random_shift + FINAL_W - 1);
    printf("\n");

    /*
     * Carica le quattro viste temporali.
     */
    for (i = 0; i < NUM_CHANNELS; i++) {
        if (!load_gray_image_272(fields[3 + i], &stack[i])) {
            return 1;
        }
    }

    printf("Caricate 4 viste 272x224.\n");
    printf("Stack concettuale: 4 x 224 x 272\n");

    /*
     * Applica crop centrale e crop perturbato a tutti i canali.
     */
    for (i = 0; i < NUM_CHANNELS; i++) {
        central_crop[i] = crop_224_from_left(stack[i], central_left);

        if (central_crop[i] == NULL) {
            printf("Errore memoria crop centrale canale %d.\n", i);
            return 1;
        }

        perturbed_crop[i] = crop_224_from_left(stack[i], random_shift);

        if (perturbed_crop[i] == NULL) {
            printf("Errore memoria crop perturbato canale %d.\n", i);
            return 1;
        }
    }

    printf("Crop centrale e crop perturbato applicati.\n");
    printf("Output concettuale: 4 x 224 x 224 per entrambe le versioni.\n\n");

    /*
     * Salva le quattro viste centrali come PNG separate.
     */
    for (i = 0; i < NUM_CHANNELS; i++) {
        sprintf(
            output_path,
            "tensor_test_preview/central_event_%s_%s_224.png",
            fields[0],
            duration_names[i]
        );

        ok = stbi_write_png(
            output_path,
            FINAL_W,
            FINAL_H,
            1,
            central_crop[i],
            FINAL_W
        );

        if (!ok) {
            printf("Errore salvataggio:\n%s\n", output_path);
        } else {
            printf("Creato: %s\n", output_path);
        }
    }

    /*
     * Salva le quattro viste perturbate come PNG separate.
     */
    for (i = 0; i < NUM_CHANNELS; i++) {
        sprintf(
            output_path,
            "tensor_test_preview/perturbed_event_%s_shift_%02d_%s_224.png",
            fields[0],
            random_shift,
            duration_names[i]
        );

        ok = stbi_write_png(
            output_path,
            FINAL_W,
            FINAL_H,
            1,
            perturbed_crop[i],
            FINAL_W
        );

        if (!ok) {
            printf("Errore salvataggio:\n%s\n", output_path);
        } else {
            printf("Creato: %s\n", output_path);
        }
    }

    for (i = 0; i < NUM_CHANNELS; i++) {
        if (stack[i] != NULL) {
            stbi_image_free(stack[i]);
        }

        if (central_crop[i] != NULL) {
            free(central_crop[i]);
        }

        if (perturbed_crop[i] != NULL) {
            free(perturbed_crop[i]);
        }
    }

    printf("\nPreview completata.\n");

    return 0;
}