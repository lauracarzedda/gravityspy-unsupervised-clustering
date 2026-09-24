#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Costruzione del manifest a livello evento.
 *
 * Input:
 *   manifest_preprocessing_272_only.csv
 *
 * Output:
 *   manifest_events_272.csv
 *
 * Il manifest di input ha una riga per ogni PNG preprocessata.
 * Questo programma raggruppa le quattro viste temporali dello stesso evento:
 *
 *   0.5 s
 *   1.0 s
 *   2.0 s
 *   4.0 s
 *
 * e produce una riga per ogni glitch/evento completo.
 *
 * Questo file sarà usato per costruire il tensore:
 *
 *   4 x 224 x 272
 *
 * prima del crop temporale finale descritto da Sakai.
 */

#define MAX_LINE_LEN 8192
#define MAX_FIELDS 16
#define MAX_FIELD_LEN 4096

#define D05 0
#define D10 1
#define D20 2
#define D40 3

typedef struct {
    char *label;
    char *event_id;
    char *img272[4];
    int has_duration[4];
} Event;


char *duplicate_string(const char *s)
{
    char *p;

    p = (char *)malloc(strlen(s) + 1);

    if (p == NULL) {
        return NULL;
    }

    strcpy(p, s);

    return p;
}


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


int duration_to_index(const char *duration)
{
    if (strcmp(duration, "0.5") == 0) {
        return D05;
    }

    if (strcmp(duration, "1.0") == 0) {
        return D10;
    }

    if (strcmp(duration, "2.0") == 0) {
        return D20;
    }

    if (strcmp(duration, "4.0") == 0) {
        return D40;
    }

    return -1;
}


int find_event(Event *events, int n_events, const char *label, const char *event_id)
{
    int i;

    for (i = 0; i < n_events; i++) {
        if (strcmp(events[i].label, label) == 0 &&
            strcmp(events[i].event_id, event_id) == 0) {
            return i;
        }
    }

    return -1;
}


int add_event(
    Event **events_ptr,
    int *n_events_ptr,
    int *capacity_ptr,
    const char *label,
    const char *event_id
)
{
    Event *events;
    Event *new_events;
    int n_events;
    int capacity;
    int i;

    events = *events_ptr;
    n_events = *n_events_ptr;
    capacity = *capacity_ptr;

    if (n_events >= capacity) {
        if (capacity == 0) {
            capacity = 1024;
        } else {
            capacity = capacity * 2;
        }

        new_events = (Event *)realloc(events, (size_t)capacity * sizeof(Event));

        if (new_events == NULL) {
            return -1;
        }

        events = new_events;
        *events_ptr = events;
        *capacity_ptr = capacity;
    }

    events[n_events].label = duplicate_string(label);
    events[n_events].event_id = duplicate_string(event_id);

    if (events[n_events].label == NULL || events[n_events].event_id == NULL) {
        return -1;
    }

    for (i = 0; i < 4; i++) {
        events[n_events].img272[i] = NULL;
        events[n_events].has_duration[i] = 0;
    }

    *n_events_ptr = n_events + 1;

    return n_events;
}


int event_is_complete(const Event *event)
{
    if (!event->has_duration[D05]) {
        return 0;
    }

    if (!event->has_duration[D10]) {
        return 0;
    }

    if (!event->has_duration[D20]) {
        return 0;
    }

    if (!event->has_duration[D40]) {
        return 0;
    }

    return 1;
}


void free_events(Event *events, int n_events)
{
    int i;
    int j;

    if (events == NULL) {
        return;
    }

    for (i = 0; i < n_events; i++) {
        if (events[i].label != NULL) {
            free(events[i].label);
        }

        if (events[i].event_id != NULL) {
            free(events[i].event_id);
        }

        for (j = 0; j < 4; j++) {
            if (events[i].img272[j] != NULL) {
                free(events[i].img272[j]);
            }
        }
    }

    free(events);
}


int main(int argc, char **argv)
{
    const char *input_manifest_path;

    FILE *input_file;
    FILE *events_file;

    char line[MAX_LINE_LEN];
    char fields[MAX_FIELDS][MAX_FIELD_LEN];

    Event *events;
    int n_events;
    int capacity;

    int row_count;
    int png_rows;
    int bad_rows;
    int duplicate_rows;
    int complete_events;
    int incomplete_events;

    int n_fields;
    int duration_index;
    int event_pos;
    int added_pos;
    int i;
    int event_index;

    if (argc < 2) {
        printf("Uso:\n");
        printf("  02_make_event_manifest.exe manifest_preprocessing_272_only.csv\n");
        return 1;
    }

    input_manifest_path = argv[1];

    input_file = fopen(input_manifest_path, "r");

    if (input_file == NULL) {
        printf("Errore: impossibile aprire il manifest di input:\n");
        printf("%s\n", input_manifest_path);
        return 1;
    }

    events_file = fopen("manifest_events_272.csv", "w");

    if (events_file == NULL) {
        printf("Errore: impossibile creare manifest_events_272.csv\n");
        fclose(input_file);
        return 1;
    }

    events = NULL;
    n_events = 0;
    capacity = 0;

    row_count = 0;
    png_rows = 0;
    bad_rows = 0;
    duplicate_rows = 0;

    while (fgets(line, MAX_LINE_LEN, input_file) != NULL) {
        remove_newline_and_bom(line);

        if (strlen(line) == 0) {
            continue;
        }

        row_count++;

        if (row_count == 1) {
            continue;
        }

        n_fields = parse_csv_line(line, fields);

        /*
         * Manifest atteso:
         * 0 index
         * 1 label
         * 2 event_id
         * 3 duration
         * 4 input_png
         * 5 out272_gray
         */
        if (n_fields < 6) {
            bad_rows++;
            continue;
        }

        png_rows++;

        duration_index = duration_to_index(fields[3]);

        if (duration_index < 0) {
            bad_rows++;
            continue;
        }

        event_pos = find_event(events, n_events, fields[1], fields[2]);

        if (event_pos < 0) {
            added_pos = add_event(
                &events,
                &n_events,
                &capacity,
                fields[1],
                fields[2]
            );

            if (added_pos < 0) {
                printf("Errore: memoria insufficiente durante la creazione degli eventi.\n");
                free_events(events, n_events);
                fclose(events_file);
                fclose(input_file);
                return 1;
            }

            event_pos = added_pos;
        }

        if (events[event_pos].has_duration[duration_index]) {
            duplicate_rows++;
        } else {
            events[event_pos].img272[duration_index] = duplicate_string(fields[5]);

            if (events[event_pos].img272[duration_index] == NULL) {
                printf("Errore: memoria insufficiente durante la copia del percorso immagine.\n");
                free_events(events, n_events);
                fclose(events_file);
                fclose(input_file);
                return 1;
            }

            events[event_pos].has_duration[duration_index] = 1;
        }
    }

    fprintf(
        events_file,
        "event_index,label,event_id,img_0_5_272,img_1_0_272,img_2_0_272,img_4_0_272\n"
    );

    complete_events = 0;
    incomplete_events = 0;
    event_index = 1;

    for (i = 0; i < n_events; i++) {
        if (event_is_complete(&events[i])) {
            complete_events++;

            fprintf(events_file, "%d,", event_index);

            csv_write_field(events_file, events[i].label);
            fputc(',', events_file);

            csv_write_field(events_file, events[i].event_id);
            fputc(',', events_file);

            csv_write_field(events_file, events[i].img272[D05]);
            fputc(',', events_file);

            csv_write_field(events_file, events[i].img272[D10]);
            fputc(',', events_file);

            csv_write_field(events_file, events[i].img272[D20]);
            fputc(',', events_file);

            csv_write_field(events_file, events[i].img272[D40]);
            fputc('\n', events_file);

            event_index++;
        } else {
            incomplete_events++;
        }
    }

    fclose(events_file);
    fclose(input_file);

    printf("Manifest eventi creato correttamente.\n");
    printf("Righe PNG lette: %d\n", png_rows);
    printf("Eventi trovati: %d\n", n_events);
    printf("Eventi completi con 4 viste: %d\n", complete_events);
    printf("Eventi incompleti esclusi: %d\n", incomplete_events);
    printf("Righe problematiche: %d\n", bad_rows);
    printf("Duplicati evento/durata: %d\n", duplicate_rows);
    printf("\nFile creato:\n");
    printf("manifest_events_272.csv\n");

    free_events(events, n_events);

    return 0;
}