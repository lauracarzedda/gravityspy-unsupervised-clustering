#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>


static int is_blank_line(const char *line)
{
    if (line == NULL) {
        return 1;
    }

    while (*line != '\0') {
        if (*line != ' ' &&
            *line != '\t' &&
            *line != '\r' &&
            *line != '\n') {

            return 0;
        }

        line++;
    }

    return 1;
}


static int validate_split_row(
    const char *line,
    const char *expected_name,
    size_t expected_pos
)
{
    const char *first_comma;
    const char *second_comma;

    char pos_buffer[64];
    size_t name_length;
    size_t pos_length;

    char *endptr;
    unsigned long parsed_pos;

    if (line == NULL || expected_name == NULL) {
        return 1;
    }

    first_comma = strchr(line, ',');

    if (first_comma == NULL) {
        return 1;
    }

    name_length = (size_t)(first_comma - line);

    if (strlen(expected_name) != name_length ||
        strncmp(line, expected_name, name_length) != 0) {

        return 1;
    }

    second_comma = strchr(first_comma + 1, ',');

    if (second_comma == NULL) {
        return 1;
    }

    pos_length =
        (size_t)(second_comma - (first_comma + 1));

    if (pos_length == 0 ||
        pos_length >= sizeof(pos_buffer)) {

        return 1;
    }

    memcpy(
        pos_buffer,
        first_comma + 1,
        pos_length
    );

    pos_buffer[pos_length] = '\0';

    errno = 0;
    endptr = NULL;

    parsed_pos = strtoul(
        pos_buffer,
        &endptr,
        10
    );

    if (errno != 0 ||
        endptr == pos_buffer ||
        *endptr != '\0' ||
        parsed_pos != (unsigned long)expected_pos) {

        return 1;
    }

    return 0;
}


static int copy_subset(
    const char *input_path,
    const char *output_path,
    const char *expected_name,
    size_t requested_rows
)
{
    FILE *input;
    FILE *output;

    char line[4096];
    size_t copied_rows;
    size_t source_line_number;

    if (input_path == NULL ||
        output_path == NULL ||
        expected_name == NULL ||
        requested_rows == 0) {

        printf("Errore: argomenti non validi in copy_subset.\n");
        return 1;
    }

    input = fopen(input_path, "r");

    if (input == NULL) {
        printf("Errore: impossibile aprire il file sorgente:\n");
        printf("%s\n", input_path);
        return 1;
    }

    output = fopen(output_path, "w");

    if (output == NULL) {
        printf("Errore: impossibile creare il file di output:\n");
        printf("%s\n", output_path);

        fclose(input);
        return 1;
    }

    /*
       Copia e controlla l'header.
    */
    if (fgets(line, sizeof(line), input) == NULL) {
        printf("Errore: file sorgente vuoto:\n");
        printf("%s\n", input_path);

        fclose(output);
        fclose(input);
        return 1;
    }

    if (strncmp(
            line,
            "split_name,split_pos,manifest_index,event_index,event_id,tensor_path",
            strlen("split_name,split_pos,manifest_index,event_index,event_id,tensor_path")
        ) != 0) {

        printf("Errore: header inatteso in:\n");
        printf("%s\n", input_path);

        fclose(output);
        fclose(input);
        return 1;
    }

    fputs(line, output);

    copied_rows = 0;
    source_line_number = 1;

    while (copied_rows < requested_rows &&
           fgets(line, sizeof(line), input) != NULL) {

        source_line_number++;

        if (is_blank_line(line)) {
            continue;
        }

        /*
           Poiche prendiamo le prime righe dello split ufficiale,
           split_pos deve gia essere 0, 1, 2, ...
        */
        if (validate_split_row(
                line,
                expected_name,
                copied_rows
            ) != 0) {

            printf("Errore: riga split non valida.\n");
            printf("File: %s\n", input_path);
            printf("Riga sorgente: %lu\n",
                   (unsigned long)source_line_number);
            printf("Posizione subset attesa: %lu\n",
                   (unsigned long)copied_rows);

            fclose(output);
            fclose(input);
            return 1;
        }

        fputs(line, output);
        copied_rows++;
    }

    fclose(output);
    fclose(input);

    if (copied_rows != requested_rows) {
        printf("Errore: righe insufficienti nel file sorgente.\n");
        printf("Richieste: %lu\n",
               (unsigned long)requested_rows);
        printf("Copiate:   %lu\n",
               (unsigned long)copied_rows);

        return 1;
    }

    printf("%s creato correttamente.\n", output_path);
    printf("  split       = %s\n", expected_name);
    printf("  righe dati  = %lu\n",
           (unsigned long)copied_rows);

    return 0;
}


int main(int argc, char **argv)
{
    const char *train_input;
    const char *val_input;

    const char *train_output;
    const char *val_output;

    size_t train_count;
    size_t val_count;

    int rc;

    train_input =
        "split_train_indices_seed123.csv";

    val_input =
        "split_val_indices_seed123.csv";

    train_output =
        "sakai_multires_c/configs/subset224_train_800_seed123.csv";

    val_output =
        "sakai_multires_c/configs/subset224_val_200_seed123.csv";

    train_count = 800;
    val_count = 200;

    /*
       Uso opzionale:

       make_subset_224.exe
           train_input.csv
           val_input.csv
           train_output.csv
           val_output.csv
           train_count
           val_count
    */
    if (argc >= 2) {
        train_input = argv[1];
    }

    if (argc >= 3) {
        val_input = argv[2];
    }

    if (argc >= 4) {
        train_output = argv[3];
    }

    if (argc >= 5) {
        val_output = argv[4];
    }

    if (argc >= 6) {
        train_count =
            (size_t)strtoul(argv[5], NULL, 10);
    }

    if (argc >= 7) {
        val_count =
            (size_t)strtoul(argv[6], NULL, 10);
    }

    if (train_count == 0 || val_count == 0) {
        printf("Errore: train_count e val_count devono essere maggiori di zero.\n");
        return 1;
    }

    printf("Creazione subset locale 224 x 224\n");
    printf("Train:      %lu eventi\n",
           (unsigned long)train_count);
    printf("Validation: %lu eventi\n\n",
           (unsigned long)val_count);

    rc = copy_subset(
        train_input,
        train_output,
        "train",
        train_count
    );

    if (rc != 0) {
        return 1;
    }

    rc = copy_subset(
        val_input,
        val_output,
        "validation",
        val_count
    );

    if (rc != 0) {
        return 1;
    }

    printf("\nOK: subset 224 creato.\n");
    printf("Totale eventi: %lu\n",
           (unsigned long)(train_count + val_count));
    printf("Nessuna label e' stata usata per selezionare gli eventi.\n");
    printf("Gli eventi sono i primi elementi degli split gia' mescolati con seed 123.\n");

    return 0;
}
