#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: ./collaborative_filtering <user_item_rating_matrix>\n");
        exit(EXIT_FAILURE);
    }

    fprintf(stdout, "File name: %s\n", argv[1]);

    return EXIT_SUCCESS;
}
