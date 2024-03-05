// Author: APD team, except where source was noted

#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef __USE_XOPEN2K
#define __USE_XOPEN2K
#endif

#ifndef __USE_MISC
#define __USE_MISC
#endif

#include <pthread.h>

#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
#define STEP                    8
#define SIGMA                   200
#define RESCALE_X               2048
#define RESCALE_Y               2048

#define CLAMP(v, min, max) if(v < min) { v = min; } else if(v > max) { v = max; }

// macro used for timing how much each step of the algorithm takes
#define TIME_IT(x, diff) {\
    struct timespec begin;\
    struct timespec end;\
    timespec_get(&begin, TIME_UTC);\
    x;\
    timespec_get(&end, TIME_UTC);\
    int64_t begin_milis = ((int64_t) begin.tv_sec) * 1000 + ((int64_t) begin.tv_nsec) / 1000000;\
    int64_t end_milis = ((int64_t) end.tv_sec) * 1000 + ((int64_t) end.tv_nsec) / 1000000;\
    diff = end_milis - begin_milis;\
}

// a struct with every piece of information needed by a thread
using args = struct {
    // image specific fields
    ppm_image **contour_map;
    ppm_image *orig_image;
    ppm_image *new_image;
    unsigned char **grid;
    int step_x;
    int step_y;
    int sigma;

    // thread specific fields
    // this pointer will point to the same barrier for all threads
    // otherwise it would be pointless
    pthread_barrier_t *barrier;
    int tid;
    int no_threads;
};

// unchanged
// Creates a map between the binary configuration (e.g. 0110_2) and the corresponding pixels
// that need to be set on the output image. An array is used for this map since the keys are
// binary numbers in 0-15. Contour images are located in the './contours' directory.
ppm_image **init_contour_map() {
    ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!map) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        map[i] = read_ppm(filename);
    }

    return map;
}

// taken from the first laboratory
static inline int get_start_bound(int thread_ID, int no_threads, int upper_bound) {
    return (int)(thread_ID * (double)upper_bound / no_threads);
}

// also taken from the first laboratory
static inline int get_end_bound(int thread_ID, int no_threads, int upper_bound) {
    int id_based_value = (int)((thread_ID + 1) * (double)upper_bound / no_threads);
    return id_based_value < upper_bound ? id_based_value : upper_bound;
}

// Updates a particular section of an image with the corresponding contour pixels.
// Used to create the complete contour image.
void update_image(ppm_image *image, const ppm_image *contour, int x, int y) {
    for (int i = 0; i < contour->x; i++) {
        for (int j = 0; j < contour->y; j++) {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * image->y + y + j;

            // i replaced the three assigments with this memcpy because
            // i thought it would be more efficient
            memcpy(&(image->data[image_pixel_index]), &(contour->data[contour_pixel_index]), sizeof(ppm_pixel));
        }
    }
}

// unchanged
// Calls `free` method on the utilized resources.
void free_resources(ppm_image *image, ppm_image **contour_map, unsigned char **grid, int step_x) {
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

    for (int i = 0; i <= image->x / step_x; i++) {
        free(grid[i]);
    }
    free(grid);

    free(image->data);
    free(image);
}

ppm_image* make_new_empty_image(ppm_image *image) {
    // we only rescale downwards
    if (image->x <= RESCALE_X && image->y <= RESCALE_Y) {
        return image;
    }

    // alloc memory for image
    ppm_image *new_image = (ppm_image *)malloc(sizeof(ppm_image));
    if (!new_image) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }
    new_image->x = RESCALE_X;
    new_image->y = RESCALE_Y;

    new_image->data = (ppm_pixel*)malloc(new_image->x * new_image->y * sizeof(ppm_pixel));
    if (!new_image) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }
    return new_image;
}

void rescale_new_image(args* a) {
    uint8_t sample[3] = {0};
    
    // get the start and end bounds for this thread
    int start = get_start_bound(a->tid, a->no_threads, a->new_image->x);
    int end = get_end_bound(a->tid, a->no_threads, a->new_image->x);
    
    // i split these operations evenly between the threads
    for (int i = start; i < end; i++) {
        // move this outside the inner loop to remove unnecessary
        // divisions(they probably got optimized anyways but i tried)
        float u = (float)i / (float)(a->new_image->x - 1);
        for (int j = 0; j < a->new_image->y; j++) {
            float v = (float)j / (float)(a->new_image->y - 1);
            sample_bicubic(a->orig_image, u, v, sample);

            // i replaced these 3 assigments as well with this assignment
            // after seeing that the layout of the struct matches the byte order
            // of sample
            memcpy(&(a->new_image->data[i * a->new_image->y + j]), sample, 3);
        }
    }
}

// first half of sample_grid, not parallel, copy pasted and left as it was
unsigned char **initialize_grid(const ppm_image *image, int step_x, int step_y) {
    int p = image->x / step_x;
    int q = image->y / step_y;

    unsigned char **grid = (unsigned char **)malloc((p + 1) * sizeof(unsigned char*));
    if (!grid) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i <= p; i++) {
        grid[i] = (unsigned char *)malloc((q + 1) * sizeof(unsigned char));
        if (!grid[i]) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }
    return grid;
}

void fill_grid(const args* a) {
    int q = a->new_image->x / a->step_x;
    int p = a->new_image->y / a->step_y;

    // get the corresponding bounds for the current thread
    int start_p = get_start_bound(a->tid, a->no_threads, p);
    int end_p = get_end_bound(a->tid, a->no_threads, p);
    int start_q = get_start_bound(a->tid, a->no_threads, q);
    int end_q = get_end_bound(a->tid, a->no_threads, q);

    // i split these operations evenly between the threads
    // by what i think are the lines of the pixel matrix
    // (i may be wrong, didn't analyze the algorithm thoroughly
    // enough but it works)
    for (int i = start_p; i < end_p; i++) {
        for (int j = 0; j < q; j++) {
            ppm_pixel curr_pixel = a->new_image->data[i * a->step_x * a->new_image->y + j * a->step_y];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            if (curr_color > a->sigma) {
                a->grid[i][j] = 0;
            } else {
                a->grid[i][j] = 1;
            }
        }
    }

    // added as per the last update of the skeleton
    a->grid[p][q] = 0;

    // i split these operations evenly between the threads
    for (int i = start_p; i < end_p; i++) {
        ppm_pixel curr_pixel = a->new_image->data[i * a->step_x * a->new_image->y + a->new_image->x - 1];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > a->sigma) {
            a->grid[i][q] = 0;
        } else {
            a->grid[i][q] = 1;
        }
    }
    
    // i split these operations evenly between the threads
    for (int j = start_q; j < end_q; j++) {
        ppm_pixel curr_pixel = a->new_image->data[(a->new_image->x - 1) * a->new_image->y + j * a->step_y];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > a->sigma) {
            a->grid[p][j] = 0;
        } else {
            a->grid[p][j] = 1;
        }
    }
}

void march_parallel(args* a) {
    int p = a->new_image->x / a->step_x;
    int q = a->new_image->y / a->step_y;

    // get the start and end bounds
    int start_p = get_start_bound(a->tid, a->no_threads, q);
    int end_p = get_end_bound(a->tid, a->no_threads, q);

    for (int i = 0; i < p; i++) {
        // i split these operations evenly between the threads
        for (int j = start_p; j < end_p; j++) {
            unsigned char k = 8 * a->grid[i][j] + 4 * a->grid[i][j + 1] + 2 * a->grid[i + 1][j + 1] + a->grid[i + 1][j];
            update_image(a->new_image, a->contour_map[k], i * a->step_x, j * a->step_y);
        }
    }
}

void* thread_function(void *arguments) {
    args* a = (args*) arguments;
    
    // if the two images are the same there is no need for rescaling
    if (a->new_image != a->orig_image) {
        rescale_new_image(a);

        // the other operations can be executed only after
        // the rerscaling has been done
        pthread_barrier_wait(a->barrier);
    }

    fill_grid(a);
    
    // the march algorithm can be performed only after the grid
    // has been filled
    pthread_barrier_wait(a->barrier);

    march_parallel(a);
    return nullptr;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <P>\n");
        return 1;
    }

    // get the number of threads from the arguments
    int no_threads = 0;

    // error checking
    if ( int ret = sscanf(argv[3], "%d", &no_threads); ret == 0 || no_threads <= 0) {
        fprintf(stderr, "Invalid number of threads\n");
        return 1;
    }

    

    ppm_image *image = read_ppm(argv[1]);
    int step_x = STEP;
    int step_y = STEP;
    
    // 0. Initialize contour map
    ppm_image **contour_map;
    contour_map = init_contour_map();
    
    // 1. Rescale the image
    
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, no_threads);

    // preallocate common resources for threads if necessary
    ppm_image *scaled_image = make_new_empty_image(image);
    unsigned char **grid = initialize_grid(scaled_image, step_x, step_y);
    
    pthread_t* threads = (pthread_t*)malloc(no_threads * sizeof(*threads));
    args* thread_args = (args*)malloc(sizeof(*thread_args) * no_threads);

    for (int i = 0; i < no_threads; i++) {
        // initialize arguments for the new thread
        // that is being created
        thread_args[i] = (args) {
            .contour_map = contour_map,
            .orig_image = image,
            .new_image = scaled_image,
            .grid = grid,
            .step_x = step_x,
            .step_y = step_y,
            .sigma = SIGMA,
            .barrier = &barrier,
            .tid = i,
            .no_threads = no_threads,
        };

        // did not need custom attributes for this context
        pthread_create(&threads[i], nullptr, thread_function, &thread_args[i]);
    }
    for (int i = 0; i < no_threads; i++) {
        // i'm not returning anything useful so i just passed NULL
        pthread_join(threads[i], nullptr);
    }
    // 4. Write output
    write_ppm(scaled_image, argv[2]);
    
    // clean stuff up
    pthread_barrier_destroy(&barrier);
    free(thread_args);
    free(threads);
    free_resources(scaled_image, contour_map, grid, step_x);

    return 0;
}
