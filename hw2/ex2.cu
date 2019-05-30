/* compile with: nvcc -O3 -maxrregcount=32 hw2.cu -o hw2 */

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <string.h>

///////////////////////////////// DO NOT CHANGE ///////////////////////////////
#define IMG_DIMENSION 32
#define NREQUESTS 10000

typedef unsigned char uchar;

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

#define SQR(a) ((a) * (a))

void process_image(uchar *img_in, uchar *img_out) {
    int histogram[256] = { 0 };
    for (int i = 0; i < SQR(IMG_DIMENSION); i++) {
        histogram[img_in[i]]++;
    }

    int cdf[256] = { 0 };
    int hist_sum = 0;
    for (int i = 0; i < 256; i++) {
        hist_sum += histogram[i];
        cdf[i] = hist_sum;
    }

    int cdf_min = 0;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }

    uchar map[256] = { 0 };
    for (int i = 0; i < 256; i++) {
        int map_value = (float)(cdf[i] - cdf_min) / (SQR(IMG_DIMENSION) - cdf_min) * 255;
        map[i] = (uchar)map_value;
    }

    for (int i = 0; i < SQR(IMG_DIMENSION); i++) {
        img_out[i] = map[img_in[i]];
    }
}

double static inline get_time_msec(void) {
    struct timespec t;
    int res = clock_gettime(CLOCK_MONOTONIC, &t);
    if (res) {
        perror("clock_gettime failed");
        exit(1);
    }
    return t.tv_sec * 1e+3 + t.tv_nsec * 1e-6;
}

/* we'll use these to rate limit the request load */
struct rate_limit_t {
    double last_checked;
    double lambda;
    unsigned seed;
};

void rate_limit_init(struct rate_limit_t *rate_limit, double lambda, int seed) {
    rate_limit->lambda = lambda;
    rate_limit->seed = (seed == -1) ? 0 : seed;
    rate_limit->last_checked = 0;
}

int rate_limit_can_send(struct rate_limit_t *rate_limit) {
    if (rate_limit->lambda == 0) return 1;
    double now = get_time_msec() * 1e-3;
    double dt = now - rate_limit->last_checked;
    double p = dt * rate_limit->lambda;
    rate_limit->last_checked = now;
    if (p > 1) p = 1;
    double r = (double)rand_r(&rate_limit->seed) / RAND_MAX;
    return (p > r);
}

void rate_limit_wait(struct rate_limit_t *rate_limit) {
    while (!rate_limit_can_send(rate_limit)) {
        struct timespec t = {
            0,
            long(1. / (rate_limit->lambda * 1e-9) * 0.01)
        };
        nanosleep(&t, NULL);
    }
}

double distance_sqr_between_image_arrays(uchar *img_arr1, uchar *img_arr2) {
    double distance_sqr = 0;
    for (int i = 0; i < NREQUESTS * SQR(IMG_DIMENSION); i++) {
        distance_sqr += SQR(img_arr1[i] - img_arr2[i]);
    }
    return distance_sqr;
}

/* we won't load actual files. just fill the images with random bytes */
void load_images(uchar *images) {
    srand(0);
    for (int i = 0; i < NREQUESTS * SQR(IMG_DIMENSION); i++) {
        images[i] = rand() % 256;
    }
}

__device__ int arr_min(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int rhs, lhs;

    for (int stride = 1; stride < arr_size; stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            rhs = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            lhs = arr[tid];
            if (rhs != 0) {
                if (lhs == 0)
                    arr[tid] = rhs;
                else
                    arr[tid] = min(arr[tid], rhs);
            }
        }
        __syncthreads();
    }

    int ret = arr[arr_size - 1];
    return ret;
}

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;

    for (int stride = 1; stride < min(blockDim.x, arr_size); stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

__global__ void gpu_process_image(uchar *in, uchar *out) {
    __shared__ int histogram[256];
    __shared__ int hist_min[256];

    int tid = threadIdx.x;

    if (tid < 256) {
        histogram[tid] = 0;
    }
    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x)
        atomicAdd(&histogram[in[i]], 1);

    __syncthreads();

    prefix_sum(histogram, 256);

    if (tid < 256) {
        hist_min[tid] = histogram[tid];
    }
    __syncthreads();

    int cdf_min = arr_min(hist_min, 256);

    __shared__ uchar map[256];
    if (tid < 256) {
        int map_value = (float)(histogram[tid] - cdf_min) / (SQR(IMG_DIMENSION) - cdf_min) * 255;
        map[tid] = (uchar)map_value;
    }

    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x) {
        out[i] = map[in[i]];
    }
    return;
}

void print_usage_and_die(char *progname) {
    printf("usage:\n");
    printf("%s streams <load (requests/sec)>\n", progname);
    printf("OR\n");
    printf("%s queue <#threads> <load (requests/sec)>\n", progname);
    exit(1);
}


enum {PROGRAM_MODE_STREAMS = 0, PROGRAM_MODE_QUEUE};
int main(int argc, char *argv[]) {

    int mode = -1;
    int threads_queue_mode = -1; /* valid only when mode = queue */
    double load = 0;
    if (argc < 3) print_usage_and_die(argv[0]);

    if (!strcmp(argv[1], "streams")) {
        if (argc != 3) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_STREAMS;
        load = atof(argv[2]);
    } else if (!strcmp(argv[1], "queue")) {
        if (argc != 4) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_QUEUE;
        threads_queue_mode = atoi(argv[2]);
        load = atof(argv[3]);
    } else {
        print_usage_and_die(argv[0]);
    }

    uchar *images_in; /* we concatenate all images in one huge array */
    uchar *images_out;
    CUDA_CHECK( cudaHostAlloc(&images_in, NREQUESTS * SQR(IMG_DIMENSION), 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out, NREQUESTS * SQR(IMG_DIMENSION), 0) );

    load_images(images_in);
    double t_start, t_finish;

    /* using CPU */
    printf("\n=== CPU ===\n");
    t_start  = get_time_msec();
    for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx)
        process_image(&images_in[img_idx * SQR(IMG_DIMENSION)],
                &images_out[img_idx * SQR(IMG_DIMENSION)]);
    t_finish = get_time_msec();
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);

    double total_distance = 0;

    /* using GPU task-serial.. just to verify the GPU code makes sense */
    printf("\n=== GPU Task Serial ===\n");

    uchar *images_out_from_gpu;
    CUDA_CHECK( cudaHostAlloc(&images_out_from_gpu, NREQUESTS * SQR(IMG_DIMENSION), 0) );

    do {
        uchar *gpu_image_in, *gpu_image_out;
        CUDA_CHECK(cudaMalloc(&gpu_image_in, SQR(IMG_DIMENSION)));
        CUDA_CHECK(cudaMalloc(&gpu_image_out, SQR(IMG_DIMENSION)));

        t_start = get_time_msec();
        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {
            CUDA_CHECK(cudaMemcpy(gpu_image_in, &images_in[img_idx * SQR(IMG_DIMENSION)],
                        SQR(IMG_DIMENSION), cudaMemcpyHostToDevice));
            gpu_process_image<<<1, 1024>>>(gpu_image_in, gpu_image_out);
            CUDA_CHECK(cudaMemcpy(&images_out_from_gpu[img_idx * SQR(IMG_DIMENSION)],
                        gpu_image_out, SQR(IMG_DIMENSION), cudaMemcpyDeviceToHost));
        }
        total_distance += distance_sqr_between_image_arrays(images_out, images_out_from_gpu);
        CUDA_CHECK(cudaDeviceSynchronize());
        t_finish = get_time_msec();
        printf("distance from baseline %lf (should be zero)\n", total_distance);
        printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);

        CUDA_CHECK(cudaFree(gpu_image_in));
        CUDA_CHECK(cudaFree(gpu_image_out));
    } while (0);

    /* now for the client-server part */
    printf("\n=== Client-Server ===\n");
    double *req_t_start = (double *) malloc(NREQUESTS * sizeof(double));
    memset(req_t_start, 0, NREQUESTS * sizeof(double));

    double *req_t_end = (double *) malloc(NREQUESTS * sizeof(double));
    memset(req_t_end, 0, NREQUESTS * sizeof(double));

    struct rate_limit_t rate_limit;
    rate_limit_init(&rate_limit, load, 0);

    /* TODO allocate / initialize memory, streams, etc... */
    CUDA_CHECK(cudaMemset(images_out_from_gpu, 0, NREQUESTS * SQR(IMG_DIMENSION)));

    double ti = get_time_msec();
    if (mode == PROGRAM_MODE_STREAMS) {
        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {

            /* TODO query (don't block) streams for any completed requests.
             * update req_t_end of completed requests
             */

            rate_limit_wait(&rate_limit);
            req_t_start[img_idx] = get_time_msec();

            /* TODO place memcpy's and kernels in a stream */
        }
        /* TODO now make sure to wait for all streams to finish */

    } else if (mode == PROGRAM_MODE_QUEUE) {
        // TODO launch GPU consumer-producer kernel
        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {
            /* TODO check producer consumer queue for any responses.
             * don't block. if no responses are there we'll check again in the next iteration
             * update req_t_end of completed requests 
             */

            rate_limit_wait(&rate_limit);
            req_t_start[img_idx] = get_time_msec();

            /* TODO push task to queue */
        }
        /* TODO wait until you have responses for all requests */
    } else {
        assert(0);
    }
    double tf = get_time_msec();

    total_distance = distance_sqr_between_image_arrays(images_out, images_out_from_gpu);
    double avg_latency = 0;
    for (int i = 0; i < NREQUESTS; i++) {
        avg_latency += (req_t_end[i] - req_t_start[i]);
    }
    avg_latency /= NREQUESTS;

    printf("mode = %s\n", mode == PROGRAM_MODE_STREAMS ? "streams" : "queue");
    printf("load = %lf (req/sec)\n", load);
    if (mode == PROGRAM_MODE_QUEUE) printf("threads = %d\n", threads_queue_mode);
    printf("distance from baseline %lf (should be zero)\n", total_distance);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (tf - ti) * 1e+3);
    printf("average latency = %lf (msec)\n", avg_latency);

    return 0;
}
