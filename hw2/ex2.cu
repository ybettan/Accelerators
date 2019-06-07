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

__device__ void gpu_process_image_aux(uchar *in, uchar *out) {
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

__global__ void gpu_process_image(uchar *in, uchar *out) {
    gpu_process_image_aux(in, out);
}

void print_usage_and_die(char *progname) {
    printf("usage:\n");
    printf("%s streams <load (requests/sec)>\n", progname);
    printf("OR\n");
    printf("%s queue <#threads> <load (requests/sec)>\n", progname);
    exit(1);
}

//-----------------------------------------------------------------------------
//                              Streams Aux
//-----------------------------------------------------------------------------

#define NSTREAMS 64
#define AVAILABLE -1

bool is_stream_available(cudaStream_t *streams, int *stream_to_img, int idx) {
    bool cond1 = cudaStreamQuery(streams[idx]) == cudaSuccess;
    bool cond2 = stream_to_img[idx] == AVAILABLE;
    return cond1 && cond2;
}

bool are_all_streams_available(cudaStream_t *streams, int *stream_to_img) {
    for (int i=0 ; i<NSTREAMS ; i++) {
        if (!is_stream_available(streams, stream_to_img, i))
            return false;
    }
    return true;
}

/*
 * iterate over the streams and check if any of them has finished their tasks,
 * if so the end time of the images that have been proccesed is recorded.
 */
void check_completed_requests(cudaStream_t *streams, int *stream_to_img,
        double *req_t_end) {

    int img_idx;
    for (int i=0 ; i<NSTREAMS ; i++) {
        if (cudaStreamQuery(streams[i]) == cudaSuccess) {
            img_idx = stream_to_img[i];
            if (img_idx != AVAILABLE) {
                /* stream[i] has finished its task */
                req_t_end[img_idx] = get_time_msec();
                stream_to_img[i] = AVAILABLE;
            }
            /* stream[i] didn't have any taks in process */
        }
    }
}

/* blocking until an available stream is found */
int get_available_stream_idx(cudaStream_t *streams, int *stream_to_img,
        double *req_t_end) {

    int stream_idx = 0;
    while (!is_stream_available(streams, stream_to_img, stream_idx)) {
        stream_idx = (stream_idx + 1) % NSTREAMS;
        //FIXME: can we make it bether? it is done in the new iteration anyway
        /* all streams are buisy, check if some of them has finished */
        if (stream_idx == 0)
            check_completed_requests(streams, stream_to_img, req_t_end);
    }
    return stream_idx;
}

void wait_streams_done(cudaStream_t *streams, int *stream_to_img,
        double *req_t_end) {

    while (!are_all_streams_available(streams, stream_to_img)) {
        check_completed_requests(streams, stream_to_img, req_t_end);
    }
}

//-----------------------------------------------------------------------------
//                        Produce-Consumer Aux
//-----------------------------------------------------------------------------

#define QUEUE_SIZE 10
#define STOP -1

int get_num_concurrent_TBs(int threads_queue_mode) {

    /* sinlge block utilization */
    int shared_memory_per_block = 2*256*sizeof(int) + 256 * sizeof(uchar) + sizeof(int);
    int threads_per_block = threads_queue_mode;
    int registers_per_block = 32 * threads_per_block;

    /* HW limitaion */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int shared_memory_per_sm = prop.sharedMemPerMultiprocessor;
    int threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int registers_per_sm = prop.regsPerMultiprocessor;
    int num_sm = prop.multiProcessorCount;

    int res1 = shared_memory_per_sm / shared_memory_per_block;
    int res2 = registers_per_sm / registers_per_block;
    int res3 = threads_per_sm / threads_per_block;

    int res = min(min(res1, res2), res3) * num_sm;
    return res;
}

typedef struct Queue {
    int head, tail, cpu_cnt, gpu_cnt;
    int arr[QUEUE_SIZE];
} Queue;

void queue_init_cpu_side(Queue *queue) {
    queue->head = 0;
    queue->tail = -1;
    queue->cpu_cnt = 0;
    queue->gpu_cnt = 0;
    __sync_synchronize();
}

int queue_get_size_cpu_side(Queue *queue) {
    return abs(queue->cpu_cnt - queue->gpu_cnt);
}

bool queue_is_full_cpu_side(Queue *queue) {
    return queue_get_size_cpu_side(queue) == QUEUE_SIZE;
}

bool queue_is_empty_cpu_side(Queue *queue) {
    return queue_get_size_cpu_side(queue) == 0;
}

/* assumes the queue isn't full */
void queue_enqueue_cpu_side(Queue *queue, int img_idx) {

    assert(!queue_is_full_cpu_side(queue));

    queue->tail = (queue->tail + 1) % QUEUE_SIZE;
    queue->arr[queue->tail] = img_idx;
    queue->cpu_cnt++;

    __sync_synchronize();
}

/* assumes the queue isn't empty */
void queue_dequeue_cpu_side(Queue *queue, int *img_idx) {

    assert(!queue_is_empty_cpu_side(queue));

    *img_idx = queue->arr[queue->head];
    queue->head = (queue->head + 1) % QUEUE_SIZE;
    queue->cpu_cnt++;

    __sync_synchronize();
}

__device__ int queue_get_size_gpu_side(Queue *queue) {
    return abs(queue->cpu_cnt - queue->gpu_cnt);
}

__device__ bool queue_is_full_gpu_side(Queue *queue) {
    return queue_get_size_gpu_side(queue) == QUEUE_SIZE;
}

__device__ bool queue_is_empty_gpu_side(Queue *queue) {
    return queue_get_size_gpu_side(queue) == 0;
}

/* assumes the queue isn't empty, onley thread 0 is enqueuing */
__device__ void queue_enqueue_gpu_side(Queue *queue, int img_idx) {

    int tid = threadIdx.x;

    if (tid == 0) {
        __threadfence();
        assert(!queue_is_full_gpu_side(queue));
    }
    __threadfence();

    if (tid == 0) {
        __threadfence();
        queue->tail = (queue->tail + 1) % QUEUE_SIZE;
        __threadfence();
        queue->arr[queue->tail] = img_idx;
        __threadfence();
        queue->gpu_cnt++;
    }
    __threadfence_system();
    __syncthreads();
}

/* assumes the queue isn't empty, only thread 0 is dequeuing */
__device__ void queue_dequeue_gpu_side(Queue *queue, int *img_idx) {

    int tid = threadIdx.x;

    if (tid == 0) {
        __threadfence();
        assert(!queue_is_empty_gpu_side(queue));
    }
    __threadfence();

    if (tid == 0) {
        __threadfence();
        *img_idx = queue->arr[queue->head];
        __threadfence();
        queue->head = (queue->head + 1) % QUEUE_SIZE;
    }
    __threadfence();

    if (tid == 0) {
        __threadfence();
        queue->gpu_cnt++;
    }
    __threadfence_system();
    __syncthreads();
}

/*
 * iterate over the gpu-cpu queues and check for img_idx that have been processed,
 * if so the end time of the images that have been proccesed is recorded.
 * return true if all images has been processed and false otherwise.
 */
bool check_completed_requests_cpu_side(Queue *gpu_cpu_queues_cpu_ptr,
        int num_threadblocks, double *req_t_end, bool *threadblocks_done) {

    int img_idx;
    for (int i=0 ; i<num_threadblocks ; i++) {

        while (!queue_is_empty_cpu_side(&gpu_cpu_queues_cpu_ptr[i])) {

            queue_dequeue_cpu_side(&gpu_cpu_queues_cpu_ptr[i], &img_idx);

            if (img_idx == STOP) {
                threadblocks_done[i] = true;
            } else {
                req_t_end[img_idx] = get_time_msec();
            }
            __sync_synchronize();
        }
    }

    for (int i=0 ; i<num_threadblocks ; i++) {
        if (threadblocks_done[i] == false) {
            return false;
        }
    }
    __sync_synchronize();
    return true;
}

__global__ void producer_consumer_loop(uchar *images_in, uchar *images_out,
        Queue *cpu_gpu_queues_gpu_ptr, Queue *gpu_cpu_queues_gpu_ptr) {

    __shared__ int img_idx;
    int bid = blockIdx.x;
    uchar *in, *out;
    Queue *cpu_gpu_queue_gpu_ptr = &cpu_gpu_queues_gpu_ptr[bid];
    Queue *gpu_cpu_queue_gpu_ptr = &gpu_cpu_queues_gpu_ptr[bid];

    while (true) {

        /* buisy wait to requests if stil no request arrived */
        while (queue_is_empty_gpu_side(cpu_gpu_queue_gpu_ptr)) {
            __threadfence_system();
        }

        /* only thread 0 in fact make the dequeue and return after __syncthread() */
        queue_dequeue_gpu_side(cpu_gpu_queue_gpu_ptr, &img_idx);

        __threadfence_system();

        /* if block got a STOP msg its work is done */
        if (img_idx == STOP) {

            __threadfence_system();

            /* buisy wait until gpu-cpu queue can enqueue */
            while (queue_is_full_gpu_side(gpu_cpu_queue_gpu_ptr)) {
                __threadfence_system();
            }

            __threadfence_system();

            /* enqueue the processed img_idx */
            queue_enqueue_gpu_side(gpu_cpu_queue_gpu_ptr, STOP);

            __threadfence_system();

            break;
        }

        __threadfence_system();

        /* compute the ptrs according to the img id request */
        in = &images_in[img_idx * SQR(IMG_DIMENSION)];
        out = &images_out[img_idx * SQR(IMG_DIMENSION)];

        __threadfence_system();

        /* process the image */
        gpu_process_image_aux(in, out);

        __threadfence_system();

        /* buisy wait until gpu-cpu queue can enqueue */
        while (queue_is_full_gpu_side(gpu_cpu_queue_gpu_ptr)) {
            __threadfence_system();
        }

        __threadfence_system();

        /* enqueue the processed img_idx */
        queue_enqueue_gpu_side(gpu_cpu_queue_gpu_ptr, img_idx);

        __threadfence_system();
        __syncthreads();
    }
    __threadfence_system();
    __syncthreads();
}

//-----------------------------------------------------------------------------


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

    /* allocate all memory needed for this part */
    uchar *gpu_images_in, *gpu_images_out;

    int stream_to_img[NSTREAMS];
    for (int i=0 ; i<NSTREAMS ; i++)
        stream_to_img[i] = AVAILABLE;

    CUDA_CHECK(cudaMalloc(&gpu_images_in, NREQUESTS * SQR(IMG_DIMENSION)));
    CUDA_CHECK(cudaMalloc(&gpu_images_out, NREQUESTS * SQR(IMG_DIMENSION)));

    /* reset the GPU-Serial results */
    CUDA_CHECK(cudaMemset(images_out_from_gpu, 0, NREQUESTS * SQR(IMG_DIMENSION)));

    double ti = get_time_msec();
    if (mode == PROGRAM_MODE_STREAMS) {

        /* create the streams */
        cudaStream_t streams[NSTREAMS];
        for (int i=0 ; i<NSTREAMS ; i++)
            CUDA_CHECK(cudaStreamCreate(&streams[i]));

        int stream_idx, img_offset;
        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {

            /* check and record end-time for completed requests */
            check_completed_requests(streams, stream_to_img, req_t_end);

            /* wait for the next client "request" */
            if (!rate_limit_can_send(&rate_limit)) {
                --img_idx;
                continue;
            }

            /* record start-time for the new request */
            req_t_start[img_idx] = get_time_msec();

            /* assign work to the available stream */
            stream_idx = get_available_stream_idx(streams, stream_to_img, req_t_end);
            stream_to_img[stream_idx] = img_idx;
            img_offset = img_idx * SQR(IMG_DIMENSION);
            CUDA_CHECK(cudaMemcpyAsync(&gpu_images_in[img_offset], &images_in[img_offset],
                        SQR(IMG_DIMENSION), cudaMemcpyHostToDevice, streams[stream_idx]));
            gpu_process_image<<<1, 1024, 0, streams[stream_idx]>>>
                (&gpu_images_in[img_offset], &gpu_images_out[img_offset]);
            CUDA_CHECK(cudaMemcpyAsync(&images_out_from_gpu[img_offset],
                        &gpu_images_out[img_offset], SQR(IMG_DIMENSION),
                        cudaMemcpyDeviceToHost, streams[stream_idx]));
        }

        /* wait for all streams to finish */
        wait_streams_done(streams, stream_to_img, req_t_end);

        /* free streams */
        for (int i=0 ; i<NSTREAMS ; i++)
            CUDA_CHECK(cudaStreamDestroy(streams[i]));

    } else if (mode == PROGRAM_MODE_QUEUE) {

        /* compute num of TBs that can run concurentlly */
        int num_threadblocks = get_num_concurrent_TBs(threads_queue_mode);

        Queue *cpu_gpu_queues_cpu_ptr, *cpu_gpu_queues_gpu_ptr;
        Queue *gpu_cpu_queues_cpu_ptr, *gpu_cpu_queues_gpu_ptr;

        uchar *images_in_gpu_ptr, *images_out_from_gpu_gpu_ptr;
        bool *threadblocks_done;
        CUDA_CHECK( cudaHostAlloc(&threadblocks_done, num_threadblocks * sizeof(bool), 0) );
        for (int i=0 ; i<num_threadblocks ; i++) {
            threadblocks_done[i] = false;
        }

        /* allocate the producer-consumer queues in CPU-GPU shared memory */
        CUDA_CHECK( cudaHostAlloc(&cpu_gpu_queues_cpu_ptr,
                    num_threadblocks * sizeof(Queue), 0) );
        CUDA_CHECK( cudaHostAlloc(&gpu_cpu_queues_cpu_ptr,
                    num_threadblocks * sizeof(Queue), 0) );

        /* give the GPU ptrs to those queues */
        CUDA_CHECK( cudaHostGetDevicePointer(&cpu_gpu_queues_gpu_ptr,
                    cpu_gpu_queues_cpu_ptr, 0) );
        CUDA_CHECK( cudaHostGetDevicePointer(&gpu_cpu_queues_gpu_ptr,
                    gpu_cpu_queues_cpu_ptr, 0) );

        /* give the GPU ptrs to the images */
        CUDA_CHECK( cudaHostGetDevicePointer(&images_in_gpu_ptr, images_in, 0) );
        CUDA_CHECK( cudaHostGetDevicePointer(&images_out_from_gpu_gpu_ptr,
                    images_out_from_gpu, 0) );

        /* initialize the queues */
        for (int i=0 ; i<num_threadblocks ; i++) {
            queue_init_cpu_side(&cpu_gpu_queues_cpu_ptr[i]);
            queue_init_cpu_side(&gpu_cpu_queues_cpu_ptr[i]);
        }

        /* lunch the kernel producer consumer loop */
        producer_consumer_loop<<<num_threadblocks, threads_queue_mode>>>
            (images_in_gpu_ptr, images_out_from_gpu_gpu_ptr,
             cpu_gpu_queues_gpu_ptr, gpu_cpu_queues_gpu_ptr);

        /* send all the requests to the queues */
        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {

            /* check for gpu response */
            check_completed_requests_cpu_side(gpu_cpu_queues_cpu_ptr,
                    num_threadblocks, req_t_end, threadblocks_done);

            /* wait for the next request */
            if (!rate_limit_can_send(&rate_limit)) {
                --img_idx;
                continue;
            }
            req_t_start[img_idx] = get_time_msec();

            /* push task to queue */
            int bid = img_idx % num_threadblocks;
            while (queue_is_full_cpu_side(&cpu_gpu_queues_cpu_ptr[bid])) {
                __sync_synchronize();
            }
            queue_enqueue_cpu_side(&cpu_gpu_queues_cpu_ptr[bid], img_idx);
        }

        /* notify all blocks that there are no more requests */
        for (int i=0 ; i<num_threadblocks ; i++) {

            /* check for gpu response */
            check_completed_requests_cpu_side(gpu_cpu_queues_cpu_ptr,
                    num_threadblocks, req_t_end, threadblocks_done);

            while (queue_is_full_cpu_side(&cpu_gpu_queues_cpu_ptr[i])) {
                __sync_synchronize();
            }
            queue_enqueue_cpu_side(&cpu_gpu_queues_cpu_ptr[i], STOP);
        }

        /* wait until all requests have been process */
        while (!check_completed_requests_cpu_side(gpu_cpu_queues_cpu_ptr,
                num_threadblocks, req_t_end, threadblocks_done));

        /* free queues memory */
        CUDA_CHECK( cudaFreeHost(cpu_gpu_queues_cpu_ptr) );
        CUDA_CHECK( cudaFreeHost(gpu_cpu_queues_cpu_ptr) );
        CUDA_CHECK( cudaFreeHost(threadblocks_done) );

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

    /* free all memory allocated */
    CUDA_CHECK(cudaFree(gpu_images_in));
    CUDA_CHECK(cudaFree(gpu_images_out));

    return 0;
}
