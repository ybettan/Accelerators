/* compile with: nvcc -O3 hw1.cu -o hw1 */

#include <stdio.h>
#include <sys/time.h>
#include <assert.h>

#define INF (1<<16 - 1)

////////////////////////////// DO NOT CHANGE //////////////////////////////////
#define IMG_HEIGHT 256
#define IMG_WIDTH 256
#define N_IMAGES 10000

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
    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
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
        int map_value = (float)(cdf[i] - cdf_min) / (IMG_WIDTH * IMG_HEIGHT - cdf_min) * 255;
        map[i] = (uchar)map_value;
    }

    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
        img_out[i] = map[img_in[i]];
    }
}

double static inline get_time_msec(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
}

long long int distance_sqr_between_image_arrays(uchar *img_arr1, uchar *img_arr2) {
    long long int distance_sqr = 0;
    for (int i = 0; i < N_IMAGES * IMG_WIDTH * IMG_HEIGHT; i++) {
        distance_sqr += SQR(img_arr1[i] - img_arr2[i]);
    }
    return distance_sqr;
}
///////////////////////////////////////////////////////////////////////////////

//__device__ int* compute_histogram(uchar *img, uchar *out) {
//
//     /* use the first 256*sizeof(int) byte of out argument to store the global
//      * histogram */
//    int *global_hist = (int*)out;
//    int tid = blockIdx.x * blockDim.x + blockIdx.x;
//
//    /* each threadBlock have its own local histogram */
//    __shared__ int local_hist[256];
//
//    /* initialize the local histogram */
//    if (tid < 256) {
//        local_hist[tid] = 0;
//    }
//    __syncthreads();
//
//    /* compute the local histogram */
//    atomicAdd(&local_hist[tid], 1);
//    __syncthreads();
//
//    /* update the global histogram */
//    if (tid < 256) {
//        atomicAdd(&global_hist[tid], local_hist[tid]);
//    }
//    __syncthreads();
//
//    /* return the global histogram addr */
//    return global_hist;
//}

//__global__ void process_image_kernel(uchar *in, uchar *out) {
//
//    int tid = threadIdx.x;
//    int index = blockIdx.x * blockDim.x + threadIdx.x;
//    int cdf_min;
//    __shared__ int cdf[256], m[256];
//
//    ///* compute the global histogram */
//    //int *global_hist = compute_histogram(in, out);
//
//    //FIXME: remove
//    if (tid == 0)
//        printf("blockDim = %d\tblockIdx = %d\n", blockDim.x, blockIdx.x);
//        //printf("index = %d\tblockId = %d\ttid = %d\n", index, blockIdx.x, tid);
//    //if (index == 0) {
//    //    for (int i=0 ; i<256 ; i++) {
//    //        printf("global_hist[%d] = %d\n", global_hist[i]);
//    //    }
//    //}
//
//    ///* copy the global histogram localy */
//    //if (tid < 256) {
//    //    cdf[tid] = global_hist[tid];
//    //}
//    //__syncthreads();
//
//    ///* compute the prefix sum */
//    //prefix_sum(cdf, 256);
//
//    ///* find cdf-min, and save it in register */
//    //cdf_min = arr_min(cdf, 256);
//
//    ///* compute the map function to remape img values */
//    //if (tid < 256) {
//    //    m[tid] = 255 * (cdf[tid] - cdf_min) / (IMG_HEIGHT * IMG_WIDTH - cdf_min);
//    //}
//    //__syncthreads();
//
//    ///* process the img */
//    //out[index] = m[in[index]];
//    //__syncthreads();
//}

__device__ void compute_histogram(uchar *img, int *res, int res_size) {

    int tid = threadIdx.x;
    int work_per_thread = (IMG_WIDTH * IMG_HEIGHT) / blockDim.x;
    

    /* initialize the histogram */
    if (tid < 256) {
        res[tid] = 0;
    }
    __syncthreads();

    /* compute the histogram */
    int index;
    for (int i=0 ; i<work_per_thread ; i++) {
        index = blockDim.x * i + tid;
        atomicAdd(&res[img[index]], 1);
    }
    __syncthreads();
}

__device__ void prefix_sum(int arr[], int arr_size) {

    int tid = threadIdx.x;
    int increment;

    for (int stride=1 ; stride<arr_size ; stride*=2) {
    
        if (tid < arr_size && tid >= stride) {
            increment = arr[tid-stride];
        }
        __syncthreads();

        if (tid < arr_size && tid >= stride) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

__device__ void arr_min(int arr[], int arr_size, int *res) {

    int tid = threadIdx.x;

    /* initialize res to 'inf' */
    if (tid == 0) {
        *res = INF;
    }
    __syncthreads();

    if (tid < arr_size)
        arr[tid] && atomicMin(res, arr[tid]);
    __syncthreads();
}

__device__ void compute_map(int *cdf, int cdf_positive_min, uchar *map) {

    int tid = threadIdx.x;

    if (tid < 256) {
        map[tid] = (float)(cdf[tid] - cdf_positive_min) /
            (IMG_HEIGHT * IMG_WIDTH - cdf_positive_min) * 255;
    }
    __syncthreads();
}

__device__ void remap_img(uchar *in, uchar *out, uchar *map) {

    int tid = threadIdx.x;
    int work_per_thread = (IMG_WIDTH * IMG_HEIGHT) / blockDim.x;
    
    /* remap the image */
    int index;
    for (int i=0 ; i<work_per_thread ; i++) {
        index = blockDim.x * i + tid;
        out[index] = map[in[index]];
    }
    __syncthreads();
}

/* process a single image by a single threadBlock */
__global__ void process_image_kernel(uchar *in, uchar *out) {

    __shared__ int cdf[256];
    __shared__ uchar m[256];
    __shared__ int cdf_positive_min;

    compute_histogram(in, cdf, 256);
    prefix_sum(cdf, 256);
    arr_min(cdf, 256, &cdf_positive_min);
    compute_map(cdf, cdf_positive_min, m);
    remap_img(in, out, m);
}


int main() {
////////////////////////////// DO NOT CHANGE //////////////////////////////////
    uchar *images_in;
    uchar *images_out_cpu; //output of CPU computation. In CPU memory.
    uchar *images_out_gpu_serial; //output of GPU task serial computation. In CPU memory.
    uchar *images_out_gpu_bulk; //output of GPU bulk computation. In CPU memory.
    CUDA_CHECK( cudaHostAlloc(&images_in, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_cpu, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_gpu_serial, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_gpu_bulk, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );

    /* instead of loading real images, we'll load the arrays with random data */
    srand(0);
    for (long long int i = 0; i < N_IMAGES * IMG_WIDTH * IMG_HEIGHT; i++) {
        images_in[i] = rand() % 256;
    }

    double t_start, t_finish;

    // CPU computation. For reference. Do not change
    printf("\n=== CPU ===\n");
    t_start = get_time_msec();
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img_in = &images_in[i * IMG_WIDTH * IMG_HEIGHT];
        uchar *img_out = &images_out_cpu[i * IMG_WIDTH * IMG_HEIGHT];
        process_image(img_in, img_out);
    }
    t_finish = get_time_msec();
    printf("total time %f [msec]\n", t_finish - t_start);

    long long int distance_sqr;
///////////////////////////////////////////////////////////////////////////////

    // GPU task serial computation
    printf("\n=== GPU Task Serial ===\n"); //Do not change

    uchar *device_img_in = NULL;
    uchar *device_img_out = NULL;

    /* allocate memeory on the GPU global memory */
    CUDA_CHECK(cudaMalloc(&device_img_in, IMG_WIDTH * IMG_HEIGHT));
    CUDA_CHECK(cudaMalloc(&device_img_out, IMG_WIDTH * IMG_HEIGHT));

    t_start = get_time_msec(); //Do not change

    for (int i=0 ; i<N_IMAGES ; i++) {

        uchar *img_in = &images_in[i * IMG_WIDTH * IMG_HEIGHT];
        uchar *img_out = &images_out_gpu_serial[i * IMG_WIDTH * IMG_HEIGHT];

        /* copy the relevant image from images_in to the GPU memory allocated */
        CUDA_CHECK(cudaMemcpy(device_img_in, img_in, IMG_WIDTH * IMG_HEIGHT,
                    cudaMemcpyHostToDevice));

        int blocks = 1;
        int threads_in_block = 1024;

        /* invoke the GPU kernel */
        process_image_kernel<<<blocks, threads_in_block>>>(device_img_in, device_img_out);

        /* copy output from GPU memory to relevant location in images_out_gpu_serial */
        CUDA_CHECK(cudaMemcpy(img_out, device_img_out, IMG_WIDTH * IMG_HEIGHT,
                    cudaMemcpyDeviceToHost));
    }

    t_finish = get_time_msec(); //Do not change

    /* free the GPU global memory allocated */
    CUDA_CHECK(cudaFree(device_img_in));
    CUDA_CHECK(cudaFree(device_img_out));

    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu,
            images_out_gpu_serial); // Do not change
    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n",
            t_finish - t_start, distance_sqr); //Do not change

    // GPU bulk
    printf("\n=== GPU Bulk ===\n"); //Do not change
    //TODO: allocate GPU memory for a all input images and all output images
    t_start = get_time_msec(); //Do not change
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out_gpu_bulk
    t_finish = get_time_msec(); //Do not change
    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu,
            images_out_gpu_bulk); // Do not change
    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n",
            t_finish - t_start, distance_sqr); //Do not chhange

    //FIXME: remove
    /* check correctnes */
    printf("\n=== CORECTNESS ===\n");
    for (int i=0 ; i<N_IMAGES * IMG_HEIGHT * IMG_WIDTH ; i++) {
        if (images_out_cpu[i] != images_out_gpu_serial[i]) {
            printf("first wrong index is: %d\n", i);
            printf("in[%d] = %d\n", i, images_in[i]);
            printf("out_cpu[%d] = %d\n", i, images_out_cpu[i]);
            printf("out_gpu[%d] = %d\n", i, images_out_gpu_serial[i]);
        }
        assert(images_out_cpu[i] == images_out_gpu_serial[i]);
        //assert(images_out_cpu[i] == images_out_gpu_bulk[i]);
    }

    /* free allocated memory */
    CUDA_CHECK(cudaFreeHost(images_in));
    CUDA_CHECK(cudaFreeHost(images_out_cpu));
    CUDA_CHECK(cudaFreeHost(images_out_gpu_serial));
    CUDA_CHECK(cudaFreeHost(images_out_gpu_bulk));

    return 0;
}
