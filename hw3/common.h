#pragma once

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

///////////////////////////////// DO NOT CHANGE ////////////////////////////////

#define IMG_DIMENSION 32
//FIXME: is this ok ?!
//#define OUTSTANDING_REQUESTS 100
#define OUTSTANDING_REQUESTS 10000

#define SQR(a) ((a) * (a))

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char uchar;

double static inline get_time_msec(void) {
    struct timespec t;
    int res = clock_gettime(CLOCK_MONOTONIC, &t);
    if (res) {
        perror("clock_gettime failed");
        exit(1);
    }
    return t.tv_sec * 1e+3 + t.tv_nsec * 1e-6;
}

struct rpc_request
{
    /* Returned to the client via RDMA write immediate value; use -1 to terminate */
    int request_id;

    /* Input buffer */
    int input_rkey;
    int input_length;
    uint64_t input_addr;

    /* Output buffer */
    int output_rkey;
    int output_length;
    uint64_t output_addr;
};

#define IB_PORT_SERVER 1
#define IB_PORT_CLIENT 2

//=============================================================================
//                              Queue hw2
//=============================================================================

#define QUEUE_SIZE 10
#define STOP -1

typedef struct Queue {
    int head, tail, cpu_cnt, gpu_cnt;
    int arr[QUEUE_SIZE];
} Queue;

//-----------------------------------------------------------------------------

/////////////////////////////////////////////////////////////////////////////////////////////////////

struct ib_info_t {
    int lid;
    int qpn;

    /* queues */
    int num_threadblocks;
    int cpu_gpu_queues_rkey;
    int gpu_cpu_queues_rkey;
    Queue *cpu_gpu_queues_addr;
    Queue *gpu_cpu_queues_addr;

    /* images */
    int images_in_rkey;
    int images_out_rkey;
    uchar *images_in_addr;
    uchar *images_out_addr;

};

enum mode_enum {
    MODE_RPC_SERVER,
    MODE_QUEUE,
};

void parse_arguments(int argc, char **argv, enum mode_enum *mode, int *tcp_port);


#ifdef __cplusplus
}
#endif
