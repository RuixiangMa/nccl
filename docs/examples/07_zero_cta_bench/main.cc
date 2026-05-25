#include "cuda_runtime.h"
#include "nccl.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define WARMUP_ITERS 5
#define BENCH_ITERS 20

#define NCCLCHECK(cmd) do { ncclResult_t _r = cmd; if(_r!=ncclSuccess) { fprintf(stderr, "NCCL FAIL: %s line %d %s\n", #cmd, __LINE__, ncclGetErrorString(_r)); exit(1); } } while(0)
#define CUDACHECK(cmd) do { cudaError_t _e = cmd; if(_e!=cudaSuccess) { fprintf(stderr, "CUDA FAIL: %s line %d %s\n", #cmd, __LINE__, cudaGetErrorString(_e)); exit(1); } } while(0)

typedef enum {
    COLL_ALLGATHER = 0,
    COLL_ALLTOALL,
    COLL_SCATTER,
    COLL_GATHER,
    COLL_COUNT
} coll_type_t;

static const char *coll_names[COLL_COUNT] = {
    "AllGather", "AlltoAll", "Scatter", "Gather",
};

static size_t data_sizes[] = {
    1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20,
    1 << 22, 1 << 24, 1 << 26,
};
#define NUM_DATA_SIZES (sizeof(data_sizes) / sizeof(data_sizes[0]))

static ncclUniqueId g_nccl_id;
static pthread_barrier_t g_barrier;

typedef struct {
    int rank;
    int nranks;
    int device;
} thread_arg_t;

typedef struct {
    double avg_us;
    double min_us;
    double bw_gbs;
} bench_result_t;

static void broadcast_id(int root, int rank, ncclUniqueId *id) {
    if (rank == root) g_nccl_id = *id;
    pthread_barrier_wait(&g_barrier);
    if (rank != root) *id = g_nccl_id;
    pthread_barrier_wait(&g_barrier);
}

static double get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static ncclResult_t run_coll(ncclComm_t comm, cudaStream_t stream,
                             coll_type_t coll, void *sendbuf, void *recvbuf,
                             size_t count, ncclDataType_t dtype, int rank, int nranks) {
    switch (coll) {
    case COLL_ALLGATHER:
        return ncclAllGather(sendbuf, recvbuf, count, dtype, comm, stream);
    case COLL_ALLTOALL:
        return ncclAlltoAll(sendbuf, recvbuf, count, dtype, comm, stream);
    case COLL_SCATTER:
        return ncclScatter(sendbuf, recvbuf, count, dtype, 0, comm, stream);
    case COLL_GATHER:
        return ncclGather(sendbuf, recvbuf, count, dtype, 0, comm, stream);
    default:
        return ncclInternalError;
    }
}

static bench_result_t bench_coll(ncclComm_t comm, cudaStream_t stream,
                                 coll_type_t coll, void *sendbuf, void *recvbuf,
                                 size_t count, ncclDataType_t dtype, int rank, int nranks,
                                 size_t total_bytes) {
    bench_result_t res = {0};
    res.min_us = 1e18;

    for (int i = 0; i < WARMUP_ITERS; i++)
        NCCLCHECK(run_coll(comm, stream, coll, sendbuf, recvbuf, count, dtype, rank, nranks));
    CUDACHECK(cudaStreamSynchronize(stream));

    double total_us = 0;
    for (int i = 0; i < BENCH_ITERS; i++) {
        double t0 = get_time_us();
        NCCLCHECK(run_coll(comm, stream, coll, sendbuf, recvbuf, count, dtype, rank, nranks));
        CUDACHECK(cudaStreamSynchronize(stream));
        double t1 = get_time_us();
        double elapsed = t1 - t0;
        total_us += elapsed;
        if (elapsed < res.min_us) res.min_us = elapsed;
    }

    res.avg_us = total_us / BENCH_ITERS;
    res.bw_gbs = (total_bytes / (res.avg_us * 1e-6)) / 1e9;
    return res;
}

static void *run(void *arg) {
    thread_arg_t *a = (thread_arg_t *)arg;
    int rank = a->rank, nranks = a->nranks, dev = a->device;

    CUDACHECK(cudaSetDevice(dev));

    ncclUniqueId id1, id2;
    if (rank == 0) NCCLCHECK(ncclGetUniqueId(&id1));
    broadcast_id(0, rank, &id1);

    ncclComm_t comm_cta;
    NCCLCHECK(ncclCommInitRank(&comm_cta, nranks, id1, rank));
    fprintf(stderr, "[Rank %d] comm_cta init OK\n", rank); fflush(stderr);

    if (rank == 0) NCCLCHECK(ncclGetUniqueId(&id2));
    broadcast_id(0, rank, &id2);

    ncclComm_t comm_zcta;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
    NCCLCHECK(ncclCommInitRankConfig(&comm_zcta, nranks, id2, rank, &config));
    fprintf(stderr, "[Rank %d] comm_zcta init OK\n", rank); fflush(stderr);

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    size_t max_size = data_sizes[NUM_DATA_SIZES - 1];
    size_t max_alloc = max_size * nranks;

    void *sbuf_cta, *rbuf_cta, *sbuf_zcta, *rbuf_zcta;
    NCCLCHECK(ncclMemAlloc(&sbuf_cta, max_alloc));
    NCCLCHECK(ncclMemAlloc(&rbuf_cta, max_alloc));
    NCCLCHECK(ncclMemAlloc(&sbuf_zcta, max_alloc));
    NCCLCHECK(ncclMemAlloc(&rbuf_zcta, max_alloc));

    ncclWindow_t sw_cta, rw_cta, sw_zcta, rw_zcta;
    NCCLCHECK(ncclCommWindowRegister(comm_cta, sbuf_cta, max_alloc, &sw_cta, NCCL_WIN_COLL_SYMMETRIC));
    NCCLCHECK(ncclCommWindowRegister(comm_cta, rbuf_cta, max_alloc, &rw_cta, NCCL_WIN_COLL_SYMMETRIC));
    NCCLCHECK(ncclCommWindowRegister(comm_zcta, sbuf_zcta, max_alloc, &sw_zcta, NCCL_WIN_COLL_SYMMETRIC));
    NCCLCHECK(ncclCommWindowRegister(comm_zcta, rbuf_zcta, max_alloc, &rw_zcta, NCCL_WIN_COLL_SYMMETRIC));
    fprintf(stderr, "[Rank %d] WindowRegister OK\n", rank); fflush(stderr);

    if (rank == 0) {
        printf("\n============================================================\n");
        printf("  Zero-CTA Benchmark: %d GPUs, Warmup %d, Iters %d\n", nranks, WARMUP_ITERS, BENCH_ITERS);
        printf("  CTA = default | Zero-CTA = NCCL_CTA_POLICY_ZERO (CE)\n");
        printf("============================================================\n");
        printf("\n%-12s %-10s %12s %12s %10s %12s %12s %10s %8s\n",
               "Collective", "Size", "CTA Avg(us)", "CTA Min(us)", "CTA(GB/s)",
               "ZCTA Avg(us)", "ZCTA Min(us)", "ZCTA(GB/s)", "Speedup");
        printf("%-12s %-10s %12s %12s %10s %12s %12s %10s %8s\n",
               "----------", "------", "-----------", "-----------", "--------",
               "-----------", "-----------", "--------", "------");
        fflush(stdout);
    }

    for (int ci = 0; ci < COLL_COUNT; ci++) {
        for (int si = 0; si < (int)NUM_DATA_SIZES; si++) {
            size_t elem_count = data_sizes[si] / sizeof(float);
            size_t total_bytes = data_sizes[si] * nranks;

            bench_result_t res_cta = bench_coll(comm_cta, stream, (coll_type_t)ci,
                sbuf_cta, rbuf_cta, elem_count, ncclFloat, rank, nranks, total_bytes);

            bench_result_t res_zcta = bench_coll(comm_zcta, stream, (coll_type_t)ci,
                sbuf_zcta, rbuf_zcta, elem_count, ncclFloat, rank, nranks, total_bytes);

            if (rank == 0) {
                char size_str[32];
                if (data_sizes[si] >= (1<<20)) sprintf(size_str, "%zuMB", data_sizes[si]/(1<<20));
                else sprintf(size_str, "%zuKB", data_sizes[si]/(1<<10));

                double speedup = res_cta.avg_us / res_zcta.avg_us;
                printf("%-12s %-10s %12.1f %12.1f %10.2f %12.1f %12.1f %10.2f %8.2f\n",
                       coll_names[ci], size_str,
                       res_cta.avg_us, res_cta.min_us, res_cta.bw_gbs,
                       res_zcta.avg_us, res_zcta.min_us, res_zcta.bw_gbs,
                       speedup);
                fflush(stdout);
            }
        }
        if (rank == 0) { printf("\n"); fflush(stdout); }
    }

    if (rank == 0) {
        printf("============================================================\n");
        printf("  Benchmark complete. Speedup >1 means Zero-CTA is faster.\n");
        printf("============================================================\n");
        fflush(stdout);
    }

    NCCLCHECK(ncclCommWindowDeregister(comm_cta, sw_cta));
    NCCLCHECK(ncclCommWindowDeregister(comm_cta, rw_cta));
    NCCLCHECK(ncclCommWindowDeregister(comm_zcta, sw_zcta));
    NCCLCHECK(ncclCommWindowDeregister(comm_zcta, rw_zcta));
    NCCLCHECK(ncclMemFree(sbuf_cta)); NCCLCHECK(ncclMemFree(rbuf_cta));
    NCCLCHECK(ncclMemFree(sbuf_zcta)); NCCLCHECK(ncclMemFree(rbuf_zcta));
    CUDACHECK(cudaStreamDestroy(stream));
    NCCLCHECK(ncclCommFinalize(comm_cta)); NCCLCHECK(ncclCommDestroy(comm_cta));
    NCCLCHECK(ncclCommFinalize(comm_zcta)); NCCLCHECK(ncclCommDestroy(comm_zcta));

    return NULL;
}

int main() {
    int nranks = 2;
    pthread_barrier_init(&g_barrier, NULL, nranks);
    pthread_t threads[2];
    thread_arg_t args[2];
    for (int i = 0; i < nranks; i++) {
        args[i] = (thread_arg_t){i, nranks, i};
        pthread_create(&threads[i], NULL, run, &args[i]);
    }
    for (int i = 0; i < nranks; i++) pthread_join(threads[i], NULL);
    pthread_barrier_destroy(&g_barrier);
    return 0;
}
