/*
   Copyright (C) 2022 Jungbae Nam <jungbae.nam@gmail.com>

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 2 of the License, or
 (at your option) any later version.
                  https://www.gnu.org/licenses/
*/

#ifndef CUDA_API_PER_THREAD_DEFAULT_STEAM
#define CUDA_API_PER_THREAD_DEFAULT_STEAM
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuComplex.h>

#include "twists_dirichlet_character.h"
#include "twists_clve.cuh"


// these constants depend on the cuda hardware and compute capability (gtx 1080 Ti and cc 6.1 for the below).
const unsigned int max_k = 1024;
__constant__ double arr_z_k_real[max_k]; // k is assumed to be less than 1024.
__constant__ double arr_z_k_imag[max_k]; // k is assumed to be less than 1024.  


__host__ void twists_complex_set(twists_complex *a, twists_complex b)
{
    a->x = b.x;
    a->y = b.y;
}

__host__ void twists_cuda_primitive_root_of_unity(ulong k, twists_complex *p_root_1_ptr)
{
    double theta;

    theta = 2.0*twists_pi/((double)k);

    p_root_1_ptr->x = cos(theta);
    p_root_1_ptr->y = sin(theta);
}

__host__ void twists_cuda_array_z_k(ulong k, twists_arr_complex *arr_z_k)
{
    twists_complex z_k;
    ulong a = 2;

    twists_cuda_primitive_root_of_unity(k, &z_k);

    arr_z_k->len = (k+1);
    arr_z_k->x_ptr = (double *)flint_realloc(arr_z_k->x_ptr, (arr_z_k->len)*sizeof(double));
    arr_z_k->y_ptr = (double *)flint_realloc(arr_z_k->y_ptr, (arr_z_k->len)*sizeof(double));
    arr_z_k->x_ptr[0] = 0.0;
    arr_z_k->y_ptr[0] = 0.0;
    arr_z_k->x_ptr[1] = z_k.x;
    arr_z_k->y_ptr[1] = z_k.y;

    while(a <= k)
    {
        arr_z_k->x_ptr[a] = ((arr_z_k->x_ptr[a-1])*(z_k.x) - (arr_z_k->y_ptr[a-1])*(z_k.y));
        arr_z_k->y_ptr[a] = ((arr_z_k->x_ptr[a-1])*(z_k.y) + (arr_z_k->y_ptr[a-1])*(z_k.x));
        a += 1;
    }
}

// add [chi[n]] into h_chi_2d; assume that buff_chi_ptr->num_chi <= 32.
// also change the number of rows for each chi in it by the maximum conductor when it is full.
__host__ void twists_cuda_add_chi(ulong N_mod_f, ulong n_terms, double finv, twists_k_e *h_chi_2d, ulong *h_arr_f_chi, double *h_C_chi_real, \
    double *h_C_chi_imag, twists_twists *tw, twists_chi *chi, twists_cuda_buff_chi *buff_chi_ptr, twists_arr_complex *arr_z_k_ptr, ulong *buff_s_idx_per_chi)
{
    ulong i, j;
    ulong temp_row_0 = 0, temp_row_1;
//    ulong temp_row_2;
    ulong rest_num_chis_of_f = 0, rest_num_chis_in_buff, off_set;
    double temp_C_chi;

    // compute the number of chi's added
    rest_num_chis_of_f = (chi->num_chi_f) - (buff_chi_ptr->chi_start_idx);
    rest_num_chis_in_buff = NUM_THREADS_PER_BLOCK - buff_chi_ptr->num_chi;

    if(rest_num_chis_of_f <= rest_num_chis_in_buff)
    {   off_set = rest_num_chis_of_f;   }
    else
    {   off_set = rest_num_chis_in_buff;    }

    // update tw and h_C_chi_real and imag as well as h_arr_f_chi.
    tw->num_chi += off_set;

    temp_C_chi = ((double)tw->sign_E)/((double)chi->f_chi);
    
    for(i = 0; i < off_set; i++)
    {
        h_arr_f_chi[(buff_chi_ptr->num_chi) + i] = chi->f_chi;
        tw->f_chi_ptr[(buff_chi_ptr->tw_start_idx) + i] = chi->f_chi;
        tw->r_ptr[(buff_chi_ptr->tw_start_idx) + i] = chi->r[(buff_chi_ptr->chi_start_idx) + i];
        for(j = 2; j < chi->f_chi; j++)
        {
            temp_row_1 = (ulong)(chi->chi_2d_arr_ptr[(buff_chi_ptr->chi_start_idx) + i][j]);
            if(twists_gcd(temp_row_1, tw->k) == 1)
            {   break;  }
        }

        tw->c_ptr[(buff_chi_ptr->tw_start_idx) + i] = j;
        tw->chi_c_ptr[(buff_chi_ptr->tw_start_idx) + i] = chi->chi_2d_arr_ptr[(buff_chi_ptr->chi_start_idx) + i][tw->c_ptr[(buff_chi_ptr->tw_start_idx) + i]];
        tw->parity_ptr[(buff_chi_ptr->tw_start_idx) + i] = chi->chi_2d_arr_ptr[(buff_chi_ptr->chi_start_idx) + i][(chi->f_chi)-1];

        h_C_chi_real[(buff_chi_ptr->num_chi) + i] = temp_C_chi*(arr_z_k_ptr->x_ptr[chi->chi_2d_arr_ptr[(buff_chi_ptr->chi_start_idx) + i][N_mod_f]]);
        h_C_chi_imag[(buff_chi_ptr->num_chi) + i] = temp_C_chi*(arr_z_k_ptr->y_ptr[chi->chi_2d_arr_ptr[(buff_chi_ptr->chi_start_idx) + i][N_mod_f]]);
        tw->chi_N_ptr[(buff_chi_ptr->tw_start_idx) + i] = chi->chi_2d_arr_ptr[(buff_chi_ptr->chi_start_idx) + i][N_mod_f];
    }


    // add chi's into h_chi_2d
    temp_row_1 = (buff_chi_ptr->chi_start_idx);

    for(i = 0; i < (chi->f_chi); i++)
    {
        if(i >= (buff_chi_ptr->max_f))
        {
            for(j = 0; j < (buff_chi_ptr->num_chi); j++)
            {
                h_chi_2d[temp_row_0 + j] = h_chi_2d[buff_s_idx_per_chi[j] + j];
                buff_s_idx_per_chi[j] += NUM_THREADS_PER_BLOCK;
            }
        }
        for(j = 0; j < off_set; j++)
        {
            h_chi_2d[temp_row_0 + buff_chi_ptr->num_chi + j] = chi->chi_2d_arr_ptr[temp_row_1 + j][i];
        }
        temp_row_0 += NUM_THREADS_PER_BLOCK;
    }

    // update buff_chi_ptr and buff_f_mod_max_f.
    if(rest_num_chis_of_f < rest_num_chis_in_buff)
    {
        buff_chi_ptr->chi_start_idx = (chi->num_chi_f);
        buff_chi_ptr->num_chi += off_set;
    }
    else
    {   
        buff_chi_ptr->chi_start_idx += off_set;
        buff_chi_ptr->num_chi = NUM_THREADS_PER_BLOCK;
    }
    buff_chi_ptr->max_f = chi->f_chi;
    buff_chi_ptr->num_terms = n_terms;
    buff_chi_ptr->tw_start_idx = tw->num_chi;
}

// compute L(E, 1, chi), its algebraic value and the Gauss sum of chi.
// note that k is assume to be relatively small as 2^(32) – 1.
__global__ void twists_cuda_l_cal(ulong N, ulong k, ulong max_f, ulong num_terms, ulong *d_f_ptr, double *d_gs_real_ptr, double *d_gs_imag_ptr, \
    double *d_l_val_real_ptr, double *d_l_val_imag_ptr, const double *C_chi_real_ptr, const double *C_chi_imag_ptr, const twists_k_e *d_chi_2d, \
    double *z_k_c_chi_real_2d_ptr, double *z_k_c_chi_imag_2d_ptr, int *d_a_an)
{
    int tid = threadIdx.x;
    ulong f, temp_tid, temp_r, temp_i, val_chi, n = 0;
    double temp_f, q_r, q_i, t_q_r, t_q_i;
    extern __shared__ double arr_gs_r_i[];

    f = d_f_ptr[tid];
    temp_r = 2*(k+1);
    temp_f = ((double)2)*twists_pi/((double)f);
    q_r = cos(temp_f);
    q_i = sin(temp_f);
    t_q_r = q_r;
    t_q_i = q_i;

    __syncthreads();

    // allocate a double array of length 2*NUM_THREADS_PER_BLOCK*(k+1): the first half are for reals and the rests for imaginaries of Gauss sums.
    temp_tid = tid;
    while(n < temp_r)
    {
        arr_gs_r_i[temp_tid] = 0;
        temp_tid += NUM_THREADS_PER_BLOCK;
        n += 1;
    }
    // compute the Gauss sum
    n = 1;
    temp_tid = tid + NUM_THREADS_PER_BLOCK;
    temp_i = (k+1)*NUM_THREADS_PER_BLOCK;
    while(n < f)
    {
        val_chi = (ulong)(d_chi_2d[temp_tid]);
        temp_r = tid + val_chi*NUM_THREADS_PER_BLOCK;
        arr_gs_r_i[temp_r] += t_q_r;
        arr_gs_r_i[temp_r + temp_i] += t_q_i;
        temp_f = t_q_r*q_r - t_q_i*q_i;
        t_q_i = t_q_r*q_i + t_q_i*q_r;
        t_q_r = temp_f;
        n += 1;
        temp_tid += NUM_THREADS_PER_BLOCK;
    }
    t_q_r = 0;
    t_q_i = 0;
    n = 1;
    temp_tid = tid + NUM_THREADS_PER_BLOCK;
    while(n <= k)
    {
        t_q_r += (arr_gs_r_i[temp_tid]*arr_z_k_real[n] - arr_gs_r_i[temp_tid + temp_i]*arr_z_k_imag[n]);
        t_q_i += (arr_gs_r_i[temp_tid]*arr_z_k_imag[n] + arr_gs_r_i[temp_tid + temp_i]*arr_z_k_real[n]);
        n += 1;
        temp_tid += NUM_THREADS_PER_BLOCK;
    }
    d_gs_real_ptr[tid] = t_q_r;
    d_gs_imag_ptr[tid] = t_q_i;

    // compute L(E,1,chi)
    n = 0;
    temp_r = k+1;
    temp_tid = tid;
    while(n < temp_r)
    {
        arr_gs_r_i[temp_tid] = 0;
        temp_tid += NUM_THREADS_PER_BLOCK;
        n += 1;
    }
    q_r = (t_q_r*t_q_r) - (t_q_i*t_q_i);
    q_i = t_q_r*t_q_i;
    q_i += q_i;
    t_q_r = C_chi_real_ptr[tid]*q_r - C_chi_imag_ptr[tid]*q_i;
    t_q_i = C_chi_real_ptr[tid]*q_i + C_chi_imag_ptr[tid]*q_r;    

    z_k_c_chi_real_2d_ptr[tid] = (double)0;
    z_k_c_chi_imag_2d_ptr[tid] = (double)0;
    n = 1;
    temp_tid = tid + NUM_THREADS_PER_BLOCK;
    while(n <= k)
    {
        z_k_c_chi_real_2d_ptr[temp_tid] = ((double)1 + t_q_r)*arr_z_k_real[n] + t_q_i*arr_z_k_imag[n];
        z_k_c_chi_imag_2d_ptr[temp_tid] = ((double)1 - t_q_r)*arr_z_k_imag[n] + t_q_i*arr_z_k_real[n];
        temp_tid += NUM_THREADS_PER_BLOCK;
        n += 1;     
    }
    q_i = (((double)1)/((double)f));
    q_r = ((double)1)/exp(((double)2)*((double)twists_pi)*rsqrt((double)N)*q_i);
    if(f != max_f)
    {
        temp_i = ((ulong)(max_f - f*floor(((double)max_f)*q_i)))*NUM_THREADS_PER_BLOCK;
    }
    else
    {
        temp_i = 0;
    }

    t_q_r = q_r;
    temp_tid = tid + NUM_THREADS_PER_BLOCK;
    f = tid + (max_f-1)*NUM_THREADS_PER_BLOCK;
    for(n = 1; n <= num_terms; n++)
    {
        if(temp_tid > f)
        {   temp_tid = tid + temp_i;  }
        val_chi = (ulong)(d_chi_2d[temp_tid]);
        temp_r = tid + val_chi*NUM_THREADS_PER_BLOCK;
        arr_gs_r_i[temp_r] += ((double)(d_a_an[n]))*t_q_r/((double)n);
        t_q_r *= q_r;
        temp_tid += NUM_THREADS_PER_BLOCK;
    }
    t_q_r = 0;
    t_q_i = 0;
    n = 1;
    temp_tid = tid + NUM_THREADS_PER_BLOCK;
    while(n <= k)
    {
        t_q_r += arr_gs_r_i[temp_tid]*z_k_c_chi_real_2d_ptr[temp_tid];
        t_q_i += arr_gs_r_i[temp_tid]*z_k_c_chi_imag_2d_ptr[temp_tid];
        temp_tid += NUM_THREADS_PER_BLOCK;
        n += 1;
    }
    d_l_val_real_ptr[tid] = t_q_r;
    d_l_val_imag_ptr[tid] = t_q_i;  
}

int main(int argc, char *argv[])
{
    timeit_t t0;

    FILE *file;
    //file path with its name: Note that its length should be less than or equal to FILE_PATH_LENGTH.
    char file_path[FILE_PATH_LENGTH] = "";

    twists_twists tw;
    ulong x_start, x_end, f, N_mod_f, n_terms = 0, temp_n_terms;

    int i, j, m_k = 0;
    double finv, kinv = 1;
    twists_f_chi_k *t_f_k_ptr = NULL;
    twists_f *k_ptr = NULL;
    int *k_e_ptr = NULL;
    twists_chi chi;

    twists_int_array h_a_an;
    int *d_a_an = NULL;

    twists_complex p_kth_root_1;
    twists_arr_complex arr_z_k;

    twists_k_e *h_chi_2d = NULL;
    ulong *h_arr_f_chi = NULL;
    double *h_C_chi_real = NULL;
    double *h_C_chi_imag = NULL;

    twists_k_e *d_chi_2d[NUM_STREAMS];
    ulong *d_f_chi_2d[NUM_STREAMS];
    double *d_C_chi_real_2d[NUM_STREAMS];
    double *d_C_chi_imag_2d[NUM_STREAMS];
    double *d_z_k_c_chi_real_2d[NUM_STREAMS];
    double *d_z_k_c_chi_imag_2d[NUM_STREAMS];

    double *d_gs_real_2d[NUM_STREAMS];
    double *d_gs_imag_2d[NUM_STREAMS];
    double *d_lval_real_2d[NUM_STREAMS];
    double *d_lval_imag_2d[NUM_STREAMS];

    size_t size_tw_k_e;
    size_t size_ulong;
    size_t size_double;

    twists_k_e *temp_tw_tke = NULL;
    ulong *temp_tw_ul = NULL;
    double *temp_tw_dbl = NULL;
    size_t tw_size_ulong_x;
    size_t tw_size_double_x;
    size_t tw_size_twists_k_e_x;
    ulong temp_x;

    size_t shared_mem_size;

    cudaStream_t streams[NUM_STREAMS];
    
    twists_cuda_buff_chi buff_twist;
    ulong previous_buff_start_idx;
    ulong buff_s_idx_per_chi[NUM_THREADS_PER_BLOCK] = {0};

    int max_stream_idx = 0;
    ulong test_intv;
    ulong test_x;

    timeit_start(t0);

    // some cuda settings
    checkCudaErrors( cudaSetDeviceFlags(cudaDeviceMapHost) );
    checkCudaErrors( cudaFuncSetCacheConfig(twists_cuda_l_cal, cudaFuncCachePreferShared) );
    checkCudaErrors( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );

    if(argc < TWISTS_NUM_ARGS)
    {
		flint_printf("The number of args should be %d.\n", (TWISTS_NUM_ARGS-1));
		return 0;
    }	
    tw.N = atol(argv[1]);
    tw.sign_E = atol(argv[2]);
    tw.k = atol(argv[3]);
    x_start = atol(argv[4]);
    x_end = atol(argv[5]);
    temp_x = x_end;
    test_intv = (ulong)(floor(double(x_end - x_start)/((double)100)));

    test_x = test_intv;

    // load a_n's for E and copy a_an into device and free it from host.
    if(twists_load_an(&h_a_an, argv[6]) == EXIT_FAILURE)
    {
        fprintf(stderr, "Error: load failed for %s\n", argv[5]);
        exit(EXIT_FAILURE);
    }
    checkCudaErrors( cudaMalloc((void **)&d_a_an, (h_a_an.len)*sizeof(int)) );
    checkCudaErrors( cudaMemcpy(d_a_an, h_a_an.a_ptr, (h_a_an.len)*sizeof(int), cudaMemcpyHostToDevice) );

    flint_printf("The length of a_n is %wu\n", h_a_an.len);

    // free a_an.
    twists_int_arr_free(h_a_an.a_ptr);
    h_a_an.a_ptr = NULL;
    h_a_an.len = 0;

    flint_printf("a_n are loaded into device!\n");

    strcpy(file_path, argv[7]);

    // initialize tw
    tw_size_ulong_x = x_end*sizeof(ulong);
    tw_size_double_x = x_end*sizeof(double);
    tw_size_twists_k_e_x = x_end*sizeof(twists_k_e);

    shared_mem_size = 2*NUM_THREADS_PER_BLOCK*((tw.k)+1)*sizeof(double);

    checkCudaErrors( cudaMallocHost((void **)&(tw.f_chi_ptr), tw_size_ulong_x) );
    checkCudaErrors( cudaMallocHost((void **)&(tw.r_ptr), tw_size_ulong_x) );
    checkCudaErrors( cudaMallocHost((void **)&(tw.l_val_real_ptr), tw_size_double_x) );
    checkCudaErrors( cudaMallocHost((void **)&(tw.l_val_imag_ptr), tw_size_double_x) );
    checkCudaErrors( cudaMallocHost((void **)&(tw.g_sum_real_ptr), tw_size_double_x) );
    checkCudaErrors( cudaMallocHost((void **)&(tw.g_sum_imag_ptr), tw_size_double_x) );
    checkCudaErrors( cudaMallocHost((void **)&(tw.chi_N_ptr), tw_size_twists_k_e_x) );
    checkCudaErrors( cudaMallocHost((void **)&(tw.c_ptr), tw_size_twists_k_e_x) );
    checkCudaErrors( cudaMallocHost((void **)&(tw.chi_c_ptr), tw_size_twists_k_e_x) );
    checkCudaErrors( cudaMallocHost((void **)&(tw.parity_ptr), tw_size_twists_k_e_x) );
    checkCudaErrors( cudaMallocHost((void **)&(tw.num_terms_ptr), tw_size_ulong_x) );

    size_tw_k_e = NUM_THREADS_PER_BLOCK*sizeof(twists_k_e);
    size_ulong = NUM_THREADS_PER_BLOCK*sizeof(ulong);
    size_double = NUM_THREADS_PER_BLOCK*sizeof(double);

    // factorize k and compute 1/k
    k_ptr = twists_factor(tw.k);
    m_k = (1 << k_ptr->len)-1;
    k_e_ptr = (int *)flint_calloc(k_ptr->len, sizeof(int));
    for(j = 0; j < k_ptr->len; j++)
        k_e_ptr[j] = k_ptr->array_f_i[j].a_i;
    for(j = 0; j < k_ptr->len; j++)
        kinv *= k_ptr->array_f_i[j].f_a_inv[k_ptr->array_f_i[j].a_i-1];

    // allocate [0, zeta_k, zeta_k^2, zeta_k^3, ..., zeta_k^k]
    twists_cuda_primitive_root_of_unity(tw.k, &p_kth_root_1);
    twists_cuda_array_z_k(tw.k, &arr_z_k);
    checkCudaErrors( cudaMemcpyToSymbol(arr_z_k_real, arr_z_k.x_ptr, (arr_z_k.len)*sizeof(double)) );
    checkCudaErrors( cudaMemcpyToSymbol(arr_z_k_imag, arr_z_k.y_ptr, (arr_z_k.len)*sizeof(double)) );

    // allocate all kinds of buffers in host and device for chi[n]'s of dimension x_end by NUM_THREADS_PER_BLOCK.
    checkCudaErrors( cudaMallocHost((void **)&h_chi_2d, size_tw_k_e*x_end) );
    checkCudaErrors( cudaMallocHost((void **)&h_arr_f_chi, size_ulong) );
    checkCudaErrors( cudaMallocHost((void **)&h_C_chi_real, size_double) );
    checkCudaErrors( cudaMallocHost((void **)&h_C_chi_imag, size_double) );

    for(j = 0; j < NUM_STREAMS; j++)
    {
        checkCudaErrors( cudaMalloc((void **)&(d_chi_2d[j]), size_tw_k_e*x_end) );
        checkCudaErrors( cudaMalloc((void **)&(d_f_chi_2d[j]), size_ulong) );
        checkCudaErrors( cudaMalloc((void **)&(d_C_chi_real_2d[j]), size_double) );
        checkCudaErrors( cudaMalloc((void **)&(d_C_chi_imag_2d[j]), size_double) );
        checkCudaErrors( cudaMalloc((void **)&(d_z_k_c_chi_real_2d[j]), size_double*((tw.k)+1)) );
        checkCudaErrors( cudaMalloc((void **)&(d_z_k_c_chi_imag_2d[j]), size_double*((tw.k)+1)) );

        checkCudaErrors( cudaMalloc((void **)&(d_lval_real_2d[j]), size_double) );
        checkCudaErrors( cudaMalloc((void **)&(d_lval_imag_2d[j]), size_double) );
        checkCudaErrors( cudaMalloc((void **)&(d_gs_real_2d[j]), size_double) );
        checkCudaErrors( cudaMalloc((void **)&(d_gs_imag_2d[j]), size_double) );
    }

    for(j = 0; j < NUM_STREAMS; j++)
    { checkCudaErrors( cudaStreamCreate(&streams[j]) );}

    flint_printf("Twists for N = %wu, k = %wu, kinv = %.17g, zeta_k = %.17g + %.17g*I\n", tw.N, tw.k, kinv, p_kth_root_1.x, p_kth_root_1.y);

    // loop over f_chi from x_start to x_end
    for(f = x_start; f <= x_end; f++)
    {
        if(f > test_x)
        {
            flint_printf("Done for f < %wu. tw.num_chi: %wu\n", test_x, tw.num_chi);
            test_x += test_intv;
        }
        t_f_k_ptr = twists_init_f_chi_k(k_ptr->len);            
        if(twists_is_twist(f, tw.k, tw.N, m_k, t_f_k_ptr, k_ptr))
        {
            if(twists_dirichlet_character(f, tw.k, kinv, k_e_ptr, m_k, t_f_k_ptr, k_ptr, &chi) == EXIT_SUCCESS)
            {                    
                if(temp_x < (tw.num_chi + chi.num_chi_f))
                {
                    flint_printf("Buffer for tw is doubled at %wu\n", chi.f_chi);
                    tw_size_twists_k_e_x *= 2;
                    tw_size_ulong_x *= 2;
                    tw_size_double_x *= 2;
                    checkCudaErrors( cudaMallocHost((void **)&(temp_tw_ul), tw_size_ulong_x) );
                    checkCudaErrors( cudaMallocHost((void **)&(temp_tw_dbl), tw_size_double_x) );
                    checkCudaErrors( cudaMallocHost((void **)&(temp_tw_tke), tw_size_twists_k_e_x) );
                    
                    cudaDeviceSynchronize();

                    checkCudaErrors( cudaMemcpy(temp_tw_ul, tw.f_chi_ptr, (tw.num_chi)*sizeof(ulong), cudaMemcpyHostToHost) );
                    checkCudaErrors( cudaFreeHost(tw.f_chi_ptr) );
                    tw.f_chi_ptr = temp_tw_ul;
                    checkCudaErrors( cudaMallocHost((void **)&(temp_tw_ul), tw_size_ulong_x) );
                    checkCudaErrors( cudaMemcpy(temp_tw_ul, tw.r_ptr, (tw.num_chi)*sizeof(ulong), cudaMemcpyHostToHost) );
                    checkCudaErrors( cudaFreeHost(tw.r_ptr) );
                    tw.r_ptr = temp_tw_ul;
                    checkCudaErrors( cudaMallocHost((void **)&(temp_tw_dbl), tw_size_double_x) );
                    checkCudaErrors( cudaMemcpy(temp_tw_dbl, tw.l_val_real_ptr, (tw.num_chi)*sizeof(double), cudaMemcpyHostToHost) );
                    checkCudaErrors( cudaFreeHost(tw.l_val_real_ptr) );
                    tw.l_val_real_ptr = temp_tw_dbl;
                    checkCudaErrors( cudaMallocHost((void **)&(temp_tw_dbl), tw_size_double_x) );
                    checkCudaErrors( cudaMemcpy(temp_tw_dbl, tw.l_val_imag_ptr, (tw.num_chi)*sizeof(double), cudaMemcpyHostToHost) );
                    checkCudaErrors( cudaFreeHost(tw.l_val_imag_ptr) );
                    tw.l_val_imag_ptr = temp_tw_dbl;
                    checkCudaErrors( cudaMallocHost((void **)&(temp_tw_dbl), tw_size_double_x) );
                    checkCudaErrors( cudaMemcpy(temp_tw_dbl, tw.g_sum_real_ptr, (tw.num_chi)*sizeof(double), cudaMemcpyHostToHost) );
                    checkCudaErrors( cudaFreeHost(tw.g_sum_real_ptr) );
                    tw.g_sum_real_ptr = temp_tw_dbl;
                    checkCudaErrors( cudaMallocHost((void **)&(temp_tw_dbl), tw_size_double_x) );
                    checkCudaErrors( cudaMemcpy(temp_tw_dbl, tw.g_sum_imag_ptr, (tw.num_chi)*sizeof(double), cudaMemcpyHostToHost) );
                    checkCudaErrors( cudaFreeHost(tw.g_sum_imag_ptr) );
                    tw.g_sum_imag_ptr = temp_tw_dbl;
                    checkCudaErrors( cudaMallocHost((void **)&(temp_tw_tke), tw_size_twists_k_e_x) );
                    checkCudaErrors( cudaMemcpy(temp_tw_tke, tw.chi_N_ptr, (tw.num_chi)*sizeof(twists_k_e), cudaMemcpyHostToHost) );
                    checkCudaErrors( cudaFreeHost(tw.chi_N_ptr) );
                    tw.chi_N_ptr = temp_tw_tke;
                    checkCudaErrors( cudaMallocHost((void **)&(temp_tw_tke), tw_size_twists_k_e_x) );
                    checkCudaErrors( cudaMemcpy(temp_tw_tke, tw.c_ptr, (tw.num_chi)*sizeof(twists_k_e), cudaMemcpyHostToHost) );
                    checkCudaErrors( cudaFreeHost(tw.c_ptr) );
                    tw.c_ptr = temp_tw_tke;
                    checkCudaErrors( cudaMallocHost((void **)&(temp_tw_tke), tw_size_twists_k_e_x) );
                    checkCudaErrors( cudaMemcpy(temp_tw_tke, tw.chi_c_ptr, (tw.num_chi)*sizeof(twists_k_e), cudaMemcpyHostToHost) );
                    checkCudaErrors( cudaFreeHost(tw.chi_c_ptr) );
                    tw.chi_c_ptr = temp_tw_tke;
                    checkCudaErrors( cudaMallocHost((void **)&(temp_tw_tke), tw_size_twists_k_e_x) );
                    checkCudaErrors( cudaMemcpy(temp_tw_tke, tw.parity_ptr, (tw.num_chi)*sizeof(twists_k_e), cudaMemcpyHostToHost) );
                    checkCudaErrors( cudaFreeHost(tw.parity_ptr) );
                    tw.parity_ptr = temp_tw_tke;
                    checkCudaErrors( cudaMallocHost((void **)&(temp_tw_ul), tw_size_ulong_x) );
                    checkCudaErrors( cudaMemcpy(temp_tw_ul, tw.num_terms_ptr, (tw.num_chi)*sizeof(ulong), cudaMemcpyHostToHost) );
                    checkCudaErrors( cudaFreeHost(tw.num_terms_ptr) );
                    tw.num_terms_ptr = temp_tw_ul;

                    temp_x *= 2;
                }
                n_terms = twists_num_terms(chi.f_chi, tw.N);
                finv = n_precompute_inverse(chi.f_chi);
                N_mod_f = n_mod_precomp(tw.N, chi.f_chi, finv);

                while(buff_twist.chi_start_idx < chi.num_chi_f)
                {
                    // add chi into buffer
                    twists_cuda_add_chi(N_mod_f, n_terms, finv, h_chi_2d, h_arr_f_chi, h_C_chi_real, h_C_chi_imag, &tw, &chi, &buff_twist, &arr_z_k, buff_s_idx_per_chi);

                    if(buff_twist.num_chi == NUM_THREADS_PER_BLOCK)
                    {
                        //update num_terms_ptr in tw
                        temp_n_terms = tw.num_chi - NUM_THREADS_PER_BLOCK;
                        for(j = 0; j < NUM_THREADS_PER_BLOCK; j++)
                        {
                            tw.num_terms_ptr[temp_n_terms + j] = buff_twist.num_terms;
                        }

                        // loop until a stream is available.
                        j = 0;
                        while(cudaStreamQuery(streams[j]) != cudaSuccess)
                        {                            
                            if(j == (NUM_STREAMS-1))
                            {   j = 0;  }
                            else
                            {   j += 1; }
                        }
                        if(j > max_stream_idx)
                        {
                            max_stream_idx = j;
                        }

                        // initialize memories in device
                        checkCudaErrors( cudaMemsetAsync(d_lval_real_2d[j], 0, size_double, streams[j]) );
                        checkCudaErrors( cudaMemsetAsync(d_lval_imag_2d[j], 0, size_double, streams[j]) );
                        checkCudaErrors( cudaMemsetAsync(d_gs_real_2d[j], 0, size_double, streams[j]) );
                        checkCudaErrors( cudaMemsetAsync(d_gs_imag_2d[j], 0, size_double, streams[j]) );
                        // memcpyAsync into device and launch kernel
                        checkCudaErrors( cudaMemcpyAsync(d_chi_2d[j], h_chi_2d, (buff_twist.max_f)*size_tw_k_e, cudaMemcpyHostToDevice, streams[j]) );
                        checkCudaErrors( cudaMemcpyAsync(d_f_chi_2d[j], h_arr_f_chi, size_ulong, cudaMemcpyHostToDevice, streams[j]) );
                        checkCudaErrors( cudaMemcpyAsync(d_C_chi_real_2d[j], h_C_chi_real, size_double, cudaMemcpyHostToDevice, streams[j]) );
                        checkCudaErrors( cudaMemcpyAsync(d_C_chi_imag_2d[j], h_C_chi_imag, size_double, cudaMemcpyHostToDevice, streams[j]) );
                        
                        checkCudaErrors( cudaStreamSynchronize(streams[j]) );

                        // launch a kernel into streams[j]
                        twists_cuda_l_cal<<<1, buff_twist.num_chi, shared_mem_size, streams[j]>>>(tw.N, tw.k, buff_twist.max_f, buff_twist.num_terms, \
                            d_f_chi_2d[j], d_gs_real_2d[j], d_gs_imag_2d[j], d_lval_real_2d[j], d_lval_imag_2d[j], d_C_chi_real_2d[j], d_C_chi_imag_2d[j], \
                            d_chi_2d[j], d_z_k_c_chi_real_2d[j], d_z_k_c_chi_imag_2d[j], d_a_an);

                        previous_buff_start_idx = buff_twist.tw_start_idx - NUM_THREADS_PER_BLOCK;

                        // collect the values of L(E,1,chi) and the Gauss sums of chi's
                        checkCudaErrors( cudaMemcpyAsync(&(tw.l_val_real_ptr)[previous_buff_start_idx], d_lval_real_2d[j], size_double, cudaMemcpyDeviceToHost, streams[j]) );
                        checkCudaErrors( cudaMemcpyAsync(&(tw.l_val_imag_ptr)[previous_buff_start_idx], d_lval_imag_2d[j], size_double, cudaMemcpyDeviceToHost, streams[j]) );
                        checkCudaErrors( cudaMemcpyAsync(&(tw.g_sum_real_ptr)[previous_buff_start_idx], d_gs_real_2d[j], size_double, cudaMemcpyDeviceToHost, streams[j]) );
                        checkCudaErrors( cudaMemcpyAsync(&(tw.g_sum_imag_ptr)[previous_buff_start_idx], d_gs_imag_2d[j], size_double, cudaMemcpyDeviceToHost, streams[j]) );

                        // reset buff_s_idx_per_chi and buff_twist
                        for(i = 0; i < buff_twist.num_chi; i++)
                        {   buff_s_idx_per_chi[i] = 0; }
                        buff_twist.num_chi = 0;
                    }
                }
                // collect chi's into d_chi_2d until NUM_THREADS_PER_BLOCK number of chi's are stored.
                twists_chi_free(chi);
                buff_twist.chi_start_idx = 0;
            }
        }

        if(t_f_k_ptr->len_f != 0 || t_f_k_ptr != NULL)
        {   twists_f_chi_k_free(t_f_k_ptr); }
    }
    // loop until a stream is available.
    //update num_terms_ptr in tw
    temp_n_terms = tw.num_chi - buff_twist.num_chi;
    for(j = 0; j < buff_twist.num_chi; j++)
    {
        tw.num_terms_ptr[temp_n_terms + j] = buff_twist.num_terms;
    }
    j = 0;
    while(cudaStreamQuery(streams[j]) != cudaSuccess)
    {                            
        if(j == (NUM_STREAMS-1))
        {   j = 0;  }
        else
        {   j += 1; }
    }

    // initialize memories in device
    checkCudaErrors( cudaMemsetAsync(d_lval_real_2d[j], 0, size_double, streams[j]) );
    checkCudaErrors( cudaMemsetAsync(d_lval_imag_2d[j], 0, size_double, streams[j]) );
    checkCudaErrors( cudaMemsetAsync(d_gs_real_2d[j], 0, size_double, streams[j]) );
    checkCudaErrors( cudaMemsetAsync(d_gs_imag_2d[j], 0, size_double, streams[j]) );
    // memcpyAsync into device and launch kernel
    checkCudaErrors( cudaMemcpyAsync(d_chi_2d[j], h_chi_2d, (buff_twist.max_f)*size_tw_k_e, cudaMemcpyHostToDevice, streams[j]) );
    checkCudaErrors( cudaMemcpyAsync(d_f_chi_2d[j], h_arr_f_chi, size_ulong, cudaMemcpyHostToDevice, streams[j]) );
    checkCudaErrors( cudaMemcpyAsync(d_C_chi_real_2d[j], h_C_chi_real, size_double, cudaMemcpyHostToDevice, streams[j]) );
    checkCudaErrors( cudaMemcpyAsync(d_C_chi_imag_2d[j], h_C_chi_imag, size_double, cudaMemcpyHostToDevice, streams[j]) );
                        
    checkCudaErrors( cudaStreamSynchronize(streams[j]) );

    // launch a kernel into streams[j]
    twists_cuda_l_cal<<<1, buff_twist.num_chi, shared_mem_size, streams[j]>>>(tw.N, tw.k, buff_twist.max_f, buff_twist.num_terms, \
        d_f_chi_2d[j], d_gs_real_2d[j], d_gs_imag_2d[j], d_lval_real_2d[j], d_lval_imag_2d[j], d_C_chi_real_2d[j], d_C_chi_imag_2d[j], \
        d_chi_2d[j], d_z_k_c_chi_real_2d[j], d_z_k_c_chi_imag_2d[j], d_a_an);

    // collect the values of L(E,1,chi) and the Gauss sums of chi's
    previous_buff_start_idx = buff_twist.tw_start_idx - buff_twist.num_chi;
    checkCudaErrors( cudaMemcpyAsync(&(tw.l_val_real_ptr)[previous_buff_start_idx], d_lval_real_2d[j], size_double, cudaMemcpyDeviceToHost, streams[j]) );
    checkCudaErrors( cudaMemcpyAsync(&(tw.l_val_imag_ptr)[previous_buff_start_idx], d_lval_imag_2d[j], size_double, cudaMemcpyDeviceToHost, streams[j]) );
    checkCudaErrors( cudaMemcpyAsync(&(tw.g_sum_real_ptr)[previous_buff_start_idx], d_gs_real_2d[j], size_double, cudaMemcpyDeviceToHost, streams[j]) );
    checkCudaErrors( cudaMemcpyAsync(&(tw.g_sum_imag_ptr)[previous_buff_start_idx], d_gs_imag_2d[j], size_double, cudaMemcpyDeviceToHost, streams[j]) );
    checkCudaErrors( cudaStreamSynchronize(streams[j]) );

    // free host memories
    cudaFreeHost(h_chi_2d);
    cudaFreeHost(h_arr_f_chi);
    cudaFreeHost(h_C_chi_real);
    cudaFreeHost(h_C_chi_imag);

    // free k_ptr and k_e_ptr
    twists_factor_free(k_ptr);
    twists_int_arr_free(k_e_ptr);

    // free a_n's in device
    cudaFree(d_a_an);

    for(j = 0; j < NUM_STREAMS; j++)
    {
        checkCudaErrors( cudaStreamSynchronize(streams[j]) );
        checkCudaErrors( cudaStreamDestroy(streams[j]) );
    }

    flint_free(arr_z_k.x_ptr);
    flint_free(arr_z_k.y_ptr);
    arr_z_k.x_ptr = NULL;
    arr_z_k.y_ptr = NULL;
    arr_z_k.len = 0;

    // save tw into a file
    file = fopen(file_path, "w");
    tw.num_chi -= 1;
    for(j = 0; j < tw.num_chi; j++)
    {
        flint_fprintf(file, "%wu, %wu, %wu, %wu, %.17g, %.17g, %.17g, %.17g, %d, %d, %d, %d, %wu\n", tw.N, tw.k, tw.f_chi_ptr[j], \
            tw.r_ptr[j], tw.l_val_real_ptr[j], tw.l_val_imag_ptr[j], tw.g_sum_real_ptr[j], tw.g_sum_imag_ptr[j], tw.chi_N_ptr[j], \
            tw.c_ptr[j], tw.chi_c_ptr[j], tw.parity_ptr[j], tw.num_terms_ptr[j]);
    }
    flint_fprintf(file, "%wu, %wu, %wu, %wu, %.17g, %.17g, %.17g, %.17g, %d, %d, %d, %d, %wu\n", tw.N, tw.k, tw.f_chi_ptr[j], \
            tw.r_ptr[j], tw.l_val_real_ptr[j], tw.l_val_imag_ptr[j], tw.g_sum_real_ptr[j], tw.g_sum_imag_ptr[j], tw.chi_N_ptr[j], \
            tw.c_ptr[j], tw.chi_c_ptr[j], tw.parity_ptr[j], tw.num_terms_ptr[j]);
    fclose(file);

    // free memories allocated pitched in device.
    for(j = 0; j < NUM_STREAMS; j++)
    {
        cudaFree(d_chi_2d[j]);
        cudaFree(d_f_chi_2d[j]);
        cudaFree(d_gs_real_2d[j]);
        cudaFree(d_gs_imag_2d[j]);
        cudaFree(d_lval_real_2d[j]);
        cudaFree(d_lval_imag_2d[j]);
        cudaFree(d_C_chi_real_2d[j]);
        cudaFree(d_C_chi_imag_2d[j]);
        cudaFree(d_z_k_c_chi_real_2d[j]);
        cudaFree(d_z_k_c_chi_imag_2d[j]);
   }

   // free tw
    cudaFreeHost(tw.f_chi_ptr);
    cudaFreeHost(tw.r_ptr);
    cudaFreeHost(tw.l_val_real_ptr);
    cudaFreeHost(tw.l_val_imag_ptr);
    cudaFreeHost(tw.g_sum_real_ptr);
    cudaFreeHost(tw.g_sum_imag_ptr);
    cudaFreeHost(tw.chi_N_ptr);
    cudaFreeHost(tw.c_ptr);
    cudaFreeHost(tw.chi_c_ptr);
    cudaFreeHost(tw.parity_ptr);
    cudaFreeHost(tw.num_terms_ptr);

    timeit_stop(t0);

    flint_printf("The computations are all done. Max stream idx = %d\n", max_stream_idx);
    flint_printf("cpu = %wd ms wall = %wd ms\n", t0->cpu, t0->wall);

	return cudaSuccess;
}