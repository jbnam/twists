/*
   Copyright (C) 2022 Jungbae Nam <jungbae.nam@gmail.com>

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 2 of the License, or
 (at your option) any later version.
                  https://www.gnu.org/licenses/
*/

/*

*/

#ifndef _TWISTS_CLVE_
#define _TWISTS_CLVE_

#ifdef ULONG_EXTRAS_INLINES_C
#define ULONG_EXTRAS_INLINE FLINT_DLL
#else
#define ULONG_EXTRAS_INLINE static __inline__
#endif

#undef ulong
#define ulong ulongxx
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <cuComplex.h>
#undef ulong
#define ulong mp_limb_t

#include "gmp.h"
#include "flint/flint.h"
#include "flint/ulong_extras.h"
#include "flint/profiler.h"

#include "twists_dirichlet_character.h"

#ifdef __cplusplus
 extern "C" {
#endif

#define NUM_DATA 1024
#define TWISTS_NUM_ARGS 8
#define FILE_PATH_LENGTH 512
// these constants depend on the cuda hardware and compute capability (gtx 1080 Ti and cc 6.1 for the below).
#define NUM_STREAMS 32
#define NUM_THREADS_PER_BLOCK 32

typedef cuDoubleComplex twists_complex;

typedef struct 
{
   ulong len = 0;
   double *x_ptr = NULL;
   double *y_ptr = NULL;
}
twists_arr_complex;

typedef struct 
{
   ulong chi_start_idx = 0; // the index from which chi should start to be added
   ulong num_chi = 0; // the total number of [chi[n]]'s added in the buffer (up to NUM_NUM_THREADS_PER_BLOCK)
   ulong max_f = 0; // the maximum conductor of chi's added in the buffer
   ulong num_terms = 0; // the maximum number of terms to be computed for L(E,1,chi)
   ulong tw_start_idx = 0; // starting index in twists_twists
}
twists_cuda_buff_chi;

typedef struct
{
   ulong N; // the conductor of E
   ulong k; // the order of chi
   ulong num_chi = 0; // the number of primitive characters
   ulong *f_chi_ptr = NULL; // array of conductors of character chi
   ulong *r_ptr = NULL; // array of labels of chi

   double *l_val_real_ptr = NULL; // array of real parts of central L-values of L(E, 1, chi)
   double *l_val_imag_ptr = NULL; // array of imaginary parts of central L-values of L(E, 1, chi)
   double *g_sum_real_ptr = NULL; // array of real parts of Gauss sums of chi's.
   double *g_sum_imag_ptr = NULL; // array of imaginary parts of Gauss sums of chi's.
   twists_k_e *chi_N_ptr = NULL; // array of chi(N).
   twists_k_e *c_ptr = NULL; // array of the least positive integer with chi(c) not equal to 1.
   twists_k_e *chi_c_ptr = NULL; // array of chi(c).
   twists_k_e *parity_ptr = NULL; // array of chi(-1)
   ulong *num_terms_ptr = NULL; // array of the number of terms computed for desired errors
   slong sign_E = 1; // sign of functional equation of L(E,s)
}
twists_twists;

__host__ void twists_complex_set(twists_complex *a, twists_complex b);

__host__ void twists_cuda_primitive_root_of_unity(ulong k, twists_complex *p_root_1_ptr);

__host__ void twists_cuda_array_z_k(ulong k, twists_arr_complex *arr_z_k);

__host__ void twists_cuda_add_chi(ulong N_mod_f, ulong nterms, double finv, twists_k_e *h_chi_2d, ulong *h_arr_f_chi, double *h_C_chi_real, \
    double *h_C_chi_imag, twists_twists *tw, twists_chi *chi, twists_cuda_buff_chi *buff_chi_ptr, twists_arr_complex *arr_z_k_ptr, ulong *buff_s_idx_per_chi);

//__host__ void twists_cuda_reset_buff_chi(twists_cuda_buff_chi *buff_chi_ptr, ulong chi_st_idx, ulong mx_f, ulong n_terms, ulong tw_st_idx);

__global__ void twists_cuda_l_cal(ulong N, ulong k, ulong max_f, ulong num_terms, ulong *d_f_ptr, double *d_gs_real_ptr, double *d_gs_imag_ptr, \
    double *d_l_val_real_ptr, double *d_l_val_imag_ptr, const double *C_chi_real_ptr, const double *C_chi_imag_ptr, const twists_k_e *d_chi_2d, \
    double *z_k_c_chi_real_2d_ptr, double *z_k_c_chi_imag_2d_ptr, int *d_a_an);

#ifdef __cplusplus
}
#endif

#endif