/*
BSD 3-Clause License

Copyright (c) 2022, Jungbae Nam <jungbae.nam@gmail.com>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef TEST_FLINT_H
#define TEST_FLINT_H

#ifdef ULONG_EXTRAS_INLINES_C
#define ULONG_EXTRAS_INLINE FLINT_DLL
#else
#define ULONG_EXTRAS_INLINE static __inline__
#endif

#undef ulong
#define ulong ulongxx
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#undef ulong
#define ulong mp_limb_t

#include "gmp.h"
#include "flint/flint.h"
#include "flint/ulong_extras.h"
#include "flint/profiler.h"

#ifdef __cplusplus
 extern "C" {
#endif

// limit of the number of digits for Fourier coefficients of L(E, s)
#define TWISTS_LEN_STR_LONG 20

// constant pi
#define twists_pi 3.141592653589793238462643383279502884L

// the sum of truncated terms in L(E,1,chi)
#define TWISTS_ERROR_BOUND 0.000000000000001

// the minimum number of terms for L(E,1,chi) to be computed
//#define twists_min_num_terms 90000UL
//#define twists_sqrt_min_num_terms_div_by_4 75.0

// type of array elements as the values of chi of order int k, which should be 0 to k-1
// if Cuda has more memory it can be increased as 4 bytes or larger.

//typedef int twists_k_e;
typedef unsigned short twists_k_e;

typedef struct
{
    ulong len;
    slong *a_ptr;
}
twists_slong_array;


typedef struct
{
    ulong len;
    int *a_ptr;
}
twists_int_array;

typedef struct
{
    ulong len;
    ulong *a_ptr;
}
twists_ulong_array;

typedef struct
{
    ulong f_i;
    int a_i;
    ulong *f_a; //f_i^a for 1 <= a <= a_i
    double *f_a_inv; //f_i^(-a) for 1 <= a <= a_i
}
twists_f_i;

typedef struct
{
   int len;
   twists_f_i *array_f_i;
}
twists_f;

typedef struct
{
    ulong f_i;
    int a_i;
    ulong f_a_i; // f_i^(a_i)
    double f_a_i_inv; // f_i^(-a_i)
    int *eprime_j; // exponents of k_j, which is 0, 1, 2,..., e_j. 
}
twists_f_i_chi_k;

typedef struct
{
   int len_f;
   int len_k;
   twists_f_i_chi_k *array_f_i_k; //with length len_f
}
twists_f_chi_k;

// structure for chi_i's of prime power conductor f_i^{a_i} of order dividing k.
typedef struct
{
   int num_chi_i; //should be initialized by 0
   ulong *r_i_ptr; //should be initialized by NULL
   int *flag_k_ptr; //should be initialized by NULL
   twists_k_e **chi_i_2d_arr_ptr; //should be initialized by NULL
}
twists_chi_i;

// structure for chi's of a fixed conductor f_chi of order k. 
typedef struct
{
   ulong f_chi; // conductor
   ulong num_chi_f; // number of characters of conductor f_chi and order k
   ulong *r; // labels of such characters
   twists_k_e **chi_2d_arr_ptr; // 2d arrays of values of such characters
}
twists_chi;

FLINT_DLL ulong twists_mod_2(slong s_n);

FLINT_DLL twists_k_e twists_mod_k(ulong a, ulong k, double kinv);

FLINT_DLL bool twists_is_even(ulong n);

FLINT_DLL int twists_ord_2(ulong n);

FLINT_DLL int twists_set_bit(int n, int i);

FLINT_DLL ulong twists_base_k_to_ul(ulong *m_r_i_ptr, ulong k, int len_m_r_i);

FLINT_DLL ulong twists_gcd(ulong a, ulong b);

FLINT_DLL twists_f *twists_factor(ulong n);

FLINT_DLL void twists_factor_free(twists_f *f_n);

FLINT_DLL ulong twists_euler_phi_composite(ulong *f_i_ptr, int *a_i_ptr, int len);

FLINT_DLL ulong twists_euler_phi_prime(ulong f_i, int a_i);

FLINT_DLL twists_f_chi_k *twists_init_f_chi_k(int len_k);

FLINT_DLL int twists_add_f_i_chi_k(twists_f_chi_k *f_k_ptr, ulong p, int e);

FLINT_DLL int twists_pow_f_i_chi_k(twists_f_chi_k *f_k_ptr);

FLINT_DLL void twists_f_chi_k_free(twists_f_chi_k *f_k);

FLINT_DLL ulong twists_primitive_root(ulong p, int e);

FLINT_DLL int *twists_chi_i_gcd(ulong r_i, twists_f *k_ptr, int *eprimej);

FLINT_DLL int *twists_chi_i_div(int *a_ptr, int *b_ptr, twists_f *k_ptr);

FLINT_DLL ulong twists_chi_i_eval(int *gcd_ptr, twists_f *k_ptr);

FLINT_DLL bool twists_is_primitive(ulong r_i, ulong f_i, int a_i, twists_f *k_ptr, int *eprimej, int **ordj_ptr);

FLINT_DLL void twists_int_arr_free(int *arr_ptr);

FLINT_DLL void twists_slong_arr_free(slong *arr_ptr);

FLINT_DLL void twists_ulong_arr_free(ulong *arr_ptr);

FLINT_DLL void twists_twist_arr_free(twists_k_e *arr_ptr);

FLINT_DLL void twists_chi_i_s_free(twists_chi_i chi_i_s);

FLINT_DLL void twists_chi_free(twists_chi chi);

FLINT_DLL slong twists_str_to_sl(const char *str);

FLINT_DLL int twists_load_an(twists_int_array *a_an, char *f_path);

FLINT_DLL bool twists_is_twist(ulong f, ulong k, ulong N, int m_k, twists_f_chi_k *n_f_k_ptr, twists_f *f_k_ptr);

FLINT_DLL twists_chi_i twists_prime_character(ulong k, double kinv, int *k_e_ptr, int m_k, twists_f_i_chi_k *f_k_i_ptr, twists_f *k_ptr);

FLINT_DLL twists_k_e *twists_create_chi(ulong f, ulong k, double kinv, int *r_i_j_ptr, twists_chi_i *chi_i_s_ptr, \
   twists_f_chi_k *t_f_k_ptr);

FLINT_DLL void twists_rec_gen_idices(ulong f, ulong k, double kinv, int i, int *r_i_j_ptr, int m_k, twists_chi_i *chi_i_s_ptr, \
    twists_f_chi_k *t_f_k_ptr, int gprime_k, twists_chi *chi_ptr);

FLINT_DLL int twists_dirichlet_character(ulong f, ulong k, double kinv, int *k_e_ptr, int m_k, twists_f_chi_k *t_f_k_ptr, \
   twists_f *k_ptr, twists_chi *chi_ptr);

// assume that the largest f is 3*10^6, the desired number of digits of L(E,1,chi) is 4 and the minimum number of terms computed is 10^5.
// thus the largest N_E should be 151.
FLINT_DLL ulong twists_num_terms(ulong f, ulong N);

#ifdef __cplusplus
}
#endif

#endif

