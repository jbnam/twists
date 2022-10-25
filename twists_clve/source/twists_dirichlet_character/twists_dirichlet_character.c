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


#include "twists_dirichlet_character.h"

ulong twists_mod_2(slong s_n)
{
    return (abs(s_n)&1);
}

// return a mod k using the precomputed 1/k where k is the order of a character.
// return k if a = 0 mod k.
twists_k_e twists_mod_k(ulong a, ulong k, double kinv)
{
    twists_k_e a_mod_k;

    a_mod_k = (twists_k_e) n_mod_precomp(a, k, kinv);
    if(a_mod_k == 0)
        a_mod_k = (twists_k_e)k;
    return a_mod_k;
}

bool twists_is_even(ulong n)
{
    if((n&1) == 0)
        return true;
    else
        return false;
}

// assume n is positive.
int twists_ord_2(ulong n)
{
    int m = 0;

    if(n == 0)
    {
        fprintf(stderr, "Error: input %d should be positive.\n", n);
        exit(EXIT_FAILURE);
    }

    while((n&1) == 0)
    {
        m++;
        n = (n>>1);
    }

    return m;
}

// set the i-th bit of n to 1.
int twists_set_bit(int n, int i)
{
    int m = 1;

    m = m << i;
    return (n | m);
}

// return ulong integer in base k for the label of a primitive character of order k.
ulong twists_base_k_to_ul(ulong *m_r_i_ptr, ulong k, int len_m_r_i)
{
    ulong r = 0, temp_k = 1;
    int j, temp_len;

    temp_len = len_m_r_i - 1;

    for(j = 0; j < len_m_r_i; j++)
    {
        r += (m_r_i_ptr[temp_len -j]-1)*temp_k;
        temp_k *= (k-1);
    }
    return (r+1);
}

ulong twists_gcd(ulong a, ulong b)
{   return n_gcd(a,b); }

// factor a positive integer n with the inverses of each prime power factors
twists_f *twists_factor(ulong n)
{
    int i, j;
    n_factor_t n_f;
    twists_f *t_f_ptr = NULL;
    twists_f_i *arr_f_i = NULL;
    double f_i_inv;

    n_factor_init(&n_f);
    n_factor(&n_f, n, 0); //no need of proof of prime factors since n < 2^{64}

    t_f_ptr = flint_malloc(sizeof(twists_f));
    t_f_ptr->len = n_f.num;

    arr_f_i = flint_malloc(t_f_ptr->len*sizeof(twists_f_i));
    
    for(j = 0; j < t_f_ptr->len; j++)
    {
        arr_f_i[j].f_i = n_f.p[j];
        arr_f_i[j].a_i = n_f.exp[j];
        arr_f_i[j].f_a = flint_malloc((arr_f_i[j].a_i)*sizeof(ulong));
        arr_f_i[j].f_a_inv = flint_malloc((arr_f_i[j].a_i)*sizeof(double));

        arr_f_i[j].f_a[0] = arr_f_i[j].f_i;
        f_i_inv = n_precompute_inverse(arr_f_i[j].f_i);
        arr_f_i[j].f_a_inv[0] = f_i_inv;
        
        for(i = 1; i < arr_f_i[j].a_i; i++)
        {
            arr_f_i[j].f_a[i] = (arr_f_i[j].f_a[i-1])*(arr_f_i[j].f_i);
            arr_f_i[j].f_a_inv[i] = (arr_f_i[j].f_a_inv[i-1])*f_i_inv;
        }
    }

    t_f_ptr->array_f_i = arr_f_i;

    return t_f_ptr;
}

// free twists_k.
void twists_factor_free(twists_f *f_n_ptr)
{
    int j;

    for(j = 0; j < f_n_ptr->len; j++)
    {
        flint_free(f_n_ptr->array_f_i[j].f_a);
        flint_free(f_n_ptr->array_f_i[j].f_a_inv);
    }
    flint_free(f_n_ptr->array_f_i);
    flint_free(f_n_ptr);
}

// assume that a_i_ptr[j] > 0 for all j.
ulong twists_euler_phi_composite(ulong *f_i_ptr, int *a_i_ptr, int len)
{
    ulong eulerphi = 1;
    int j;

    for(j = 0; j < len; j++)
        eulerphi *= twists_euler_phi_prime(f_i_ptr[j], a_i_ptr[j]);
    return eulerphi;
}

// assume that a_i > 0.
ulong twists_euler_phi_prime(ulong f_i, int a_i)
{
    return (f_i-1)*n_pow(f_i, (ulong)(a_i-1));
}


twists_f_chi_k *twists_init_f_chi_k(int length_k)
{
    twists_f_chi_k *f_k_ptr = NULL;

    f_k_ptr = flint_malloc(sizeof(twists_f_chi_k));
    f_k_ptr->len_f = 0;
    f_k_ptr->len_k = length_k;
    f_k_ptr->array_f_i_k = NULL;

    return f_k_ptr;
}

int twists_add_f_i_chi_k(twists_f_chi_k *f_k_ptr, ulong p, int e)
{
    twists_f_i_chi_k *f_i_k_ptr = NULL;
    
    f_k_ptr->len_f += 1;
    f_k_ptr->array_f_i_k = flint_realloc(f_k_ptr->array_f_i_k, (f_k_ptr->len_f)*sizeof(twists_f_i_chi_k));
    f_k_ptr->array_f_i_k[f_k_ptr->len_f-1].f_i = p;
    f_k_ptr->array_f_i_k[f_k_ptr->len_f-1].a_i = e;
    f_k_ptr->array_f_i_k[f_k_ptr->len_f-1].f_a_i = 0;
    f_k_ptr->array_f_i_k[f_k_ptr->len_f-1].f_a_i_inv = 0;
    f_k_ptr->array_f_i_k[f_k_ptr->len_f-1].eprime_j = flint_calloc(f_k_ptr->len_k, sizeof(int));

    return EXIT_SUCCESS;
}

int twists_pow_f_i_chi_k(twists_f_chi_k *f_k_ptr)
{
    int j;

    for(j = 0; j < f_k_ptr->len_f; j++)
    {
        f_k_ptr->array_f_i_k[j].f_a_i = n_pow(f_k_ptr->array_f_i_k[j].f_i, (ulong)f_k_ptr->array_f_i_k[j].a_i);
        f_k_ptr->array_f_i_k[j].f_a_i_inv = n_precompute_inverse(f_k_ptr->array_f_i_k[j].f_a_i);
    }

    return EXIT_SUCCESS;
}

// free twists_f_chi_k.
void twists_f_chi_k_free(twists_f_chi_k *f_k)
{
    int i;

    for(i = 0; i < f_k->len_f; i++)
        if(f_k->array_f_i_k[i].eprime_j != NULL)
            flint_free(f_k->array_f_i_k[i].eprime_j);
    if(f_k->array_f_i_k != NULL)
        flint_free(f_k->array_f_i_k);
    flint_free(f_k);
}

// compute the primitive root modulo p^e.
// Note for p = 2 and e > 2, no primitive root exists (isomorpic to <-1><5>) but return 5 instead.
ulong twists_primitive_root(ulong p, int e)
{
    ulong r, p2;
    double p2inv;

    if (p == 2)
    {
        if (e == 1)
            return 1;
        if (e == 2)
            return 3;
        else
            return 5;
    }
    else
    {
        r = n_primitive_root_prime(p);
        if (e == 1)
            return r;
        else
        {
            p2 = p*p;
            p2inv = n_precompute_inverse(p2);
            if(n_powmod_ui_precomp(r, p-1, p2, p2inv) != 1)
                return r;
            else
                return r+p;
        }
    }
}

int *twists_chi_i_gcd(ulong r_i, twists_f *k_ptr, int *eprimej)
{
    int i, j, *arr_ptr;

    arr_ptr = flint_calloc(k_ptr->len, sizeof(int));

    for(i = 0; i < k_ptr->len; i++)
        for(j = 0; j < eprimej[i]; j++)
        {
            if(n_mod_precomp(r_i, k_ptr->array_f_i[i].f_a[j], k_ptr->array_f_i[i].f_a_inv[j]) == 0)
                arr_ptr[i] += 1;
            else
                break;
        }

    return arr_ptr;
}

int *twists_chi_i_div(int *a_ptr, int *b_ptr, twists_f *k_ptr)
{
    int j, *c_ptr;

    c_ptr = flint_calloc(k_ptr->len, sizeof(int));

    for(j = 0; j < k_ptr->len; j++)
            c_ptr[j] = a_ptr[j]-b_ptr[j];

    return c_ptr;
}

ulong twists_chi_i_eval(int *a_ptr, twists_f *k_ptr)
{
    ulong a = 1;
    int j;

    for(j = 0; j < k_ptr->len; j++)
        if(a_ptr[j] != 0)
            a *= k_ptr->array_f_i[j].f_a[a_ptr[j]-1];

    return a;
}

void twists_int_arr_free(int *arr_ptr)
{
    flint_free(arr_ptr);
}

void twists_slong_arr_free(slong *arr_ptr)
{
    flint_free(arr_ptr);
}

void twists_ulong_arr_free(ulong *arr_ptr)
{
    flint_free(arr_ptr);
}

void twists_twist_arr_free(twists_k_e *arr_ptr)
{
    flint_free(arr_ptr);
}

void twists_chi_i_s_free(twists_chi_i chi_i_s)
{
    int j;

    if(chi_i_s.num_chi_i != 0)
    {
        flint_free(chi_i_s.r_i_ptr);
        flint_free(chi_i_s.flag_k_ptr);
        for(j = 0; j < chi_i_s.num_chi_i; j++)
            flint_free(chi_i_s.chi_i_2d_arr_ptr[j]);
        flint_free(chi_i_s.chi_i_2d_arr_ptr);
    }
}

void twists_chi_free(twists_chi chi)
{
    int j;

    if(chi.num_chi_f != 0)
    {
        flint_free(chi.r);
        for(j = 0; j < chi.num_chi_f; j++)
            flint_free(chi.chi_2d_arr_ptr[j]);
        flint_free(chi.chi_2d_arr_ptr);
    }
}

bool twists_is_primitive(ulong r_i, ulong f_i, int a_i, twists_f *k_ptr, int *eprimej, int **ordj_ptr)
{
    int j, *t_gcd = NULL;

    t_gcd = twists_chi_i_gcd(r_i, k_ptr, eprimej);

    *ordj_ptr = twists_chi_i_div(eprimej, t_gcd, k_ptr);  

    for(j = 0; j < k_ptr->len; j++)
    {    
        if(f_i == k_ptr->array_f_i[j].f_i)
        {
            if(f_i == 2)
            {
                if(!((eprimej[j] == 1 && (*ordj_ptr)[j] == 1) || (eprimej[j] > 1 && (*ordj_ptr)[j] == (a_i-2))))
                {
                    twists_int_arr_free(t_gcd);
                    return false;
                }
            }
            else
            {
                if(!((*ordj_ptr)[j] == (a_i-1)))
                {
                    twists_int_arr_free(t_gcd);
                    return false;
                }
            }
        }
    }
    
    twists_int_arr_free(t_gcd);

    return true;
}

// Input: string with length TWISTS_LEN_STR_LONG
slong twists_str_to_sl(const char *str)
{
    return atol(str);
}

// load the coefficients a_n located in f_path; the length of it is one larger than that in the file due to a_0 := 0.
int twists_load_an(twists_int_array *a_an, char *f_path)
{
    // Open the file for reading
    char *line_buf = NULL;
    size_t line_buf_size = 0;
    FILE *fp = fopen(f_path, "r");

    // initialize array of an as twists_slong_array
    a_an->len = 1;
    a_an->a_ptr = flint_malloc(sizeof(int));    
    if((a_an->a_ptr) == NULL)
    {
        fprintf(stderr, "Error: memory allocation failed for reading integer in %s\n", f_path);
        exit(EXIT_FAILURE);
    }
    a_an->a_ptr[a_an->len] = 0;


    if (!fp)
    {
        fprintf(stderr, "Error opening file %s\n", f_path);
        exit(EXIT_FAILURE);
    }

    // Loop through until we are done with the file
    while (getline(&line_buf, &line_buf_size, fp) >= 0)
    {
        a_an->len += 1;
        a_an->a_ptr = flint_realloc(a_an->a_ptr, (a_an->len)*sizeof(int));
        if((a_an->a_ptr) == NULL)
        {
            fprintf(stderr, "Error: memory allocation failed for reading integer in %s\n", f_path);
            free(line_buf);
            fclose(fp);
            exit(EXIT_FAILURE);
        }
        a_an->a_ptr[(a_an->len)-1] = atoi(line_buf);
    }

    // Free the allocated line buffer
    free(line_buf);
    line_buf = NULL;
    // Close the file now that we are done with it
    fclose(fp);

    return EXIT_SUCCESS;
}

// Inputs: modulus f, order k, conductor of elliptic curve N, 2^nu(k)-1,  pointer of factors of f, pointer of factors of k
//         n_f_k_ptr should be NULL in the calling routine.
// Output: true if f is a conductor of character of order k coprime to N and false otherwise
bool twists_is_twist(ulong f, ulong k, ulong N, int m_k, twists_f_chi_k *n_f_k_ptr, twists_f *f_k_ptr)
{
    bool b_k_j;
    ulong f2, k2;
    int i, j, l, h, e, g_k = 0, jj = 0;
    n_factor_t f_f;

    if(twists_gcd(f, N) > 1)
        return false;

    if(twists_is_even(f))
    {
        if(!twists_is_even(k))
            return false;
        else 
        {
            f2 = twists_ord_2(f);
            k2 = twists_ord_2(k);
            if(f2 == 1 || f2 > (k2+2))
                return false;
            else 
            {
                twists_add_f_i_chi_k(n_f_k_ptr, 2, f2);
                if(k2 == 1)
                {
                    n_f_k_ptr->array_f_i_k[n_f_k_ptr->len_f-1].eprime_j[0] = k2;
                    g_k = 1;
                    jj = 1;
                }
                else if(k2 > 1)
                {
                    if(f2 == 2)
                        n_f_k_ptr->array_f_i_k[n_f_k_ptr->len_f-1].eprime_j[0] = f2-1;
                    else
                        n_f_k_ptr->array_f_i_k[n_f_k_ptr->len_f-1].eprime_j[0] = f2-2;
                    if(f2 == (k2+2))
                        g_k = 1;
                }
            }
            f = (f >> f2);
            jj = 1;
        }
    }
    // Now f is odd and coprime to N.
    // Factorize f
    n_factor_init(&f_f);
    n_factor(&f_f, f, 0); //no need of proof of prime factors since n < 2^{64}

    if((f_k_ptr->len >= 2) && (f_f.num == 0))
        return false;

    for(i = 0; i < f_f.num; i++)
    {
        b_k_j = true;
        for(j = 0; j < f_k_ptr->len; j++)
        {
            if(f_f.p[i] == f_k_ptr->array_f_i[j].f_i)
            {
                b_k_j = false;
                break;
            }
        }
        if(b_k_j && (f_f.exp[i] > 1))
            return false;
    }

    for(i = jj; i < (f_f.num+jj); i++)
    {
        twists_add_f_i_chi_k(n_f_k_ptr, f_f.p[i-jj], f_f.exp[i-jj]);
        b_k_j = false;

        for(j = 0; j < f_k_ptr->len; j++)
        {
            if(n_f_k_ptr->array_f_i_k[i].f_i == f_k_ptr->array_f_i[j].f_i)
            {
                if((n_f_k_ptr->array_f_i_k[i].a_i > 1) && (n_f_k_ptr->array_f_i_k[i].a_i <= (f_k_ptr->array_f_i[j].a_i + 1)))
                {
                    n_f_k_ptr->array_f_i_k[i].eprime_j[j] = (f_f.exp[i]-1);
                    b_k_j = true;
                    if(n_f_k_ptr->array_f_i_k[i].a_i == (f_k_ptr->array_f_i[j].a_i + 1))
                    {
                        n_f_k_ptr->array_f_i_k[i].eprime_j[j] = f_k_ptr->array_f_i[j].a_i;
                        g_k = twists_set_bit(g_k, j);
                        for(l = 0; l < j; l++)
                        {   
                            h = 0;
                            while(h < f_k_ptr->array_f_i[l].a_i)
                            {
                                if(n_mod_precomp(n_f_k_ptr->array_f_i_k[i].f_i, f_k_ptr->array_f_i[l].f_a[h], f_k_ptr->array_f_i[l].f_a_inv[h]) != 1)
                                    break;
                                h++;
                            }
                            if(h == f_k_ptr->array_f_i[l].a_i)
                                g_k = twists_set_bit(g_k, l);
                            n_f_k_ptr->array_f_i_k[i].eprime_j[l] = h;
                        }
                    }
                }
            }
            else
            {
                    h = 0;
                    while(h < f_k_ptr->array_f_i[j].a_i)
                    {
                        if(n_mod_precomp(n_f_k_ptr->array_f_i_k[i].f_i, f_k_ptr->array_f_i[j].f_a[h], f_k_ptr->array_f_i[j].f_a_inv[h]) != 1)
                            break;
                        h++;
                    }
                    if((h != 0) && (n_f_k_ptr->array_f_i_k[i].a_i == 1))
                        b_k_j = true;
                    if(h == f_k_ptr->array_f_i[j].a_i)
                        g_k = twists_set_bit(g_k, j);
                    n_f_k_ptr->array_f_i_k[i].eprime_j[j] = h;
            }
        }
        if(b_k_j == false)
            return false;
    }
    if(g_k != m_k)
        return false;
    else
        return true;
}

// compute a character whose factor is a prime power divisor of f and order is k'/(k',r_i). 
// c_2 is \pm 1 when f_i = 2 and a_i >= 3. Otherwise, we ignore it.
// g_i = a primitive root modulo f_i^(a_i). 
// ephi_i = euler_phi(f_i^(a_i)).
// chi_i is of size twists_k_e (unsigned short (2 bytes)) with elements d_i*r_i (mod k).
twists_chi_i twists_prime_character(ulong k, double kinv, int *k_e_ptr, int m_k, twists_f_i_chi_k *f_k_i_ptr, twists_f *k_ptr)
{
    int j, *ordj_ptr = NULL, *d_k_i_ptr = NULL;
    register ulong ui, g_i, uj = 0, m_i = 0, g = 1;
    ulong r_i = 1, kprime, d_k_i, ephi_i, c_2 = 1;
    twists_chi_i chi_i_s = {.num_chi_i = 0, .r_i_ptr = NULL, .flag_k_ptr = NULL, .chi_i_2d_arr_ptr = NULL};

    // compute k'.
    kprime = twists_chi_i_eval(f_k_i_ptr->eprime_j, k_ptr);

    // compute d_k_i = k/k'.
    d_k_i_ptr = twists_chi_i_div(k_e_ptr, f_k_i_ptr->eprime_j, k_ptr);
    d_k_i = twists_chi_i_eval(d_k_i_ptr, k_ptr);
    twists_int_arr_free(d_k_i_ptr);

    // compute ephi_i = euler_phi(k')
    ephi_i = twists_euler_phi_prime(f_k_i_ptr->f_i, f_k_i_ptr->a_i);
    if(f_k_i_ptr->f_i == 2 && f_k_i_ptr->a_i > 2)
        ephi_i = (ephi_i >> 1);

    // compute g_i
    g_i = twists_primitive_root(f_k_i_ptr->f_i, f_k_i_ptr->a_i);

    while(r_i < kprime)
    {
        m_i += d_k_i;

        if(twists_is_primitive(r_i, f_k_i_ptr->f_i, f_k_i_ptr->a_i, k_ptr, f_k_i_ptr->eprime_j, &ordj_ptr) == true)
        {
            // add an element in chi_i_s.
            chi_i_s.num_chi_i += 1;
            chi_i_s.r_i_ptr = flint_realloc(chi_i_s.r_i_ptr, (chi_i_s.num_chi_i)*sizeof(ulong));
            chi_i_s.flag_k_ptr = flint_realloc(chi_i_s.flag_k_ptr, (chi_i_s.num_chi_i)*sizeof(int));
            chi_i_s.chi_i_2d_arr_ptr = flint_realloc(chi_i_s.chi_i_2d_arr_ptr, (chi_i_s.num_chi_i)*sizeof(twists_k_e *));
            chi_i_s.chi_i_2d_arr_ptr[chi_i_s.num_chi_i-1] = flint_calloc(f_k_i_ptr->f_a_i, sizeof(twists_k_e));

            // update ord(chi) by checking flag_k_ptr[chi_i_s.num_chi_i-1] with ordj_ptr.
            // create chi_i and use flint_calloc since each element should be intinitialized as 0.
            chi_i_s.flag_k_ptr[chi_i_s.num_chi_i-1] = 0;
            for(j = 0; j < k_ptr->len; j++)
                if(ordj_ptr[j] == k_ptr->array_f_i[j].a_i)
                    chi_i_s.flag_k_ptr[chi_i_s.num_chi_i-1] = twists_set_bit(chi_i_s.flag_k_ptr[chi_i_s.num_chi_i-1], j);

            // set r_i_ptr[chi_i_s.num_chi_i-1].
            chi_i_s.r_i_ptr[chi_i_s.num_chi_i-1] = m_i;
            
            // set the values of chi_i.
            if(f_k_i_ptr->f_i == 2 && f_k_i_ptr->a_i > 2)
            {            
                // add chi_i for chi(-1) = -1.
                chi_i_s.num_chi_i += 1;
                chi_i_s.r_i_ptr = flint_realloc(chi_i_s.r_i_ptr, (chi_i_s.num_chi_i)*sizeof(ulong));
                chi_i_s.flag_k_ptr = flint_realloc(chi_i_s.flag_k_ptr, (chi_i_s.num_chi_i)*sizeof(int));
                chi_i_s.chi_i_2d_arr_ptr = flint_realloc(chi_i_s.chi_i_2d_arr_ptr, (chi_i_s.num_chi_i)*sizeof(twists_k_e *));
                chi_i_s.chi_i_2d_arr_ptr[chi_i_s.num_chi_i-1] = flint_calloc(f_k_i_ptr->f_a_i, sizeof(twists_k_e));

                chi_i_s.r_i_ptr[chi_i_s.num_chi_i-1] = (f_k_i_ptr->f_a_i)-m_i;
                chi_i_s.flag_k_ptr[chi_i_s.num_chi_i-1] = chi_i_s.flag_k_ptr[chi_i_s.num_chi_i-2];
            
                for(ui = 1; ui <= ephi_i; ui++)
                {
                    g *= g_i;
                    g = n_mod_precomp(g, f_k_i_ptr->f_a_i, f_k_i_ptr->f_a_i_inv);
                    uj += m_i;
                    chi_i_s.chi_i_2d_arr_ptr[chi_i_s.num_chi_i-2][g] = twists_mod_k(uj, k, kinv);
                    chi_i_s.chi_i_2d_arr_ptr[chi_i_s.num_chi_i-2][f_k_i_ptr->f_a_i-g] = chi_i_s.chi_i_2d_arr_ptr[chi_i_s.num_chi_i-2][g];
                    chi_i_s.chi_i_2d_arr_ptr[chi_i_s.num_chi_i-1][g] = chi_i_s.chi_i_2d_arr_ptr[chi_i_s.num_chi_i-2][g];
                    chi_i_s.chi_i_2d_arr_ptr[chi_i_s.num_chi_i-1][f_k_i_ptr->f_a_i-g] \
                            = twists_mod_k(((ulong)chi_i_s.chi_i_2d_arr_ptr[chi_i_s.num_chi_i-1][g]) + (k >> 1), k, kinv);
                }

            }
            else
            {

                for(ui = 1; ui <= ephi_i; ui++)
                {
                    g *= g_i;
                    g = n_mod_precomp(g, f_k_i_ptr->f_a_i, f_k_i_ptr->f_a_i_inv);
                    uj += m_i;
                    chi_i_s.chi_i_2d_arr_ptr[chi_i_s.num_chi_i-1][g] = twists_mod_k(uj, k, kinv);
                }
            }
        }
        r_i += 1;
        twists_int_arr_free(ordj_ptr);
        ordj_ptr = NULL;
    }

    return chi_i_s;
}

// given array of indicies for chi_i.
twists_k_e *twists_create_chi(ulong f, ulong k, double kinv, int *r_i_j_ptr, twists_chi_i *chi_i_s_ptr, twists_f_chi_k *t_f_k_ptr)
{
    int i;
    ulong j, *jj_ptr = NULL;
    register twists_k_e k_e;
    register ulong t_e;
    twists_k_e *chi_ptr = NULL;

    chi_ptr = flint_malloc(f*sizeof(twists_k_e));
    jj_ptr = flint_calloc(t_f_k_ptr->len_f, sizeof(ulong));

    for(j = 0; j < f; j++)
    {
        k_e = 0;
        for(i = 0; i < t_f_k_ptr->len_f; i++)
        {
            if(!(jj_ptr[i] < t_f_k_ptr->array_f_i_k[i].f_a_i))
                jj_ptr[i] -= (t_f_k_ptr->array_f_i_k[i].f_a_i);
            t_e = chi_i_s_ptr[i].chi_i_2d_arr_ptr[r_i_j_ptr[i]][jj_ptr[i]];
            if(t_e == 0)
            {
                k_e = 0;
                break;
            }
            else
                k_e += t_e;
        }
        if(k_e == 0)
            chi_ptr[j] = 0;
        else
            chi_ptr[j] = twists_mod_k((ulong)k_e, k, kinv);
        for(i = 0; i < t_f_k_ptr->len_f; i++)
            jj_ptr[i] += 1;
    }
    flint_free(jj_ptr);
    return chi_ptr;
}

// recursive function to create chi with chi_i_s_ptr
// idx has length len_f and is initialized by 0.
void twists_rec_gen_idices(ulong f, ulong k, double kinv, int i, int *r_i_j_ptr, int m_k, twists_chi_i *chi_i_s_ptr, \
    twists_f_chi_k *t_f_k_ptr, int gprime_k, twists_chi *chi_ptr)
{
    int j, l;
    ulong *r_ptr = NULL;

    if(i == ((t_f_k_ptr->len_f)-1))
    {    
        for(j = 0; j < chi_i_s_ptr[i].num_chi_i; j++)
        {

            if((gprime_k | chi_i_s_ptr[i].flag_k_ptr[j]) == m_k)
            {
                // add chi into *chi.                                
                r_i_j_ptr[i] = j;
                r_ptr = flint_malloc((t_f_k_ptr->len_f)*sizeof(ulong));
                for(l = 0; l < t_f_k_ptr->len_f; l++)
                    r_ptr[l] = chi_i_s_ptr[l].r_i_ptr[r_i_j_ptr[l]];
                chi_ptr->num_chi_f += 1;
                chi_ptr->r = flint_realloc(chi_ptr->r, (chi_ptr->num_chi_f)*sizeof(ulong));
                chi_ptr->r[chi_ptr->num_chi_f-1] = twists_base_k_to_ul(r_ptr, k, t_f_k_ptr->len_f);
                chi_ptr->chi_2d_arr_ptr = flint_realloc(chi_ptr->chi_2d_arr_ptr, (chi_ptr->num_chi_f)*sizeof(twists_k_e *));
                chi_ptr->chi_2d_arr_ptr[chi_ptr->num_chi_f-1] = twists_create_chi(f, k, kinv, r_i_j_ptr, chi_i_s_ptr, t_f_k_ptr);
                flint_free(r_ptr);
            }
        }
    }
    else
    {
        for(j = 0; j < chi_i_s_ptr[i].num_chi_i; j++)
        {
            r_i_j_ptr[i] = j;
            twists_rec_gen_idices(f, k, kinv, (i+1), r_i_j_ptr, m_k, chi_i_s_ptr, t_f_k_ptr, (gprime_k | (chi_i_s_ptr[i].flag_k_ptr[j])), chi_ptr);
        }
    } 
}

int twists_dirichlet_character(ulong f, ulong k, double kinv, int *k_e_ptr, int m_k, twists_f_chi_k *t_f_k_ptr, \
    twists_f *k_ptr, twists_chi *chi_ptr)
{
    int i, g_k = 0;
    int *r_i_j_ptr = NULL;
    twists_chi_i *chi_i_s_ptr = NULL;

    // update f_a_i and f_a_i_inv.
    twists_pow_f_i_chi_k(t_f_k_ptr);

    // initialize chi_ptr
    chi_ptr->f_chi = f;
    chi_ptr->num_chi_f = 0;
    chi_ptr->r = NULL;
    chi_ptr->chi_2d_arr_ptr = NULL;

    // collect chi_i_s.
    chi_i_s_ptr = flint_malloc((t_f_k_ptr->len_f)*sizeof(twists_chi_i));

    for(i = 0; i < t_f_k_ptr->len_f; i++)
        chi_i_s_ptr[i] = twists_prime_character(k, kinv, k_e_ptr, m_k, &(t_f_k_ptr->array_f_i_k[i]), k_ptr);

    // loops over chi_i to generate chi.
    r_i_j_ptr = flint_calloc(t_f_k_ptr->len_f, sizeof(int));

    twists_rec_gen_idices(f, k, kinv, 0, r_i_j_ptr, m_k, chi_i_s_ptr, t_f_k_ptr, 0, chi_ptr);
    // free r_i_j_ptr.
    flint_free(r_i_j_ptr);

    // free chi_i_s_2d_ptr
    for(i = 0; i < t_f_k_ptr->len_f; i++)
        twists_chi_i_s_free(chi_i_s_ptr[i]);
    
    flint_free(chi_i_s_ptr);
    return EXIT_SUCCESS;
}

// assume that the largest f is 3*10^6, the desired number of digits of the algebraic parts of L(E,1,chi) is 4.
// thus the largest N_E should be 151.
ulong twists_num_terms(ulong f, ulong N)
{
    double q;

    q = exp(-2*twists_pi/(((double)f)*(sqrt((double)N))));
    if(f > 100000)
    {
        return (ulong)(0.864899964187738*log(TWISTS_ERROR_BOUND*(((double)1)-q))/log(q));
    }
    else if(f > 10000)
    {
        return (ulong)(1.43427433120128*log(TWISTS_ERROR_BOUND*(((double)1)-q))/log(q));
    }
    else if(f > 1000)
    {
        return (ulong)(2.02837021134845*log(TWISTS_ERROR_BOUND*(((double)1)-q))/log(q));
    }
    else
    {
        return (ulong)(3.46410161513776*log(TWISTS_ERROR_BOUND*(((double)1)-q))/log(q));
    }
}
