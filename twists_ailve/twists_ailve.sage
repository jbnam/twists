r"""
Set of functions for the algebraic parts and their integer values of central \\
L-values of elliptic curves twisted by primitive Dirichlet characters

EXAMPLES::

TODO <Lots and lots of examples>

AUTHORS:

- Jungbae Nam (2022-10-24): Initial version
"""

#*****************************************************************************
#BSD 3-Clause License
#
#Copyright (c) 2022, Jungbae Nam <jungbae.nam@gmail.com>
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#*****************************************************************************

import os, sys
import numpy as np
from zipfile import ZipFile

from bisect import bisect_left

#TWISTS DATA TYPES
tt_u16 = np.ushort  #unsigned short
tt_u32 = np.uint32  #unsigned integer
tt_u64 = np.uint64  #unsigned long
tt_i32 = np.int32   #signed integer
tt_i64 = np.int64   #signed integer
tt_c64 = np.cdouble #complex number
tt_r64 = np.double  #real number

DEFAULT_PREC = 53
FORGIVENESS_NUM_INT_DIGITS = 4
FORGIVENESS_NUM_LVAL_DIGITS = 9
RF = RealField(DEFAULT_PREC)

ERROR_INT_BND = 1./RF(10^FORGIVENESS_NUM_INT_DIGITS)

MAX_COUNT_A_CHI = 100

#NUM_ELEMENTS_PER_LINE_RAW_DATA = 13
DATA_DELIMITER = ', '

def tw_factor(n):
    r"""
        Return tuple of the discint prime factors and their powers of 
        a positive integer.
    """
    return tuple(factor(n))

def tw_is_equal_to_int(r, int_r):
    r"""Check if `abs(r - int_r)` is within ERROR_INT_BND or not."""
    if abs(r - int_r) < ERROR_INT_BND:
        return True
    else:
        return False

def tw_l_zeta_k(k, prec=DEFAULT_PREC):
    r"""Return the tuple of `0` and `zeta^j` for `0 \le j \le k` 
    where `zeta` is a primitive root of `k`-th unity."""
    CF = ComplexField(prec)
    z_k = CF.zeta(k)
    lzk = [CF(0)]
    n = 1
    z = z_k
    while n <= k:
        lzk.append(z)
        z *= z_k
        n += 1
    return tuple(lzk)

def tw_is_pow_zeta_minus_1(n, k):
    r"""check if `zeta^n` is `-1` or not for `1 \le n \le k`."""
    if is_even(k):
        return n == (k >> 1)
    else:
        return False

def tw_is_pow_zeta_1(n, k):
    r"""check if `zeta^n` is `1` or not for `1 \le n \le k`."""
    return n == k
    
def tw_alpha_chi(arr_chi_cond, arr_chi_label, arr_alg_part_chi, \
                 arr_exp_chi_minus_N, arr_exp_chi_c, eps_E, k):
    r"""
        Return `\alpha_\chi` defined in Proposition 2.1 of the paper of H. Kisilevsky
        and J. Nam: Small Algebraic Central Values of Twists of Elliptic `L`-Functions
    """
    arr_alp_chi = np.empty(arr_chi_cond.size, dtype=tt_r64)
    lzk = tw_l_zeta_k(k)
    
    for n in range(arr_chi_cond.size):
        if tw_is_pow_zeta_1(arr_exp_chi_minus_N[n], k):
            z = eps_E
        elif not tw_is_pow_zeta_minus_1(arr_exp_chi_minus_N[n], k):
            z = eps_E*lzk[arr_exp_chi_minus_N[n]]
        else: #z == -1
            z = -eps_E
        
        if z == 1:
            alp_chi = arr_alg_part_chi[n]
        elif z != -1:
            alp_chi = (1. + z.conjugate())*arr_alg_part_chi[n]
        else:
            alp_chi = (lzk[arr_exp_chi_c[n]] - lzk[arr_exp_chi_c[n]].conjugate())*arr_alg_part_chi[n]
        
        if tw_is_equal_to_int(alp_chi.imag, 0):
             arr_alp_chi[n] = alp_chi.real
        else:
            raise ValueError(f"For chi_cond {arr_chi_cond[n]} and chi_label {arr_chi_label[n]}, " 
                             f"its alp_chi {alp_chi} has a non-zero imaginary part.")
    return arr_alp_chi

#note that we predict the some common rational integer factor of alpha_chi's.
#the reason we have this definition is due the precision issue on its integer value.
#return the sage list of A_chi's and their gcd 
def tw_A_chi(k, arr_chi_cond, arr_chi_label, arr_alp_chi):
    r"""
        Computes the sage list of `A_\chi` the image of `\alpha_\chi` under 
        the field norm, as described in the paper of H. Kisilevsky and J. Nam: 
        Small Algebraic Central Values of Twists of Elliptic `L`-Functions, 
        and returns their greatest common divisor `g` and those of `A_chi/g`.
    """
    phi_k_div_by_2 = euler_phi(k)/2
    
    if phi_k_div_by_2 == 1: #when k = 2, 3, 4
        #round it since a in arr_alp_chi is a float64 
        l_A_chi = [int(round(a)) for a in arr_alp_chi]
        g = gcd(l_A_chi)
        if g != 1:
            l_A_chi = [int(a/g) for a in l_A_chi]
        return l_A_chi, g
    else:
        l_A_chi = [int(0) for n in range(arr_chi_cond.size)]
        max_label = [0, 0]
        label_digits = 0
        bs_k = k-1 #base of the label representation
        g = int(1)
        rec_denom = RF(1)
        const_count_A_chi = phi_k_div_by_2*MAX_COUNT_A_CHI
        
        l_coprime_to_k = [a for a in range(1, phi_k_div_by_2+1) if gcd(a, k) == 1]
        L_prec = DEFAULT_PREC*2
        K.<zt> = CyclotomicField(k)
        L.<r_zt> = K.maximal_totally_real_subfield()[0]
        Mink_L = L.minkowski_embedding(prec=L_prec)
        Inv_Mink_L = Mink_L.inverse()
        
        for n in range(arr_chi_cond.size):
            gal_orb_alp_chi = [arr_alp_chi[n]*rec_denom]
            
            #find the maximum chi_label among chi's with the same cond_chi
            if arr_chi_cond[n] > max_label[0]:
                max_label[0] = arr_chi_cond[n]
                for m in range(1, arr_chi_cond.size-n):
                    if arr_chi_cond[n+m] != arr_chi_cond[n]:
                        break
                max_label[1] = arr_chi_label[n+m-1]
                label_digits = Integer(max_label[1]-1).ndigits(bs_k)
                            
            #convert chi_label into a tuple of base k-1
            chi_label = Integer(arr_chi_label[n] - 1)
            chi_label_dig = chi_label.digits(base=bs_k, padto=label_digits)
            chi_label_dig = [a+1 for a in chi_label_dig]
                
            for m in range(1,len(l_coprime_to_k)):
                label_orb = 0
                bs = 1
                for b in chi_label_dig:
                    label_orb += ((b*l_coprime_to_k[m] % k)-1)*bs
                    bs *= bs_k
                label_orb += 1
                    
                #find chi_label = label_orb with the same cond_chi 
                #and multiply temp_A_chi with its alp_chi
                if label_orb < arr_chi_label[n]:
                    unit_label = -1
                elif label_orb > arr_chi_label[n]:
                    unit_label = 1
                else:
                    break
                label_orb_idx = unit_label
                while arr_chi_cond[n+label_orb_idx] == arr_chi_cond[n]:
                    if arr_chi_label[n+label_orb_idx] == label_orb:
                        gal_orb_alp_chi.append(arr_alp_chi[n+label_orb_idx]*rec_denom)
                        break
                    else:
                        label_orb_idx += unit_label
                if arr_chi_cond[n+label_orb_idx] != arr_chi_cond[n]:
                    raise ValueError(f"No Galois conjugate: {label_orb} of "
                    f"cond_chi: {arr_chi_cond[n]} chi_label: {arr_chi_label[n]} found.\n")
                        
            #compute Norm_{QQ_chi^+/QQ}(alp_chi)
            gal_orb_alp_chi.reverse()
            coeffs_alp_chi_real = Inv_Mink_L*matrix(phi_k_div_by_2,1,gal_orb_alp_chi)
            coeffs_alp_chi_int = vector([round(x[0]) for x in coeffs_alp_chi_real])
            
            #check if the coefficients are actually integers or not
            for m in range(len(gal_orb_alp_chi)):
                if tw_is_equal_to_int(coeffs_alp_chi_real[m][0],\
                                      coeffs_alp_chi_int[m]) == False:
                    raise ValueError(f"A coeffient {coeffs_alp_chi_real[m]} of " 
                                     f"cond_chi: {arr_chi_cond[n]} chi_label: {arr_chi_label[n]} "
                                     f"is not an integer.\n")
            l_A_chi_real = Mink_L*coeffs_alp_chi_int
            A_chi_real = prod(l_A_chi_real)
        
            #check if A_chi is an integer or not
            A_chi = int(round(A_chi_real))
            if tw_is_equal_to_int(A_chi_real, A_chi) == False:
                raise ValueError(f"A_chi {A_chi_real} of cond_chi: {arr_chi_cond[n]} "
                f"chi_label: {arr_chi_label[n]} is not an integer.\n")
            l_A_chi[n] = A_chi
        
            #compute the gcd of the partial A_chi's
            if n == const_count_A_chi:
                denom = int(1)
                g = gcd(l_A_chi[:n+1])
                ft_g = tw_factor(g)
                for a in ft_g:
                    b = a[1]//phi_k_div_by_2
                    if b >= 1:
                        denom *= a[0]^b
                rec_denom /= denom
                #divide the partial A_chi's by the gcd
                g = denom^phi_k_div_by_2
                l_A_chi[:n+1] = [int(a/g) for a in l_A_chi[:n+1]]

        #Lastly check if the gcd of A_chi and divided A_chi's by it.
        partial_g = gcd(l_A_chi)
        g = int(g*partial_g)
        if partial_g != 1:
            l_A_chi = [int(a/partial_g) for a in l_A_chi]
        
    return l_A_chi, g

class tw_central_l_values:
    r"""
    The data including the central `L`-values of an elliptic curve `E` 
    defined over `\mathbb{Q}` for a family of the primitive Dirichlet 
    characters `\chi` of a fixed order `k` and conductor `f_k \le X`.
    """
    def __init__(self, E_label, k, X, chi_cond, chi_label, l_value, gauss_sum_chi,\
                   exp_chi_N, c, exp_chi_c, exp_sign_chi, num_terms):
        self.E = EllipticCurve(E_label)
        self.k = int(k)
        self.num_twists = int(chi_cond.size)
        self.X = int(X) #X million for the maximum conductor of chi in data 
        
        self.chi_cond = chi_cond
        self.chi_label = chi_label
        self.l_value = l_value
        self.gauss_sum_chi = gauss_sum_chi
        self.exp_chi_N = exp_chi_N
        self.c = c
        self.exp_chi_c = exp_chi_c
        self.exp_sign_chi = exp_sign_chi
        self.num_terms = num_terms
    
    def E(self):
        return self.E
    
    def ord_chi(self):
        return self.k
    
    def num_twists(self):
        return self.num_twists
    
    def X(self):
        return self.X
    
    def __getitem__(self, n):
        r"""Return the `n`-th element of the twists data as a list."""
        return [self.E.conductor(), self.k, self.chi_cond[n], self.chi_label[n],\
                self.l_value[n], self.gauss_sum_chi[n], self.exp_chi_N[n], self.c[n],\
                self.exp_chi_c[n], self.exp_sign_chi[n], self.num_terms[n]]
    
    def __setitem__(self, n, l):
        r"""Replace the `n`-th element of the central l-value data by an input list."""
        if l[0] != self.E.conductor() or l[1] != self.k:
            raise ValueError(f"The conductor of elliptic curve or the order of\
            character does NOT match.")
        self.chi_cond[n] = tt_u32(l[2])
        self.chi_label[n] = tt_u32(l[3])
        self.l_value[n] = tt_c64(l[4])
        self.gauss_sum_chi[n] = tt_c64(l[5])
        self.exp_chi_N[n] = tt_u16(l[6])
        self.c[n] = tt_u32(l[7])
        self.exp_chi_c[n] = tt_u16(l[8])
        self.exp_sign_chi[n] = tt_16(l[9])
        self.num_terms[n] = tt_u64(l[10])

    @classmethod
    def load_from_dat(cls, E_label, k, path, X=3):
        f_name = E_label+'_'+str(k)+'_'+str(X)+'m_raw.dat'
        
        #find the number of rows in the file
        fp = open(path+f_name, 'r')
        for num_twists, line in enumerate(fp):
            pass
        num_twists += 1
        
        #create the twists data using numpy
        chi_cond = np.empty(num_twists, dtype=tt_u32)
        chi_label = np.empty(num_twists, dtype=tt_u32)
        l_value = np.empty(num_twists, dtype=tt_c64)
        gauss_sum_chi = np.empty(num_twists, dtype=tt_c64)
        exp_chi_N = np.empty(num_twists, dtype=tt_u16)
        c = np.empty(num_twists, dtype=tt_u32)
        exp_chi_c = np.empty(num_twists, dtype=tt_u16)
        exp_sign_chi = np.empty(num_twists, dtype=tt_u16)
        num_terms = np.empty(num_twists, dtype=tt_u64)
        
        #load the raw twists data using numpy
        fp.seek(0)
        for j, line in enumerate(fp):
            t_str = line.strip().split(DATA_DELIMITER)
            chi_cond[j] = t_str[2]
            chi_label[j] = t_str[3]
            l_value[j] = tt_c64(complex(tt_r64(t_str[4]), tt_r64(t_str[5])))
            gauss_sum_chi[j] = tt_c64(complex(tt_r64(t_str[6]), tt_r64(t_str[7])))
            exp_chi_N[j] = t_str[8]
            c[j] = t_str[9]
            exp_chi_c[j] = t_str[10]
            exp_sign_chi[j] = t_str[11]
            num_terms[j] = t_str[12]
            
        fp.close()
        
        return cls(E_label, k, X, chi_cond, chi_label, l_value, gauss_sum_chi,\
                   exp_chi_N, c, exp_chi_c, exp_sign_chi, num_terms)
    
    @classmethod
    def load_from_npz(cls, E_label, k, path, X=3):
        f_name = E_label+'/'+E_label+'_'+str(k)+'_'+str(X)+'m_central_l_values.npz'
        
        with np.load(path+f_name) as data:
            E_label = data['info_twists'][0]
            k = data['info_twists'][1]
            X = data['info_twists'][2]
            chi_cond = data['chi_cond']
            chi_label = data['chi_label']
            l_value = data['l_value']
            gauss_sum_chi = data['gauss_sum_chi']
            exp_chi_N = data['exp_chi_N']
            c = data['c']
            exp_chi_c = data['exp_chi_c']
            exp_sign_chi = data['exp_sign_chi']
            num_terms = data['num_terms']

        return cls(E_label, k, X, chi_cond, chi_label, l_value, gauss_sum_chi,\
                   exp_chi_N, c, exp_chi_c, exp_sign_chi, num_terms)
    
    def save_to_npz(self, path):
        os.makedirs(path+self.E.label(), exist_ok=True)
        f_name = self.E.label()+'/'+self.E.label()+'_'+str(self.k)+'_'+\
        str(self.X)+'m_central_l_values.npz'

        info_twists = np.array([self.E.label(), self.k, self.X])
        
        np.savez_compressed(path+f_name, info_twists=info_twists, chi_cond=self.chi_cond,\
                            chi_label=self.chi_label, l_value=self.l_value,\
                            gauss_sum_chi=self.gauss_sum_chi, exp_chi_N=self.exp_chi_N,\
                            c=self.c, exp_chi_c=self.exp_chi_c,\
                            exp_sign_chi=self.exp_sign_chi, num_terms=self.num_terms)

class tw_alg_int_l_values:
    r"""
    The data including the algebraic parts and integral `L`-values of 
    an elliptic curve `E` defined over `\mathbb{Q}` for a family of 
    the primitive Dirichlet characters `\chi` of a fixed order `k` and
    conductor `f_k \le X`.
    """
    def __init__(self, E, k, X, g, chi_cond, chi_label, A_chi_div_g, alg_part_l, \
                alp_chi, exp_sign_chi, exp_chi_minus_N):
        self.E = E
        self.k = int(k)
        self.num_twists = int(chi_cond.size)
        self.X = int(X)
        self.g = int(g) #gcd of A_chi's
        
        self.chi_cond = chi_cond
        self.chi_label = chi_label
        self.A_chi_div_g = A_chi_div_g #list of python int type values
        self.alg_part_l = alg_part_l
        self.alp_chi = alp_chi 
        self.exp_sign_chi = exp_sign_chi 
        self.exp_chi_minus_N = exp_chi_minus_N
        
    def E(self):
        return self.E
    
    def ord_chi(self):
        return self.k
    
    def num_twists(self):
        return self.num_twists
    
    def X(self):
        return self.X
    
    def gcd_A_chi(self):
        return self.g
    
    def __getitem__(self, n):
        r"""Return the `n`-th element of the twists data as a list."""
        return [self.E.conductor(), self.k, self.g, self.chi_cond[n], self.chi_label[n],\
                self.A_chi_div_g[n], self.alg_part_l[n], self.alp_chi[n],\
                self.exp_sign_chi[n], self.exp_chi_minus_N[n]]
    
    def __setitem__(self, n, l):
        r"""Replace the `n`-th element of the algebraic part and integer value data 
        by an input list."""
        if l[0] != self.E.conductor() or l[1] != self.k:
            raise ValueError(f"The conductor of elliptic curve or the order of\
            character does NOT match.")
        self.chi_cond[n] = tt_u32(l[2])
        self.chi_label[n] = tt_u32(l[3])
        self.A_chi_div_g[n] = l[4]
        self.alg_part_l[n] = tt_c64(l[5])
        self.alp_chi[n] = tt_r64(l[6])
        self.exp_sign_chi[n] = tt_u16(l[9])
        self.exp_chi_minus_N[n] = tt_u16(l[10])
    
    @classmethod
    def load_from_central_l_values_npz(cls, E_label, k, path, X=3):
        
        #L is an object of class: tw_central_l_values
        L = tw_central_l_values.load_from_npz(E_label, k, path, X=X)
        
        E = EllipticCurve(L.E.label())
        k = L.k
        num_twists = L.num_twists
        X = L.X
        
        chi_cond = L.chi_cond
        chi_label = L.chi_label
        exp_sign_chi = L.exp_sign_chi
        exp_chi_minus_N = np.where((L.exp_chi_N + L.exp_sign_chi)%k == 0, k, \
                                   ((L.exp_chi_N + L.exp_sign_chi)%k))
        
        #Compute some initial invariants of E and chi
        eps_E = E.root_number()
        E_p_l = E.period_lattice()
        if E_p_l.are_linearly_dependent(E_p_l.basis()) == True:
            raise ValueError(f"Basis of the period lattice should be linearly\
            independent: {E_p_l.basis()}\n")
        #Assume E_p_l[0] is a positive real
        if E_p_l.basis()[0].imag() != 0 or E_p_l.basis()[0].real() <= 0:
            raise ValueError(f"The first element of the period lattice is not\
            a positive real: {E_p_l.basis()[0]}\n")
        if E_p_l.basis()[1].real() == 0:
            t_om = tuple([E_p_l.basis()[0], abs(E_p_l.basis()[1].imag())*I])
        else:
            if E_p_l.basis()[0] > E_p_l.basis()[1].real():
                t_om = tuple([E_p_l.basis()[0], abs(E_p_l.basis()[0]*(E_p_l.basis()[1].imag())\
                                                    /E_p_l.basis()[1].real())*I])
            else:
                t_om = tuple([E_p_l.basis()[0], abs(E_p_l.basis()[1].imag())*I])
        
        #Compute the algebraic parts of the central l-values
        alg_part_l = np.where(tw_is_pow_zeta_minus_1(exp_sign_chi, k), \
                                  -2.*(L.gauss_sum_chi.conjugate())*L.l_value/t_om[1],\
                                  2.*(L.gauss_sum_chi.conjugate())*L.l_value/t_om[0])
        
        #Compute alpha_chi array for their algebraic parts
        alp_chi = tw_alpha_chi(chi_cond, chi_label, alg_part_l, exp_chi_minus_N,\
                               L.exp_chi_c, eps_E, k)
        del(L, eps_E, E_p_l)
        
        #Compute A_chi array for the field norms of their alpha_chi's
        #Note: A_chi_div_g is not a numpy array but a sage list of Integer type
        #data to prevent overflow for the integer values of A_chi
        A_chi_div_g, g = tw_A_chi(k, chi_cond, chi_label, alp_chi)
        
        return cls(E, k, X, g, chi_cond, chi_label, A_chi_div_g, alg_part_l, \
                alp_chi, exp_sign_chi, exp_chi_minus_N)

    @classmethod
    def load_from_zip(cls, E_label, k, path, X=3):
        r"""
        The algebraic/integer `L`-values. They are stored as a zip file consisting 
        of two files: one .npz and .sobj. This sobj file contains the `A_\chi` values 
        since these may not be able to fit a 64 bits integer due to some large `k` 
        such as `k = 13`.
        """
        path += E_label+'/'
        os.makedirs(path, exist_ok=True)
        os_pwd= os.getcwd()
        os.chdir(path)
        
        f_name_zip = E_label+'_'+str(k)+'_'+str(X)+'m_alg_int_l_values.zip'
        f_name_npz = E_label+'_'+str(k)+'_'+str(X)+'m_alg_int_l_values.npz'
        f_name_sobj = E_label+'_'+str(k)+'_'+str(X)+'m_alg_int_l_values.sobj'

        with ZipFile(f_name_zip, 'r') as zf:
            zf.extractall()

        with np.load(f_name_npz) as data:
            E = EllipticCurve(data['info_twists'][0])
            k = data['info_twists'][1]
            X = data['info_twists'][2]
            g = data['info_twists'][3]
            chi_cond = data['chi_cond']
            chi_label = data['chi_label']
            alg_part_l = data['alg_part_l']
            alp_chi = data['alp_chi']
            exp_sign_chi = data['exp_sign_chi']
            exp_chi_minus_N = data['exp_chi_minus_N']
        
        A_chi_div_g = load(f_name_sobj)

        os.remove(f_name_npz)
        os.remove(f_name_sobj)
        os.chdir(os_pwd)
        
        return cls(E, k, X, g, chi_cond, chi_label, A_chi_div_g, alg_part_l, \
                alp_chi, exp_sign_chi, exp_chi_minus_N)

    def save_to_zip(self, path):
        path += self.E.label()+'/'
        os.makedirs(path, exist_ok=True)
        os_pwd= os.getcwd()
        os.chdir(path)
        
        f_name_zip = self.E.label()+'_'+str(self.k)+'_'+str(self.X)+'m_alg_int_l_values.zip'
        f_name_npz = self.E.label()+'_'+str(self.k)+'_'+str(self.X)+'m_alg_int_l_values.npz'
        f_name_sobj = self.E.label()+'_'+str(self.k)+'_'+str(self.X)+'m_alg_int_l_values.sobj'
        
        info_twists = np.array([self.E.label(), self.k, self.X, self.g])
        
        np.savez_compressed(f_name_npz, info_twists=info_twists, chi_cond=self.chi_cond,\
                            chi_label=self.chi_label, alg_part_l=self.alg_part_l, \
                            alp_chi=self.alp_chi, exp_sign_chi=self.exp_sign_chi,\
                            exp_chi_minus_N=self.exp_chi_minus_N)
        
        save(self.A_chi_div_g, path+f_name_sobj)
        
        with ZipFile(f_name_zip, 'w') as zf:
            zf.write(f_name_npz)
            zf.write(f_name_sobj)
            
        os.remove(f_name_npz)
        os.remove(f_name_sobj)
        os.chdir(os_pwd)