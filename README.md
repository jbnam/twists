# twists
This repository contains the source codes for computing the central $L$-values of $L$-functions of elliptic curves twisted by a family of primitive Dirichlet characters and their algebraic parts and integer values. More precisely, it contains the sources codes of two main programs: 
- twists_clve
- twists_ailve

The information about the sample data produced by the programs is also provided. 

The author: Jungbae Nam (aka JB)

For information about the author, visit his personal website (https://jbnam.github.io/). 

If you have a comment/question regarding this codes package and the sample data set, please feel free to contact me at x.y@gmail.com where x and y are my first and last name respectively.

## Download Links for Source Codes and Sample Data
- Source codes links: The author's GitHub site (https://github.com/jbnam/twists/) or The author's Zenodo archive site [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7255396.svg)](https://doi.org/10.5281/zenodo.7255396)
- Sample data link: The author's Zenodo archive site [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7255396.svg)](https://doi.org/10.5281/zenodo.7255396)


## 1) Introduction

Let $\mathcal B_k$ be the family of primitive Dirichlet characters of order $k$ and define

$$
\mathcal B_{k,N}(X) = \lbrace \chi \in \mathcal B_k \mid \mathfrak f_{\chi} \leq X \text{ and } \text{gcd} ( N, \mathfrak{f}_\chi )=1 \rbrace
$$

where $\text{gcd}$ is the greatest common divisor function and $\mathfrak{f}_\chi$ is the conductor of $\chi$. Moreover, denote $\zeta_k := e^{2 \pi i/k}$ and $Z_k := \big[0, \zeta_k, \zeta_k^2, \ldots, \zeta_k^{k-1}, 1\big]$ for a fixed $k$.

Let $E$ be an elliptic curve defined over $\mathbb{Q}$ of conductor $N$. Then, the $L$-function of an elliptic curve $E$ twisted by $\chi$ is defined by the following Dirichlet series for $\text{Re}(s) > 3/2$:

$$
L(E, s, \chi) = \sum_{n \ge 1}\frac{\chi(n)a_n}{n^s} = \prod_{p \nmid N}\Big(1 - \frac{\chi(p)a_p}{p^s} + \frac{\chi^2(p)}{p^{2s-1}}\Big)^{-1}\prod_{p \mid N}\Big(1 - \frac{\chi(p)a_p}{p^s}\Big)^{-1}
$$

where $a_p$ is the traces of Frobenius of $E$ if $p \nmid N$ and $0, \pm 1$ depending on the reduction type of $E$ modulo $p$ otherwise. It is well-known that it can be analytically continued to $\mathbb{C}$ and satisfies some functional equation which relates $s$ to $2-s$, so that the critical strip is $\{s \in \mathbb{C} \mid 1/2 < \text{Re}(s) < 3/2\}$.

We can compute the values of $L(E, s, \chi)$ at $s = 1$ for $\chi \in \mathcal B_{k,N}(X)$ by the following well-known formula:

$$
L(E, 1, \chi) = \sum_{n \ge 1}(\chi(n) + w_E C_\chi\overline{\chi}(n))\frac{a_n}{n}\text{exp} (-2\pi n/(\mathfrak f_\chi \sqrt{N}) ) \qquad\qquad(1)
$$

where $a_n$ and $w_E$ are the coefficients and the root number of $L(E, s)$, respectively, and $C_{\chi} = \chi(N) \tau^2(\chi)/\mathfrak{f}_\chi$ where $\tau(\chi)$ is the Gauss sum of $\chi$. Here $\overline{\chi}$ is the complex conjugate of $\chi$ and $\text{exp}$ is the exponential function.

The algebraic part of $L(E,1,\chi)$ is defined as 

$$
L_E^{\text{alg}}(\chi) = \frac{2\tau(\overline{\chi})}{\Omega_\chi}L(E,1,\chi)
$$ 

where  $\tau(\chi)$ is the Gauss sum of $\chi$ and $\Omega_\chi = \Omega^{\pm}$ is a period of $E$ depending on the signs of $\chi$. It is known that the algebraic part is an algebraic integer in the cyclotomic field $\mathbb{Q}(\chi)$ adjoining with the values of $\chi$.

Denote the maximal real subfield of  $\mathbb{Q}(\chi)$ and its ring of integers by $\mathbb{Q}^+(\chi)$ and $\mathcal O_\chi^+$, respectively. Then, from Proposition 2.1 in [[1]](#reference), for each $L_E^\text{alg}(\chi)$, we can find a real cyclotomic integer $\alpha_\chi \in \mathcal O_\chi^+$ satisfying $\sigma(\alpha_\chi) = \alpha_\chi^\sigma$ for all $\sigma \in \text{G}$, the Galois group of $\mathbb{Q}(\chi)/\mathbb{Q}$. Lastly, denote 

$$A_{\chi} = \text{Nm}_ {\mathbb{Q}^+(\chi)/\mathbb{Q}}(\alpha_ \chi)\in\mathbb{Z}$$

where $\text{Nm}_{\mathbb{Q}^+(\chi)/\mathbb{Q}}$ is the field norm from $\mathbb Q^+(\chi)$ to $\mathbb Q$.

Notes: 
- The twists package uses the label of $E$ as Cremona's elliptic curve label.
- In computing $L_E^{\text{alg}}(\chi)$, the period lattice $\Omega^\pm$ is computed such that $\Omega^+ \in \mathbb R$ and $\Omega^- \in \mathbb R i$.  

[1] [Hershy Kisilevsky](https://www.concordia.ca/artsci/math-stats/faculty.html?fpid=hershy-kisilevsky) and [Jungbae Nam](https://jbnam.github.io/). *Non-Zero Central Values of Dirichlet Twists of Elliptic L-Functions*, Journal of Number Theory, 266:166-194, 2025 ([Preprint](https://arxiv.org/abs/2001.03547))

## 2) Data Conversion and Sample Data Archived

Considering the cloud storage limit of Zenodo, the raw output data obtained by twists_clve are converted into Python-compatible data format using Numpy and stored in Zenodo. Thus, one is recommended to use twists_ailve on SageMath to read these sample data in Zenodo. The sample data can be downloaded from the author's Zenodo dataset archive.

The hardware systems for obtaining these sample data are 
- OS: Rocky Linux 8.6 with 64-bit support
- Memory: 32 GB
- CPU: Intel® Core™ i7-6700 CPU @ 3.40GHz × 8
- GPU: NVIDIA GeForce ® GTX 1080 Ti

For the sample data in Zenodo, we compute $L(E, 1, \chi)$ with a massive amount of precomputed $a_n$'s (as will be mentioned below in more detail) so that the errors of its real and imaginary part are at most $10^{-10}$. It implies that the values of $\alpha_\chi \in \mathbb R$ have at least the correct first 4 digits.

Data file naming conventions for the sample data are
- Raw data file generated by twists_clve: E_k_X_raw.dat
  where
  - E \- The Cremona label of $E$
  - k \- The order of $\chi \in \mathcal{B}_{k,N}(X)$
  - X \- 3m and 1m for $k = 3, 5, 7, 13$ and $k = 6$, respectively
- Python data file equivalently converted from the raw one: E_k_X_central_l_values.npz  
  where E, k, and X are same as above.
- Python data file for algebraic and integer $L$-values: E_k_X_alg_int_l_values.zip  
  where E, k, and X are same as above.

Note: If a data file contains X = 3m (or 1m) in its name, it means the data contains the $L$-values for $\chi$ of conductor less than or equal to $3\cdot 10^6$ (or $10^6$, respectively).

In Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7255396.svg)](https://doi.org/10.5281/zenodo.7255396):

One can find two zip files: twists_clve_data.zip (17 GB) and twists_ailve_data.zip (13 GB). twists_clve_data.zip and twists_ailve_data.zip contain the E_k_X_central_l_values.npz files and E_k_X_alg_int_l_values.zip files, respectively, with some natural directory structure for k = $3, 5, 6, 7, 13$ and the following elliptic curves E:  
```
11a1, 14a1, 15a1, 17a1, 19a1, 20a1, 21a1, 24a1, 26a1, 26b1, 27a1, 30a1, 32a1, 33a1, 34a1, 35a1, 
36a1, 37a1, 37b1, 38a1, 38b1, 39a1, 40a1, 42a1, 43a1, 44a1, 45a1, 46a1, 48a1, 49a1, 50a1, 50b1, 
51a1, 52a1, 53a1, 54a1, 54b1, 55a1, 56a1, 57a1, 57b1, 57c1, 58a1, 58b1, 61a1, 62a1, 63a1, 64a1, 
65a1, 66a1, 66b1, 66c1, 67a1, 69a1, 70a1, 72a1, 73a1, 75a1, 75b1, 75c1, 76a1, 77a1, 77b1, 77c1, 
78a1, 79a1, 80a1, 80b1, 82a1, 83a1, 84a1, 84b1, 85a1, 88a1, 89a1, 89b1, 90a1, 90b1, 90c1, 91a1, 
91b1, 92a1, 92b1, 94a1, 96a1, 96b1, 98a1, 99a1, 99b1, 99c1, 99d1
```

One can unzip them on your local system and read the data directly with twists_ailve on SageMath.

## 3) twists_clve

twists_clve is a command line program written in C/C++ and CUDA for computing and storing the values of $L(E, 1, \chi)$ for $\mathcal B_{k,N}(X)$ and some other number theoretic values related with them. For a fixed $E$ and $k$, when $X$ gets large, the computations for obtaining the values of $L(E, 1, \chi)$ demand massive computational power. 

Interestingly, one of the ways to achieve this goal is to use General Purpose Graphic Processing Units (GPGPU). CUDA is one of those in the present time. For a practical example, for $X$ is a couple of millions and CUDA GPU with around 3000 cores, the total computational time can be reduced by a couple of thousand times faster than using one core of CPU.

### System and Libraries Requirements

#### Hardware and Operating System: 
- Any OS supported by the following compilers and libraries with a memory capacity larger than 10 GB
- A graphics processing unit supporting CUDA driver ver. 11.4 or later and capability ver. 6.1 or later with a global memory capacity larger than 10 GB  

#### Compilers for Building: 
- gcc(the GNU Compiler Collection) ver. 2.96 or later (https://gcc.gnu.org/)
- nvcc(Nvidia CUDA Compiler) ver. 11.4 or later (https://developer.nvidia.com/)

#### External Libraries: 
- FLINT(Fast Library for Number Theory) ver. 2.0 or later (https://flintlib.org/)
- GMP(The GNU Multiple Precision Arithmetic Library) ver. 5.1.1 or later (https://gmplib.org/)

#### Other Tools:
- Pari/GP ver. 2 or later (https://pari.math.u-bordeaux.fr/)
- CUDA samples ver. 11.4 or later (https://github.com/NVIDIA/cuda-samples/)
  
Note: For more detailed requirements of compilers and external libraries above, consult their websites.

### Instructions for Configuring and Building twists_clve
1. Download the twists package and unzip it in your working directory.
2. Check the requirements above for your systems:
   - One can check his/her GPU hardware specifications by building and running "/Samples/1_Utilities/deviceQuery" of the CUDA samples package installed. Refer to deviceQuery_output.txt in the twists package as an example.
   - The Makefile is written under the assumption that FLINT and GMP are installed as shared libraries.
   - Make sure that helper_cuda.h and helper_string.h, originally located under the directory of /common/ of the CUDA samples package, can be found in an implementation-defined directory by nvcc.
4. Run Makefile in twists_clve directory by "make" or "make all".
   
### Instructions for Running twists_clve
1. Compute and save the coefficients $a_n$ starting from $n = 1$ of $L(E, s, \chi)$ using Pari/GP as shown the following example GP code: for $E$: 11a1 and $n \le 10^6$ as an example,
   ```
   ? default(parisizemax,"20000M")
   ***   Warning: new maximum stack size = 20000002048 (19073.488 Mbytes).
   ?default(primelimit, "200000000")
   ? le = ["11a1"]
   %2 = ["11a1"]
   ? {for(j=1,length(le),E=ellinit(le[j]);van=ellan(E,10^8);
       fraw=fileopen(concat(concat("./",le[j]),"an_100m.data"),"w");
       for(k=1,length(van),filewrite(fraw,van[k]););
        fileclose(fraw);
       kill(van);
       print(le[j]);
     );}
   *** ellan: Warning: increasing stack size to 16000000.
   *** ellan: Warning: increasing stack size to 32000000.
   *** ellan: Warning: increasing stack size to 64000000.
   *** ellan: Warning: increasing stack size to 128000000.
   *** ellan: Warning: increasing stack size to 256000000.
   *** ellan: Warning: increasing stack size to 512000000.
   *** ellan: Warning: increasing stack size to 1024000000.
   11a1
   ?
   ```
   Note: For computing $L(E, 1, \chi)$ within a desired precision, firstly one needs to compute the number of $a_n$'s depending on $N$ and $\mathfrak{f}_\chi$. The formula to compute it can be easily derived from Equation (1) and can be found in the definition of function twists_num_terms in source/twists_dirichlet_character/twists_dirichlet_character.c.
2. Assume that with the data "A" of $a_n$ for $E$ of conductor "N" and root number "W" one wants to compute $L(E, 1, \chi)$ for the primitive Dirichlet characters of order "K" and conductor between "C1" and "C2" and save those in the output data "L". 
   
   Then, run twists_clve with the following arguments as: twists_clve N W K C1 C2 A L

   For $E$: 11a1 as an example, 
   ```
   [@twists_clve]$ ./twists_clve 11 1 3 2 10000 ./an.data ./output.dat
   The length of a_n is 100000001
   a_n are loaded into device!
   Twists for N = 11, k = 3, kinv = 0.33333333333333331, zeta_k = -0.50000000000000022 + 0.86602540378443849*I
   Done for f < 99. tw.num_chi: 32
   Done for f < 198. tw.num_chi: 62
       :
       :
   Done for f < 9999. tw.num_chi: 3184
   The computations are all done. Max stream idx = 31
   cpu = 5547 ms wall = 5566 ms
   [@twists_clve]$
   ```

### Output Data
The output data consist of the tuples of the following 13 entries:

 $$
 [ N, k, \mathfrak f_\chi, r_\chi, \Re(L), \Im(L), \Re(\tau(\chi)), \Im(\tau(\chi)), e_\chi(N), c, e_\chi(c), e_\chi(-1), T_\chi ]
 $$

where
- $N$ \- The conductor of an elliptic curve $E$ defined over $\mathbb{Q}$
- $k$ \- The order of a primitive Dirichlet character $\chi$
- $\mathfrak{f}_\chi$ \- The conductor of $\chi$
- $r_\chi$ \- The label of $\chi$
- $\Re(L)$ \- The real part of $L(E, 1, \chi)$
- $\Im(L)$ \- The imaginary part of $L(E, 1, \chi)$
- $\Re(\tau(\chi))$ \- The real part of the Gauss sum of $\chi$
- $\Im(\tau(\chi))$ \- The imaginary part of the Gauss sum of $\chi$
- $e_\chi(N)$ \- Index of $Z_k$ at which the value of $Z_k$ is $\chi(N)$
- $c$ \- The least positive integer with $\chi(c)$ is a primitive $k$-th root of unity
- $e_\chi(c)$ \- Index of $Z_k$ at which the value of $Z_k$ is $\chi(c)$
- $e_\chi(-1)$ \- Index of $Z_k$ at which the value of $Z_k$ is the sign of $\chi$
- $T_\chi$ \- The number of terms computed for the value $L(E,1,\chi)$ in Equation (1)

Note: Even though FLINT and GMP support arbitrary precision integer and float computations, the output float data are of double precision type at most due to the limited support of CUDA.

For $E$: 11a1 as an example,
```
[@twists_clve]$ cat output.dat 
11, 3, 7, 1, 1.9971068270600856, 1.3284392937855753, 2.3704694055761992, -1.1751062918847859, 1, 2, 2, 3, 6825
11, 3, 7, 2, 1.9971068270600865, -1.3284392937855733, 2.3704694055761992, 1.1751062918847872, 2, 2, 1, 3, 6825
    :
    :
11, 3, 9997, 3, 0.84765090208809601, -9.1002858592966529, 9.273015315297048, -99.554061629637928, 3, 2, 1, 3, 461435
11, 3, 9997, 4, 3.96079186956215e-14, -2.7321877339158544e-14, 58.849069303070934, 80.831844233314186, 1, 7, 1, 3, 461435
[@twists_clve]$
```

## 4) twists_ailve

twists_ailve is a SageMath command line program to convert the raw twists_clve data archived in Zenodo and compute the algebraic and integer $L$-values.

### Requirements
SageMath ver. 9.0 or later (https://www.sagemath.org/)

### Instructions for Running twists_ailve
1. Open SageMath command line (Optional): any interactive SageMath shell will work.
2. Load twists_ailve.sage on SageMath by typing: load('./twists_ailve.sage') for example.
3. From a E_k_X_raw.dat, computed by twists_clve, one can create a tw_central_l_values Python class object as
   ```
   sage: load('./twists_ailve.sage')
   sage: L = tw_central_l_values.load_from_dat('11a1', 3, './'); print(L[0])
   [11, 3, 7, 1, (1.9971068270600854+1.3284392937855751j), (2.3704694055761992-1.1751062918847859j), 1, 2, 2, 3, 6825]
   ```
4. Once a tw_central_l_values class object is created, one can save it as a npz (Numpy compressed) file into a path. Then, the npz file is saved in
   ```
   sage: L.save_to_npz('./')
   ```
5. Similar to Step 3. one can also create a tw_central_l_values Python class object from a E_k_X_central_l_values.npz as below:
   ```
   sage: L = tw_central_l_values.load_from_npz('11a1', 3, './'); print(L[0])
   [11, 3, 7, 1, (1.9971068270600854+1.3284392937855751j), (2.3704694055761992-1.1751062918847859j), 1, 2, 2, 3, 6825]
   ```
6. Compute the algebraic parts and integer values of $L(E, 1, \chi)$ from a E_k_X_central_l_values.npz as below:
   ```
   sage: %time A = tw_alg_int_l_values.load_from_central_l_values_npz('11a1',3,'./')
   CPU times: user 28.8 s, sys: 31.7 ms, total: 28.9 s
   Wall time: 28.9 s
   sage: print(A[0])
   [11, 3, 10, 7, 1, 1, (4.999999999999993+8.660254037844375j), 9.999999999999988, 3, 1]
   ```
   Note: It takes significantly more time as k increases for each E and X.
7. Save the algebraic parts and integer values as a zip file into a path as below:
   ```
   sage: A.save_to_zip('./')
   sage:
   ```
8. One can also load the tw_alg_int_l_values class object from a E_k_X_alg_int_l_values.zip as below:
   ```
   sage: A = tw_alg_int_l_values.load_from_zip('11a1', 3, './'); print(A[0])
   [11, 3, 10, 7, 1, 1, (4.999999999999993+8.660254037844375j), 9.999999999999988, 3, 1]
   ```
### Output Data
The output data consist of the tuples of the following 10 entries:

 $$
 [ N, k, g, \mathfrak{f}_ \chi, r_ \chi, A_ \chi/g, L_E^\text{alg}(\chi), \alpha_ \chi, e_\chi(-1), e_\chi(-N)]
 $$

where
- $N$ \- The conductor of an elliptic curve $E$ defined over $\mathbb{Q}$
- $k$ \- The order of a primitive Dirichlet character $\chi$
- $g$ \- The greatest common divisor of the values of $A_\chi$ in this data
- $\mathfrak{f}_\chi$ \- The conductor of $\chi$
- $r_\chi$ \- The label of $\chi$
- $A_\chi/g$ \- $\text{Nm}_{\mathbb{Q}^+(\chi)/\mathbb{Q}}(\alpha _{\chi})$ divided by $g$
- $L_E^\text{alg}(\chi)$ \- The algebraic part of $L(E, 1, \chi)$ defined above
- $\alpha_\chi$ \- Real number defined in Proposition 2.1 of [[1]](#reference) 
- $e_\chi(-1)$ \- Index of $Z_k$ at which the value of $Z_k$ is the sign of $\chi$
- $e_\chi(-N)$ \- Index of $Z_k$ at which the value of $Z_k$ is $\chi(-N)$
  
For $E$: 11a1, $k = 3$, and $\mathfrak{f}_\chi = 7$, as an example,
   ```
   sage: A = tw_alg_int_l_values.load_from_zip('11a1', 3, './'); print(A[0])
   [11, 3, 10, 7, 1, 1, (4.999999999999993+8.660254037844375j), 9.999999999999988, 3, 1]
   ```
Note: For better loading procedure and storage saving, the tw_central_l_values and tw_alg_int_l_values classes use Numpy for each array element except the A_chi_div_g list of tw_alg_int_l_values class. It is because the absolute value of an integer element in that list can easily be greater than the maximum allowed for a 64-bit integer (one can find those integer elements for $k = 13$ in the sample data).

### Class object tw_central_l_values for E_k_X_central_l_values.npz
Class tw_central_l_values consists of the following members:  
- E - The elliptic curve associated with E_k_X_central_l_values.npz: : SageMath EllipticCurve_rational_field_with_category class
- k - The order of the family of characters associated with E_k_X_central_l_values.npz: Python int
- num_twists - The cardinality of this family: Python int
- X - The maximum conductor in this family: 3 and 1 for $k = 3, 5, 7, 13$ and $k = 6$, respectively: Python int
- chi_cond - Numpy array of $\mathfrak f_\chi$'s: dtype=uint32
- chi_label - Numpy array of $r_\chi$'s: dtype=uint32
- l_value - Numpy array of $L(E, 1, \chi)$'s: dtype=cdouble
- gauss_sum_chi - Numpy array of $\tau(\chi)$'s: dtype=cdouble
- exp_chi_N - Numpy array of $e_\chi(N)$'s: dtype=uint16
- c - Numpy array of $c$'s: dtype=uint32
- exp_chi_c - Numpy array of $e_\chi(c)$'s: dtype=uint16
- exp_sign_chi - Numpy array of $e_\chi(-1)$'s: dtype=uint16
- num_terms - Numpy array of $T_\chi$'s: dtype=uint64

### Class object  of Algebraic and Integer $L$-values for E_k_X_alg_int_l_values.zip
Class tw_alg_int_l_values consists of the following members:  
- E - The elliptic curve associated with E_k_X_central_l_values.npz: SageMath EllipticCurve_rational_field_with_category class
- k - The order of the family of characters associated with E_k_X_central_l_values.npz: Python int
- num_twists - The cardinality of this family: Python int
- X - The maximum conductor in this family: 3 and 1 for $k = 3, 5, 7, 13$ and $k = 6$, respectively: Python int
- g - the greatest common divisor of all $A_\chi$'s in $\mathcal{B}_{k,N}(X)$: Python int
- chi_cond - Numpy array of $\mathfrak f_\chi$'s: dtype=uint32
- chi_label - Numpy array of $r_\chi$'s: dtype=uint32
- A_chi_div_g - Numpy array of $A_\chi / g$'s Python int
- alg_part_l - Numpy array of $L_E^{\text{alg}}(\chi)$'s: dtype=cdouble
- alp_chi - Numpy array of $\alpha_\chi$'s: dtype=double
- exp_sign_chi - Numpy array of $e_\chi(-1)$'s: dtype=uint16
- exp_chi_minus_N - Numpy array of $e_\chi(-N)$'s: dtype=uint16
