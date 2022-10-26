# twists
This repository contains the source codes for computing the central $L$-values of $L$-functions of elliptic curves twisted by a family of primitive Dirichlet characters and their algebraic parts and integer values. More precisely, it contains the sources codes of two main programs: 
- twists_clve
- twists_ailve

The information about the sample data produced by the programs is also provided. 

The source codes can be downloaded from the author's github site (https://github.com/jbnam/twists/) or Zenodo archive site (**TODO:id needed**)
The sample data can also be downloaded from Zenodo archive site (**TODO:id needed**)

The author: Jungbae Nam (aka JB)

For information about the author, visit his personal website (https://jbnam.github.io/). 

If you have a comment/question regarding this codes package and the sample data set, please feel free to contact me at x.y@concordia.ca or x.y@gmail.com where x and y are my first and last name respectively.

## 1) Introduction

Let $\mathcal B_k$ be the family of primitive Dirichlet characters of order $k$ and define

$$
\mathcal{B}_{k,N}(X) = \{\chi \in \mathcal B_k \mid \mathfrak{f}_\chi \leq X \text{ and } \text{gcd}(N, \mathfrak{f}_\chi)=1\}
$$

where $\text{gcd}$ is the greatest common divisor function and $\mathfrak{f}_\chi$ is the conductor of $\chi$. Moreover, denote $\zeta_k := e^{2 \pi i/k}$ and $Z_k := \big[0, \zeta_k, \zeta_k^2, \ldots, \zeta_k^{k-1}, 1\big]$ for a fixed $k$.

Let $E$ be an elliptic curve defined over $\mathbb{Q}$ of conductor $N$. We compute the values of $L(E, s, \chi)$ at $s = 1$ for $\chi \in \mathcal B_{k,N}(X)$ by the following well-known formula:

$$
L(E, 1, \chi) = \sum_{n \ge 1}(\chi(n) + w_E C_\chi\overline{\chi}(n))\frac{a_n}{n}\text{exp} (-2\pi n/\mathfrak f_\chi \sqrt{N} ) \qquad\qquad(1)
$$

where $a_n$ and $w_E$ are the coefficients and the root number of $L(E, s)$, respectively, and $C_{\chi} = \chi(N) \tau^2(\chi)/\mathfrak{f}_\chi$ where $\tau(\chi)$ is the Gauss sum of $\chi$. Here $\overline{\chi}$ is the complex conjugate of $\chi$ and $\text{exp}$ is the exponential function.

Note: The twists package uses the label of $E$ as the Cremona's elliptic curve label.

## 2) twists_clve

twists_clve is a command line program written in C/C++ and CUDA for computing and store the following number theoretic values: 

a linux compatible system with a NVIDIA graphics processing units (GPU). 

### System and Libraries Requirements

#### Hardwares and Operating System: 
- Any OS supported by the following compilers and libraries with memory capacity larger than 10 GB
- A graphics processing unit supporting CUDA driver ver. 11.4 or later and capability ver. 6.1 or later with global memory capacity larger than 10 GB  

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
1. Download the twists package and unzip it on your working directory.
2. Check the requirements above for your systems:
   - One can check his/her GPU hardware specifications by building and running "/Samples/1_Utilities/deviceQuery" of the CUDA samples package installed. Refer to deviceQuery_output.txt in the twists package as an example.
   - The Makefile is written under the assumptions that FLINT and GMP are installed as shared libraries.
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
   ? for(j=1,length(le),E=ellinit(le[j]);van=ellan(E,100000000);fraw=fileopen(concat(concat("./",le[j]),"an_100m.data"),"w");for(k=1,length(van),filewrite(fraw,van[k]););fileclose(fraw);kill(van);print(le[j]););
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
The output data consist of the tuples of following 13 entries:
 $$
 [ a ]
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
- $e_\chi(N)$ \- The index of $Z_k$ at which the value of $Z_k$ is $\chi(N)$
- $c$ \- The least positive integer with $\chi(c)$ is a primitive $k$-th root of unity
- $e_\chi(c)$ \- The index of $Z_k$ at which the value of $Z_k$ is $\chi(c)$
- $e_\chi(-1)$ \- The index of $Z_k$ at which the value of $Z_k$ is the sign of $\chi$
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

### Data Conversion and Sample Data Archived
**TODO** 
$X = 3\cdot 10^6$ for $k = 3, 5, 7, 13$ and $X = 10^6$ for $k = 6$ 
 so that the errors of its real and imaginary part are at most $10^{-10}$


The hardware systems for obtaining these sample data are 
- OS: Rocky Linux 8.6 with 64 bit support
- Memory: 32 GB
- CPU: Intel® Core™ i7-6700 CPU @ 3.40GHz × 8
- GPU: NVIDIA GeForce ® GTX 1080 Ti

Considering the cloud storage limit of Zenodo, the raw output data of twists_clve are converted into Python compatible data format and stored into Zenodo. Thus, one is recommended to use twists_ailve on SageMath to read these sample data in Zenodo.
**TODO: Archived has directory structure**

## 3) twists_ailve

twists_ailve is a SageMath command line program to convert the raw twists_clve data archived in Zenodo and compute the algebraic and integer $L$-values.

### Data File Naming Convensions Archived on Zenodo
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
4. Once a tw_central_l_values class object created, one can save it as a npz (Numpy compressed) file into a path. Then, the npz file is saved in
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
Note: For better loading procedure and storage saving, the tw_central_l_values and tw_alg_int_l_values classes use Numpy for each array element except the A_chi_div_g list of tw_alg_int_l_values class. It is because that the absolute value of an integer element in that list can easily be greater than the maximum allowed for a 64-bit integer (one can find those integer elements for $k = 13$ in the sample data).

### Python Data Converted from Raw Data Obtained by twists_clve 
The output data consist of the tuples of following 11 entries:
 $$
 [ N, k, \mathfrak{f}_\chi, r_{\chi}, L(E,1,\chi), \tau(\chi), e_\chi(N), c, e_\chi(c), e_\chi(-1), T_\chi ]
 $$
where

- $N$ \- The conductor of an elliptic curve $E$ defined over $\mathbb{Q}$
- $k$ \- The order of a primitive Dirichlet character $\chi$
- $\mathfrak{f}_\chi$ \- The conductor of $\chi$
- $r_\chi$ \- The label of $\chi$
- $L(E,1,\chi)$ \- The central value of $L(E, s, \chi)$
- $\tau(\chi)$ \- The Gauss sum of $\chi$
- $e_\chi(N)$ \- The index of $Z_k$ at which the value of $Z_k$ is $\chi(N)$
- $c$ \- The least positive integer with $\chi(c)$ is a primitive $k$-th root of unity
- $e_\chi(c)$ \- The index of $Z_k$ at which the value of $Z_k$ is $\chi(c)$
- $e_\chi(-1)$ \- The index of $Z_k$ at which the value of $Z_k$ is the sign of $\chi$
- $T_\chi$ \- The number of terms computed for the value $L(E,1,\chi)$ in Equation (1)

Note: Even though FLINT and GMP support arbitrary precision integer and float computations, the output float data are of double precision type at most due to the limited support of CUDA.

### Python Data of Algebraic and Integer $L$-values
**TODO WHY ZIP? Precision**

The output data consist of the tuples of following 11 entries:
 $$
 [ N, k, \mathfrak{f}_\chi, r_\chi, L(E,1,\chi), \tau(\chi), e_\chi(N), c, e_\chi(c), e_\chi(-1), T_\chi ]
 $$
where

- $N$ \- The conductor of an elliptic curve $E$ defined over $\mathbb{Q}$
- $k$ \- The order of a primitive Dirichlet character $\chi$
- $\mathfrak{f}_\chi$ \- The conductor of $\chi$
- $r_\chi$ \- The label of $\chi$
- $L(E,1,\chi)$ \- The central value of $L(E, s, \chi)$
- $\tau(\chi)$ \- The Gauss sum of $\chi$
- $e_\chi(N)$ \- The index of $Z_k$ at which the value of $Z_k$ is $\chi(N)$
- $c$ \- The least positive integer with $\chi(c)$ is a primitive $k$-th root of unity
- $e_\chi(c)$ \- The index of $Z_k$ at which the value of $Z_k$ is $\chi(c)$
- $e_\chi(-1)$ \- The index of $Z_k$ at which the value of $Z_k$ is the sign of $\chi$
- $T_\chi$ \- The number of terms computed for the value $L(E,1,\chi)$ in Equation (1)

### Sample Data Archived
$X = 3\cdot 10^6$ for $k = 3, 5, 7, 13$ and $X = 10^6$ for $k = 6$ 

**TODO: Archived has directory structure**
