---
marp: true
class: lead
paginate: true
math: katex
theme: uncover
style: |

  section {
    background-color: #ccc;
    letter-spacing: 1px;
    text-align: left;
  }
  h1 {
    font-size: 1.3em;
    text-align: center;
  }
  h2 {
    font-size: 1.5em;
    text-align: left;
  }
  h3 {
    font-size: 1em;
    text-align: center;
    font-weight: normal;
    letter-spacing: 1px;
  }
  h6 {
    text-align: center;
    font-weight: normal;
    letter-spacing: 1px;
  }
  p{
    text-align: left;
    font-size: 0.75em;
    letter-spacing: 0px;
  }
  img[src$="centerme"] {
   font-size: 0.8em; 
   display:block; 
   margin: 0 auto; 
  }
  footer{
    color: black;
    text-align: left;
  }
  ul {
    padding: 10;
    margin: 0;
  }
  ul li {
    color: black;
    margin: 5px;
    font-size:30px
  }
  /* Code */
  pre, code, tt {
    font-size: 0.98em;
    font-size: 20px;
    font-family: Consolas, Courier, Monospace;
  }

  code, tt {
    margin: 0px;
    padding: 2px;
    white-space: nowrap;
    border: 1px solid #eaeaea;
    border-radius: 3px;
  }

  pre {
    background-color: #f8f8f8;
    overflow: auto;
    padding: 6px 10px;
    border-radius: 3px;
  }

  pre code, pre tt {
    background-color: transparent;
    border: none;
    margin: 0;
    padding: 0;
    white-space: pre;
    border: none;
    background: transparent;
  }



---

# Guiding compilers to better autovectorize: <br> The road to portable finite element kernels

### Igor Baratta, Jørgen Dokken, Chris Richardson, Garth Wells

<br>

![width:300px](Figures/cambridge_logo.png?style=centerme)

<br>

###### ia397@cam.ac.uk

---
## Helmholtz equation in UFL

<font size="5.6px">

```python
from ufl import *

# An abstract mesh of hexahedral cells
mesh = Mesh(VectorElement("Lagrange", tetrahedron, 1))

# 6th order element to reduce pollution error
element = FiniteElement("Lagrange", tetrahedron, 6)
V = FunctionSpace(mesh, element)

# Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Bilinear form
a = inner(grad(u), grad(v))*dx - k2*inner(u, v)*dx

# Compute the action of the form on a coefficient
un = Coefficient(V)
L = action(a, un)
```

</font>

<!-- _footer: "1. Alnæs, Martin S., et al. 'Unified form language: A domain-specific language for weak formulations of partial differential equations.' ACM Transactions on Mathematical Software (TOMS) 40.2 (2014): 1-37 " -->

<!-- In Dolfinx, the users define the problem in Unified Form Language (UFL) (Alnæs et al., 2014) which captures the weak form and the function space discretization, then the form compiler takes this high-level expression and generates efficient low code.  -->

<!-- UFL is an embedded domain-specific language for the description of the weak form and discretized function spaces of finite element problems. It is embedded in Python. -->

---

## Assemble Matrix
![width:800px](Figures/matrix.png?style=centerme)

---
## Compute Action
![width:800px](Figures/action.png?style=centerme)


---
## Form compiler design

FFCx takes a form expressed in UFL and produces low-level code that assembles the form on a single cell using 5 sequential "compiler passes":

![width:800px](Figures/ffcx_diagram.png?style=centerme)

<!-- _footer: "2. https://github.com/FEniCS/ffcx/" -->

<!-- A Compiler passes here can be understood as a series of expression
transformations, and it allows us to appply some optimization techniques that are not readily applicable if the code is developed by hand or otherwise it would be a boring task.  -->


---

## Stage 1 -  Analysis

This stage preprocesses the UFL form and extracts form metadata, such as elements, coefficients, and the cell type. 
It also involves scaling the integral and the application of pullbacks:
<br>
$$
\int_K \nabla u \cdot \nabla v ~\mathrm{d}x - \int_K k^2 u \cdot v ~\mathrm{d}x
$$
becomes
$$
\int_{\tilde{K}} J^{-T}\nabla \tilde{u} \cdot J^{-T} \nabla \tilde{v}~|J| ~\mathrm{d}X + \int_{\tilde{K}} \tilde{u} \cdot \tilde{v} ~ |J| ~\mathrm{d}X
$$
<br>

A later pass may replace $J$ with the evaluation of $\nabla x$.

<!-- This involves scaling the integral and the application of pullbacks on the functions (arguments and coefficients) of the form. -->

<!-- Note the change in the integral measure (dx versus dX). -->

<!-- the Jacobian is replaced with the gradient of the spatial coordinates, determinants are expanded, divergence and curl are expressed with tensor algebra on gradients, and various products are expanded using the index notation. -->

---

## Stage 2 -  Code representation

This stage includes generation of finite element basis functions, extraction of data for mapping of degrees of freedom. When the basis function and quadrature rule have a tensor-product structure, only 1d basis functions and its derivatives are generated:
$$

    \phi_{i}\left(\mathbf{x}_q\right) = \psi_{i_0}\left(x_{q_0}\right) \cdot \psi_{i_1}\left(x_{q_1}\right)

$$
$$

    D_x\phi_{i}\left(\mathbf{x}_q\right) = \psi'_{i_0}\left(x_{q_0}\right) \cdot \psi_{i_1}\left(x_{q_1}\right)

$$
$$

    D_y\phi_{i}\left(\mathbf{x}_q\right) = \psi_{i_0}\left(x_{q_0}\right) \cdot \psi'_{i_1}\left(x_{q_1}\right)

$$

<!-- _footer: "3. Scroggs, M. W., Baratta, I. A., Richardson, C. N., & Wells, G. N. (2022). Basix: a runtime finite element basis evaluation library. Journal of Open Source Software, 7(73), 3982." -->
---

## Stage 3 - IR Optimizations

This stage examines the intermediate representation and performs optimizations.

- In FFC the goal was to reduce the number of operations.
- In FFCx the goal is to increase throughput through a better use of the computing architecture.

<br>

Modern computer architectures have become more complicated and not all flop reduction techniques improve throughput.

---
## What is vectorization?
```c++
for (int i = 0; i < n; i++)
    c[i] = a[i] + b[i];
```
 
![width:800px](Figures/vectorization.png?style=centerme)

- Naïve implementation usually achieves <10% peak performance.
- AVX2 introduced in 2013 and AVX512  in 2016.

---
## Vectorization efficiency trend 



![width:566px](Figures/efficiency.png?style=centerme)

- GCC 10 released in 2018

---

### Loop invariant code motion 
```c++
for (int i = 0; i < 8; ++i)
  for (int j = 0; j < 8; ++j)
    A[i][j] += fw[iq] * phi[iq][i] * phi[iq][j];
```
<center> ⬇ </center>

```c++
for (int i = 0; i < 8; ++i)
  double ti = fw[iq] * phi[iq][i];
  for (int j = 0; j < 8; ++j)
    A[i][j] += phi[iq][j] * ti;
```
<center> ⬇ </center>

```c++
for (int i = 0; i < 8; ++i)
  t0[i] = fw[iq] * phi[iq][i];
for (int i = 0; i < 8; ++i)
  for (int j = 0; j < 8; ++j)
    A[i][j] += phi[iq][j] * t0[i];
```



<!-- the calculation of fw[iq] * phi[iq][i] is independent of the inner loop, thus it need not be repeated for all j.  -->

---
### Eliminate operations on zeros
```c++
    static const double phi[4][6] =
        { { 0.0, 0.2, 0.0, 0.2, 0.2, 0.4 },
          { 0.0, 0.5, 0.0, 0.5, 0.0, 0.0 },
          { 0.0, 0.1, 0.0, 0.7, 0.1, 0.1 },
          { 0.0, 0.7, 0.0, 0.2, 0.1, 0.1 } };
```

```c++
double w[iq] = {0};
for (int iq = 0; iq < 4; ++iq)
  for (int id = 0; id < 6; ++id)
    w[iq] += phi[iq][j] * u[id];
```
<center> ⬇ </center>

```c++
for (int id = 1; id < 2; ++id)
  w[iq] += phi[iq][j] * u[id];
for (int id = 3; id < 4; ++id)
  w[iq] += phi[iq][id] * u[id];
```

- 33% flop reduction, however it prevents other optimizations.

---
### Loop fusion
<!-- merge a sequence of loops into one loop -->

```c++
for (int iq = 0; iq < Nq; ++iq){
  for (int id = 0; id < Nd; ++id)
    w_0[iq] += dphi_x[iq][j] * u[id];
  for (int id = 0; id < Nd; ++id)
    w_1[iq] += dphi_y[iq][j] * u[id];
}
```
<center> ⬇ </center>

```c++
for (int iq = 0; iq < Nq; ++iq){
  for (int id = 0; id < Nd; ++id){
    w_0[iq] += dphi_x[iq][j] * u[id];
    w_1[iq] += dphi_y[iq][j] * u[id];
  }
}
```
- Reduces loop control overhead
- Improves data locality

---
### Tensor contractions as matrix matrix multiplication

$$
C_{abc} = \sum_k A_{ka} B_{kbc}
$$

Naive code:

```c++
for (int a = 0; a < na; a++)
  for (int b = 0; b < nb; b++)
    for (int c = 0; c < nc; c++)
      for (int k = 0; k < nk; k++)
        C[a][b][c] += A[k][a] * B[k][b][c];
```

- $n_a, n_b, n_c, n_k \approx P$
- Compiler: vectorization not profitable!

---
### Tensor contractions as matrix matrix multiplication

$$
C[a,\{b,c\}] = A[a, k] B[k, \{b,c\}]
$$

$$
d = \{b,c\}
$$

```c++
for (int a = 0; a < na; a++)
  for (int k = 0; k < nk; k++)
    for (int d = 0; d < nb*nd; d++)
        C[a][d] += A[k][a] * B[k][d];
```

- $n_d \approx P^2$, increase inner loop range
- Improves access pattern (unit stride access pattern)

---

## Stage 2 Code Generation: <br> Results
---
### Tetrahedral mesh

![width:1000px](Figures/tetrahedron.png?style=centerme)

---

### Hexahedral mesh

![width:1000px](Figures/hex.png?style=centerme)

---

## Conclusions and Outlook

- Modern compilers don't do witchcraft:
  - we need to write sensible (simple) code to get sensible (high) performance,
- we can exceed 40% of theoretical peak performance,  with portable code.
- Road to GPUs
  - performance model takes different architectures into account
  - but different optimizations might be required (eg.: bank conflicts, coalesced memory access)

<br>

https://github.com/IgorBaratta/local_operator
