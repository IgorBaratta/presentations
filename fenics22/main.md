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
    text-align: center;

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
  p{

    text-align: left;
    font-size: 0.8em;
    letter-spacing: 0px;

  }
  img[src$="centerme"] {
   font-size: 0.8em; 
   display:block; 
   margin: 0 auto; 
  }

---

# Guiding compilers to better autovectorize: <br> The road to portable finite element kernels

### Igor Baratta, Jørgen Dokken, Chris Richardson, Garth Wells

&nbsp; 

![width:300px](Figures/cambridge_logo.png?style=centerme)

---

## Helmholtz equation in UFL

<!-- UFL is an embedded domain-specific language for the description of the weak form and discretized function spaces of finite element problems. It is embedded in Python. -->
<font size="6">

```python
from ufl import *

# An abstract mesh of hexahedral cells
mesh = Mesh(VectorElement("Lagrange", hexahedron, 2))

# 6th order element to reduce pollution error
element = FiniteElement("Lagrange", hexahedron, 6)
V = FunctionSpace(mesh, element)

# Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Bilinear form
a = inner(grad(u), grad(v))*dx - k2*inner(u, v)

# Compute the action of the form on a coefficient
un = Coefficient(V)
L = action(a, un)
```

</font>

---

## Form compiler design

FFCx takes a form expressed in UFL and produces low-level code that assembles the form on a single cell using 5 sequential "compiler passes":

![width:800px](Figures/ffcx_diagram.png?style=centerme)

<!-- A Compiler passes here can be understood as a series of expression
transformations  -->
<!-- by using optimization techniques that are not
readily applicable if the code is developed by hand -->

---

## Stage 1 -  Analysis

This stage preprocesses the UFL form and extracts form metadata, such as elements, coefficients, and the cell type. 

<!-- One can have multiple elements in a single form. Multiple coefficients, but at the moment each kernel supports a single cell type -->

It may also perform simplifications on the form, for example:
<br>
$$
\int \nabla u \cdot \nabla v ~\mathrm{d}x
$$
becomes
$$
\int J^{-T}\nabla \tilde{u} \cdot J^{-T} \nabla \tilde{v} |J| ~\mathrm{d}X
$$

<!-- This involves scaling the integral and the application of pullbacks on the functions (arguments and coefficients) of the form. -->

<!-- Note the change in the integral measure (dx versus dX). -->

<!-- the Jacobian is replaced with the gradient of the spatial coordinates, determinants are expanded, divergence and curl are expressed with tensor algebra on gradients, and various products are expanded using the index notation. -->

---
## Stage 2 -  Code representation

---
## Stage 3 - IR Optimizations
This stage examines the intermediate representation and performs optimizations.
