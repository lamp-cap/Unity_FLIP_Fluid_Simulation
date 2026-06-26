# Unity_FLIP_Fluid_Simulation
FLIP fluid simulation with MGPCG solver in unity

### Assets/GPU_FLIP
A Incompressible Flow without viscosity using FLIP written in HLSL

4M particles on 256x128x128 grid, simulated on GPU

The grid is solved with a UAAMGPCG solver, but only one iteration

Implemented the algorithm from the paper **A Fast Unsmoothed Aggregation Algebraic Multigrid Framework for the Large-Scale Simulation of Incompressible Flow** <https://computationalsciences.org/publications/shao-2022-multigrid/shao-2022-multigrid.pdf> as a preconditioner for the conjugate gradient method

**Warning: it was only tested on my RTX4070 graphics card with wave size 32**

### Assets/PF_FLIP
2D FLIP using Unity Jobs system.
and some legacy test code

### Assets/MarchingCubesGPU
build surface from density field on GPU

### Assets/Sort
GPU Radix Sort <https://github.com/b0nes164/GPUSorting>

### Documentation:
<https://zhuanlan.zhihu.com/p/2021899451053213413>
### video:
<https://www.bilibili.com/video/BV1usQvByE4h>
