# Assighment-2.1 Poisson image editing

## METHOD
 By optimizing the Laplacian loss of the blended_image to zero with boundary conditions:
blended_image'''[inner_boundary]''' == background_image'''[inner_boundary]'''

One can achive seamless blending.

see paper:
- [paper Poisson image editing](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)

## RESULT

### optimizing Laplacian loss without fixing the boundary:
<img src="poisson-editing\image\res-1.png" alt="blended image withoud fixing boundary" width="800">

### with boundary condition:
<img src="poisson-editing\image\res-2.png" alt="blended image withoud fixing boundary" width="800">

