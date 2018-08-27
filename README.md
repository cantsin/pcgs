# pcgs

Implements a preconditioned conjugate gradient with modified incomplete cholesky (0) preconditioner.

That is, given a large system of linear equations:

    Ax = b

solve x iteratively. Matrix must be symmetric positive definite.
