use vector::Vector;
use sparse_symmetric_matrix::SparseSymmetricMatrix;
use sparse_row_matrix::SparseRowMatrix;
use preconditioner::Preconditioner;

pub struct SolverResult {
    pub completed: bool,
    pub iterations: usize,
    pub best_guess: Vector,
}

const MAX_ITERATIONS: usize = 100;
const TOLERANCE_FACTOR: f64 = 1e-5;

pub fn solver(m: &SparseSymmetricMatrix, rhs: &Vector) -> SolverResult {
    let mut r = Vector(rhs.0.iter().cloned().collect());
    let residual_out = r.largest_absolute_value();
    if residual_out == 0.0 {
        return SolverResult {
            completed: false,
            iterations: 0,
            best_guess: r,
        };
    }

    let tolerance = TOLERANCE_FACTOR * residual_out;
    let ic_factor = Preconditioner::new(&m);
    let z = ic_factor.apply(&r);

    let mut rho = z.dot(&r);
    if rho == 0.0 || !rho.is_normal() {
        return SolverResult {
            completed: false,
            iterations: 0,
            best_guess: r,
        };
    }

    let mut result = Vector(vec![0.0; rhs.0.len()]);
    let mut s = z;
    let srm = SparseRowMatrix::new(&m);

    for iteration in 0..MAX_ITERATIONS {
        let mut z = srm.apply(&s);
        let alpha = rho / s.dot(&z);
        result = result + s.scale(alpha);
        r = r + z.scale(-alpha);
        if r.largest_absolute_value() < tolerance {
            return SolverResult {
                completed: true,
                iterations: iteration + 1,
                best_guess: result,
            };
        }
        z = ic_factor.apply(&r);
        let rho_new = z.dot(&r);
        let beta = rho_new / rho;
        s = z + s.scale(beta);
        rho = rho_new;
    }

    SolverResult {
        completed: false,
        iterations: MAX_ITERATIONS,
        best_guess: r,
    }
}
