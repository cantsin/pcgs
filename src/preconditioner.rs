use std::vec::Vec;

use sparse_symmetric_matrix::SparseSymmetricMatrix;
use vector::Vector;

pub struct Preconditioner {
    pub length: usize,
    pub values: Vec<f64>,
    pub row_index: Vec<usize>,
    pub column_pointers: Vec<usize>,
    pub inverse_diagonals: Vec<f64>,
}

const MODIFIED_PARAMETER: f64 = 0.97;
const MIN_DIAGONAL_RATIO: f64 = 0.25;

impl Preconditioner {
    pub fn new(matrix: &SparseSymmetricMatrix) -> Preconditioner {
        let mut values = vec![];
        let mut row_index = vec![];
        let mut column_pointers = vec![];
        let mut diagonals = vec![];
        let mut inverse_diagonals = vec![];

        // algorithm from Robert Bridson, see:
        // https://www.cs.ubc.ca/~rbridson/fluidsimulation/

        // lower triangle
        for i in 0..matrix.length + 1 {
            column_pointers.push(row_index.len());
            diagonals.push(0.0);
            inverse_diagonals.push(0.0);
            for j in 0..matrix.indices[i].len() {
                let index = matrix.indices[i][j];
                let value = matrix.values[i][j];
                if index > i {
                    row_index.push(index);
                    values.push(value);
                } else if index == i {
                    diagonals[i] = value;
                    inverse_diagonals[i] = value;
                }
            }
        }
        column_pointers.push(row_index.len());

        let length = column_pointers.len() - 1;
        for k in 0..length {
            if diagonals[k] == 0.0 {
                // null row and column
                continue;
            }

            let gauss_sidel = inverse_diagonals[k] < (MIN_DIAGONAL_RATIO * diagonals[k]);
            if gauss_sidel {
                inverse_diagonals[k] = 1.0 / diagonals[k].sqrt();
            } else {
                inverse_diagonals[k] = 1.0 / inverse_diagonals[k].sqrt();
            }

            let col_s = column_pointers[k];
            let col_t = column_pointers[k + 1];
            #[cfg_attr(feature = "cargo-clippy", allow(needless_range_loop))]
            for p in col_s..col_t {
                values[p] *= inverse_diagonals[k];
            }

            for p in col_s..col_t {
                let j = row_index[p];
                let multiplier = values[p];
                let mut missing = 0.0;
                let mut a = col_s;
                let mut b = 0;
                while a < col_t && row_index[a] < j {
                    while b < matrix.indices[j].len() {
                        let current_row = row_index[a];
                        let index = matrix.indices[j][b];
                        if index < current_row {
                            b += 1;
                        } else if index == current_row {
                            break;
                        } else {
                            missing += values[a];
                            break;
                        }
                    }
                    a += 1;
                }

                if a < col_t && row_index[a] == j {
                    inverse_diagonals[j] -= multiplier * values[a];
                }

                a += 1;
                b = column_pointers[j];
                while a < col_t && b < column_pointers[j + 1] {
                    let current_row = row_index[a];
                    if row_index[b] < current_row {
                        b += 1;
                    } else if row_index[b] == current_row {
                        values[b] -= multiplier * values[a];
                        a += 1;
                        b += 1;
                    } else {
                        missing += values[a];
                        a += 1;
                    }
                }

                while a < col_t {
                    missing += values[a];
                    a += 1;
                }

                inverse_diagonals[j] -= MODIFIED_PARAMETER * multiplier * missing;
            }
        }
        Preconditioner {
            length,
            values,
            row_index,
            column_pointers,
            inverse_diagonals,
        }
    }

    pub fn apply(&self, v: &Vector) -> Vector {
        let z = self.solve_lower(&v);
        self.solve_lower_transpose(&z)
    }

    fn solve_lower(&self, v: &Vector) -> Vector {
        let mut result = v.clone();
        for i in 0..self.length {
            result.0[i] *= self.inverse_diagonals[i];
            let x = self.column_pointers[i];
            let y = self.column_pointers[i + 1];
            for j in x..y {
                let index = self.row_index[j];
                result.0[index] -= self.values[j] * result.0[i];
            }
        }
        result
    }

    fn solve_lower_transpose(&self, v: &Vector) -> Vector {
        let mut result = v.clone();
        let n = self.length - 1;
        for i in (0..n).rev() {
            let x = self.column_pointers[i];
            let y = self.column_pointers[i + 1];
            for j in x..y {
                let index = self.row_index[j];
                result.0[i] -= self.values[j] * result.0[index];
            }
            result.0[i] *= self.inverse_diagonals[i];
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use sparse_symmetric_matrix::{SparseSymmetricMatrix, Entry};
    use preconditioner::Preconditioner;

    #[test]
    fn test_positive_definite_matrix_preconditioner() {
        let m = SparseSymmetricMatrix::new(&vec![
            Entry {
                x: 0,
                y: 0,
                v: 0.37,
            },
            Entry {
                x: 1,
                y: 0,
                v: -0.05,
            },
            Entry {
                x: 2,
                y: 0,
                v: -0.05,
            },
            Entry {
                x: 3,
                y: 0,
                v: -0.07,
            },
            Entry {
                x: 1,
                y: 1,
                v: 0.116,
            },
            Entry { x: 2, y: 1, v: 0.0 },
            Entry {
                x: 3,
                y: 1,
                v: -0.05,
            },
            Entry {
                x: 2,
                y: 2,
                v: 0.116,
            },
            Entry {
                x: 3,
                y: 2,
                v: -0.05,
            },
            Entry {
                x: 3,
                y: 3,
                v: 0.202,
            },
        ]);
        let p = Preconditioner::new(&m);
        assert_eq!(p.length, 4);
        assert_eq!(
            p.values,
            vec![
                -0.08219949365267866,
                -0.08219949365267866,
                -0.11507929111375013,
                -0.020442828820163496,
                -0.1798968936174387,
                -0.1913900502726929,
            ]
        );
        assert_eq!(p.row_index, vec![1, 2, 3, 2, 3, 3]);
        assert_eq!(p.column_pointers, vec![0, 3, 5, 6, 6]);
        assert_eq!(
            p.inverse_diagonals,
            vec![
                1.6439898730535731,
                3.0255386653841962,
                3.031342410667025,
                2.889597639959034,
            ]
        );
    }
}
