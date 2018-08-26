use std::vec::Vec;

use sparse_symmetric_matrix::SparseSymmetricMatrix;
use vector::Vector;

pub struct Preconditioner {
    length: usize,
    values: Vec<f64>,
    row_index: Vec<usize>,
    column_pointers: Vec<usize>,
    diagonals: Vec<f64>,
    inverse_diagonals: Vec<f64>,
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
            length: length,
            values: values,
            row_index: row_index,
            column_pointers: column_pointers,
            diagonals: diagonals,
            inverse_diagonals: inverse_diagonals,
        }
    }

    pub fn apply(&self, v: &Vector) -> Vector {
        // TODO
        return Vector(vec![]);
    }
}
