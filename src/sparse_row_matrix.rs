use std::vec::Vec;
use std::ops::Mul;
use std::fmt;

use vector::Vector;
use sparse_symmetric_matrix::SparseSymmetricMatrix;

// we use this structure only for multiplication as it is more
// efficient for this purpose than SparseSymmetricMatrix.
pub struct SparseRowMatrix {
    values: Vec<f64>,
    column_index: Vec<usize>,
    row_pointers: Vec<usize>,
}

impl SparseRowMatrix {
    pub fn new(matrix: &SparseSymmetricMatrix) -> SparseRowMatrix {
        let mut values = vec![];
        let mut column_index = vec![];
        let mut row_pointers = vec![0];
        for i in 0..matrix.length + 1 {
            let row = &matrix.indices[i];
            for j in 0..row.len() {
                values.push(matrix.values[i][j]);
                column_index.push(row[j]);
            }
            row_pointers.push(values.len());
        }

        SparseRowMatrix {
            values: values,
            column_index: column_index,
            row_pointers: row_pointers
        }
    }

    fn len(&self) -> usize {
        return self.row_pointers.len() - 1;
    }
}

impl fmt::Debug for SparseRowMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let n = self.len();
        let mut rows = vec![];
        let mut columns = vec![];
        let mut values = vec![];
        for i in 0..n {
            let x = self.row_pointers[i];
            let y = self.row_pointers[i+1];
            for j in x..y {
                rows.push(i + 1);
                columns.push(self.column_index[j] + 1);
                values.push(self.values[j]);
            }
        }
        writeln!(f, "sparse({:?},...", rows);
        writeln!(f, "       {:?},...", columns);
        write!(f,   "       {:?}, {}, {})", values, n, n)
    }
}

impl Mul<Vector> for SparseRowMatrix {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Vector {
        assert!(self.len() == rhs.len());
        let n = self.len();
        let mut result = vec![];
        for i in 0..n {
            result.push(0.0);
            let x = self.row_pointers[i];
            let y = self.row_pointers[i+1];
            for j in x..y {
                let index = self.column_index[j];
                result[i] += self.values[j] * rhs[index];
            }
        }
        result
    }
}
