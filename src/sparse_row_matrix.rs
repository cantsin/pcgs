use std::vec::Vec;
use std::fmt;

use vector::Vector;
use sparse_symmetric_matrix::SparseSymmetricMatrix;
use validity::Validity;

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
            for (j, &item) in row.iter().enumerate() {
                values.push(matrix.values[i][j]);
                column_index.push(item);
            }
            row_pointers.push(values.len());
        }

        SparseRowMatrix {
            values,
            column_index,
            row_pointers,
        }
    }

    fn len(&self) -> usize {
        self.row_pointers.len() - 1
    }

    // do not use the Mul trait, we want to borrow self.
    pub fn apply(&self, rhs: &Vector) -> Vector {
        assert_eq!(self.len(), rhs.0.len());
        assert!(self.is_valid());
        let n = self.len();
        let mut result = vec![0.0; n];
        #[cfg_attr(feature = "cargo-clippy", allow(needless_range_loop))]
        for i in 0..n {
            let x = self.row_pointers[i];
            let y = self.row_pointers[i + 1];
            for j in x..y {
                let index = self.column_index[j];
                result[i] += self.values[j] * rhs.0[index];
            }
        }
        Vector(result)
    }
}

impl Validity for SparseRowMatrix {
    fn is_valid(&self) -> bool {
        self.values
            .iter()
            .filter(|e| !e.is_finite())
            .collect::<Vec<&f64>>()
            .is_empty()
    }
}

impl fmt::Debug for SparseRowMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        assert!(self.is_valid());
        let n = self.len();
        let mut rows = vec![];
        let mut columns = vec![];
        let mut values = vec![];
        for i in 0..n {
            let x = self.row_pointers[i];
            let y = self.row_pointers[i + 1];
            for j in x..y {
                rows.push(i + 1);
                columns.push(self.column_index[j] + 1);
                values.push(self.values[j]);
            }
        }
        writeln!(f, "sparse({:?},...", rows);
        writeln!(f, "       {:?},...", columns);
        write!(f, "       {:?}, {}, {})", values, n, n)
    }
}

#[cfg(test)]
mod tests {
    use sparse_symmetric_matrix::{SparseSymmetricMatrix, Entry};
    use vector::Vector;
    use sparse_row_matrix::SparseRowMatrix;

    #[test]
    fn test_construct() {
        let m = SparseSymmetricMatrix::new(&vec![
            Entry { x: 0, y: 0, v: 1.0 },
            Entry { x: 0, y: 1, v: 5.0 },
            Entry { x: 0, y: 2, v: 6.0 },
            Entry { x: 1, y: 1, v: 2.0 },
        ]);
        let srm = SparseRowMatrix::new(&m);
        assert_eq!(srm.values, vec![1.0, 5.0, 6.0, 5.0, 2.0, 6.0]);
        assert_eq!(srm.column_index, vec![0, 1, 2, 0, 1, 0]);
        assert_eq!(srm.row_pointers, vec![0, 3, 5, 6]);
    }

    #[test]
    fn test_apply() {
        let m = SparseSymmetricMatrix::new(&vec![
            Entry { x: 0, y: 0, v: 1.0 },
            Entry { x: 0, y: 1, v: 5.0 },
            Entry { x: 0, y: 2, v: 6.0 },
            Entry { x: 1, y: 1, v: 2.0 },
        ]);
        let srm = SparseRowMatrix::new(&m);
        let v = Vector(vec![1.0, 2.0, 3.0]);
        let result = srm.apply(&v);
        assert_eq!(result.0, vec![29.0, 9.0, 6.0]);
    }
}
