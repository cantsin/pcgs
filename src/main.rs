extern crate rustml;

mod sparse_symmetric_matrix;
mod sparse_row_matrix;
mod vector;

use sparse_symmetric_matrix::{SparseSymmetricMatrix, Entry};
use sparse_row_matrix::SparseRowMatrix;
use vector::Vector;

fn main() {
    let m = SparseSymmetricMatrix::new(&vec![
        Entry { x: 0, y: 0, v: 1.5 },
        Entry { x: 0, y: 1, v: 5.5 },
        Entry { x: 0, y: 2, v: 6.5 },
        Entry { x: 1, y: 1, v: 2.5 },
        Entry { x: 1, y: 2, v: 8.5 },
        Entry { x: 2, y: 2, v: 9.5 },
    ]);
    println!("m: {:?}", m);

    // sparse([1 1 1 2 2 2 3 3 3],...
    // [1 2 3 1 2 3 1 2 3],...
    // [1.5 5.5 6.5 7.5 2.5 8.5 9.5 0.5 3.5], 3, 3))

    let v: Vector = vec![3.0, 2.0, 1.0];
    println!("v: {:?}", v);

    let result = SparseRowMatrix::new(&m) * v;
    println!("result: {:?}", result);
}

#[cfg(test)]
mod tests {
    use rustml::*;
    use rustml::octave::*;
    use rustml::matrix::Matrix;

    #[test]
    fn test_builder() {
        let m =
            mat![
                1, 2, 3;
                4, 5, 6
            ];
        let s = builder().add_columns("x = $1; y = $2", &m);
        assert_eq!(s.to_string(), "1;\nx = [1,4]; y = [2,5];\n");
    }
}
