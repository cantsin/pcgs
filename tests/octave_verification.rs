extern crate pcgs;
extern crate rustml;

use rustml::octave::*;
use std::process::Command;

use pcgs::sparse_symmetric_matrix::{SparseSymmetricMatrix, Entry};
use pcgs::sparse_row_matrix::SparseRowMatrix;
use pcgs::vector::Vector;

#[test]
fn test_sparse_multiplication() {
    let m = SparseSymmetricMatrix::new(&vec![
        Entry { x: 0, y: 0, v: 1.5 },
        Entry { x: 0, y: 1, v: 5.5 },
        Entry { x: 0, y: 2, v: 6.5 },
        Entry { x: 1, y: 1, v: 2.5 },
        Entry { x: 1, y: 2, v: 8.5 },
        Entry { x: 2, y: 2, v: 9.5 },
    ]);
    let v = Vector(vec![3.0, 2.0, 1.0]);
    let v2 = v.0.iter().cloned().collect::<Vec<f64>>();
    let srm = SparseRowMatrix::new(&m);
    let result = srm.apply(&v);

    let eval_str = format!("disp({:?} * $$')", m);
    let s = builder().add_vector(&eval_str, &v2);

    let filename = "tests/test.octave";
    assert!(s.run(filename).is_ok());
    let output = Command::new("octave").arg(filename).output().expect(
        "octave failed to start",
    );
    let final_string = format!(
        "   {}\n   {}\n   {}\n",
        result.0[0],
        result.0[1],
        result.0[2]
    );
    assert_eq!(final_string.as_bytes(), output.stdout.as_slice());
}
