use std::vec::Vec;
use std::fmt;

pub struct SparseSymmetricMatrix {
    pub length: usize,
    pub indices: Vec<Vec<usize>>,
    pub values: Vec<Vec<f64>>,
}

pub struct Entry {
    pub x: u32,
    pub y: u32,
    pub v: f64,
}

impl SparseSymmetricMatrix {
    pub fn new(entries: &Vec<Entry>) -> SparseSymmetricMatrix {
        // TODO force symmetricness
        SparseSymmetricMatrix {
            length: 0,
            indices: vec![vec![]],
            values: vec![vec![]],
        }
    }
}

impl fmt::Debug for SparseSymmetricMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SparseSymmetricMatrix")
    }
}
