use std::vec::Vec;
use std::cmp::{min, max};
use std::fmt;

pub struct SparseSymmetricMatrix {
    pub length: usize,
    pub indices: Vec<Vec<usize>>,
    pub values: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub x: usize,
    pub y: usize,
    pub v: f64,
}

impl Entry {
    fn lower_triangle(&self) -> Entry {
        Entry { x: min(self.x, self.y), y: max(self.x, self.y), v: self.v }
    }
}

impl SparseSymmetricMatrix {
    pub fn new(entries: &Vec<Entry>) -> SparseSymmetricMatrix {
        let mut sorted_entries = entries
            .iter()
            .cloned()
            .map(|e| e.lower_triangle())
            .collect::<Vec<Entry>>();
        sorted_entries
            .sort_by(|a, b| {
                return if a.x == b.x {
                    a.y.cmp(&b.y)
                } else {
                    a.x.cmp(&b.x)
                }
            });
        sorted_entries.dedup_by(|a, b| a.x == b.x && a.y == b.y);
        let length = sorted_entries.iter().fold(0, |acc, e| max(acc, max(e.x, e.y)));
        let mut indices = vec![vec![]];
        let mut values = vec![vec![]];
        for entry in sorted_entries {
            // empty columns
            while indices.len() <= entry.x {
                indices.push(vec![]);
                values.push(vec![]);
            }
            indices[entry.x].push(entry.y);
            values[entry.x].push(entry.v);
        }
        SparseSymmetricMatrix {
            length: length,
            indices: indices,
            values: values,
        }
    }
}

impl fmt::Debug for SparseSymmetricMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let n = self.length + 1;
        let mut rows = vec![];
        let mut columns = vec![];
        let mut values = vec![];
        for i in 0..n {
            for j in 0..self.indices[i].len() {
                rows.push(i + 1);
                columns.push(self.indices[i][j]);
            }
        }
        for i in 0..n {
            for j in 0..self.values[i].len() {
                values.push(self.values[i][j]);
            }
        }
        writeln!(f, "sparse({:?},...", rows);
        writeln!(f, "       {:?},...", columns);
        write!(f,   "       {:?}, {}, {})", values, n, n)
    }
}

#[test]
fn test_construct() {
    let m = SparseSymmetricMatrix::new(&vec![
        Entry { x: 0, y: 0, v: 1.5 },
        Entry { x: 0, y: 1, v: 5.5 },
        Entry { x: 0, y: 2, v: 6.5 },
        Entry { x: 1, y: 1, v: 2.5 },
        Entry { x: 1, y: 2, v: 8.5 },
        Entry { x: 2, y: 2, v: 9.5 },
    ]);
    assert_eq!(m.length, 2);
    assert_eq!(m.indices, vec![vec![0, 1, 2], vec![1, 2], vec![2]]);
    assert_eq!(m.values, vec![vec![1.5, 5.5, 6.5], vec![2.5, 8.5], vec![9.5]]);
}

#[test]
fn test_mixed_construct() {
    let m = SparseSymmetricMatrix::new(&vec![
        Entry { x: 2, y: 2, v: 9.5 },
        Entry { x: 0, y: 0, v: 1.5 },
        Entry { x: 0, y: 2, v: 6.5 },
        Entry { x: 1, y: 1, v: 2.5 },
        Entry { x: 1, y: 2, v: 8.5 },
        Entry { x: 0, y: 1, v: 5.5 },
    ]);
    assert_eq!(m.length, 2);
    assert_eq!(m.indices, vec![vec![0, 1, 2], vec![1, 2], vec![2]]);
    assert_eq!(m.values, vec![vec![1.5, 5.5, 6.5], vec![2.5, 8.5], vec![9.5]]);
}

#[test]
fn test_duplicate_construct() {
    let m = SparseSymmetricMatrix::new(&vec![
        Entry { x: 0, y: 1, v: 5.5 },
        Entry { x: 0, y: 0, v: 1.5 },
        Entry { x: 1, y: 1, v: 2.5 },
        Entry { x: 2, y: 1, v: 8.5 },
        Entry { x: 1, y: 1, v: 2.5 },
        Entry { x: 0, y: 2, v: 6.5 },
        Entry { x: 2, y: 2, v: 9.5 },
        Entry { x: 2, y: 2, v: 9.5 },
        Entry { x: 2, y: 0, v: 6.5 },
    ]);
    assert_eq!(m.length, 2);
    assert_eq!(m.indices, vec![vec![0, 1, 2], vec![1, 2], vec![2]]);
    assert_eq!(m.values, vec![vec![1.5, 5.5, 6.5], vec![2.5, 8.5], vec![9.5]]);
}

#[test]
fn test_sparse_construct() {
    let m = SparseSymmetricMatrix::new(&vec![
        Entry { x: 10, y: 5, v: 10.0 },
        Entry { x: 2, y: 8, v: 9.0 },
    ]);
    assert_eq!(m.length, 10);
    assert_eq!(m.indices, vec![vec![], vec![], vec![8], vec![], vec![], vec![10]]);
    assert_eq!(m.values, vec![vec![], vec![], vec![9.0], vec![], vec![], vec![10.0]]);
}
