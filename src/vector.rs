use std::vec::Vec;
use std::ops::Add;

use validity::Validity;

#[derive(Debug)]
pub struct Vector(pub Vec<f64>);

impl Vector {
    pub fn largest_absolute_value(&self) -> f64 {
        assert!(self.is_valid());
        self.0.iter().fold(0.0, |acc, &e| acc.abs().max(e.abs()))
    }

    pub fn dot(&self, other: &Vector) -> f64 {
        assert_eq!(self.0.len(), other.0.len());
        assert!(self.is_valid());
        assert!(other.is_valid());
        self.0.iter().zip(other.0.iter()).fold(
            0.0,
            |accum, (x, y)| {
                accum + (x * y)
            },
        )
    }

    // do not use the Mul trait, we want to borrow self.
    pub fn scale(&self, scale: f64) -> Vector {
        assert!(scale.is_finite());
        assert!(self.is_valid());
        Vector(self.0.iter().map(|e| e * scale).collect())
    }
}

impl Add for Vector {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.0.len(), other.0.len());
        assert!(self.is_valid());
        assert!(other.is_valid());
        Vector(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(x, y)| x + y)
                .collect(),
        )
    }
}

impl Validity for Vector {
    fn is_valid(&self) -> bool {
        self.0
            .iter()
            .filter(|e| !e.is_finite())
            .collect::<Vec<&f64>>()
            .is_empty()
    }
}

impl Clone for Vector {
    fn clone(&self) -> Vector {
        Vector(self.0.clone())
    }
}

#[test]
fn test_largest_absolute_value() {
    let v = Vector(vec![-10.0, 2.0, 9.9]);
    assert_eq!(v.largest_absolute_value(), 10.0);
}

#[test]
fn test_dot() {
    let x = Vector(vec![2.0, 4.0, 6.0]);
    let y = Vector(vec![-1.0, -2.0, 3.0]);
    assert_eq!(x.dot(&y), 8.0);
}

#[test]
fn test_scale() {
    let v = Vector(vec![2.0, 4.0, 6.0]);
    assert_eq!(v.scale(0.0).0, vec![0.0, 0.0, 0.0]);
    assert_eq!(v.scale(-2.0).0, vec![-4.0, -8.0, -12.0]);
}

#[test]
fn test_add() {
    let x = Vector(vec![2.0, 4.0, 6.0]);
    let y = Vector(vec![-1.0, -2.0, 3.0]);
    assert_eq!((x + y).0, vec![1.0, 2.0, 9.0]);
}
