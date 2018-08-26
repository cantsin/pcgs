// trait to verify that each entry in a vector or matrix is a valid number.
pub trait Validity {
    fn is_valid(&self) -> bool;
}
