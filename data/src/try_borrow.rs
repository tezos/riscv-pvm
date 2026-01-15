// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

/// A trait for things where borrowing can fail.
pub trait TryBorrow {
    type Target: Sized;
    type Error: std::error::Error;

    /// Tries to borrow the underlying.
    ///
    /// # Example
    ///
    /// ```
    /// use octez_riscv_data::try_borrow::TryBorrow;
    ///
    /// use thiserror::Error;
    ///
    /// #[derive(Default)]
    /// struct Example {
    ///     data: Option<u32>,
    /// }
    ///
    /// #[derive(Debug, Error)]
    /// pub enum ExampleError {
    ///     #[error("The data is not available")]
    ///     Unavailable,
    /// }
    ///
    /// impl TryBorrow for Example {
    ///     type Target = u32;
    ///     type Error = ExampleError;
    ///     
    ///     fn try_borrow(&self) -> Result<&Self::Target, Self::Error> {
    ///         match self.data {
    ///             Some(ref data) => Ok(&data),
    ///             None => Err(ExampleError::Unavailable),
    ///         }
    ///     }
    /// }
    ///
    /// let mut example = Example::default();
    /// assert!(matches!(example.try_borrow(), Err(ExampleError::Unavailable)));
    /// example.data = Some(12);
    /// assert!(matches!(example.try_borrow(), Ok(12)));
    ///
    /// ```
    fn try_borrow(&self) -> Result<&Self::Target, Self::Error>;
}

/// A trait for mutable borrowing something where the borrowing can fail.
pub trait TryBorrowMut: TryBorrow {
    /// Tries to borrow the underlying in a mutable way
    ///
    /// # Example
    ///
    /// ```
    /// use octez_riscv_data::try_borrow::TryBorrow;
    /// use octez_riscv_data::try_borrow::TryBorrowMut;
    ///
    /// use thiserror::Error;
    ///
    /// #[derive(Default)]
    /// struct Example {
    ///     data: Option<u32>,
    /// }
    ///
    /// #[derive(Debug, Error)]
    /// pub enum ExampleError {
    ///     #[error("The data is not available")]
    ///     Unavailable,
    /// }
    ///
    /// impl TryBorrow for Example {
    ///     type Target = u32;
    ///     type Error = ExampleError;
    ///     
    ///     fn try_borrow(&self) -> Result<&Self::Target, Self::Error> {
    ///         match self.data {
    ///             Some(ref data) => Ok(data),
    ///             None => Err(ExampleError::Unavailable),
    ///         }
    ///     }
    /// }
    ///
    /// impl TryBorrowMut for Example {
    ///     fn try_borrow_mut(&mut self) -> Result<&mut Self::Target, Self::Error> {
    ///         match self.data {
    ///             Some(ref mut data) => Ok(data),
    ///             None => Err(ExampleError::Unavailable),
    ///         }
    ///     }
    /// }
    ///
    /// let mut example = Example::default();
    /// example.data = Some(13);
    /// *example.try_borrow_mut().unwrap() = 14;
    /// assert!(matches!(example.try_borrow(), Ok(14)));
    ///
    /// ```
    fn try_borrow_mut(&mut self) -> Result<&mut Self::Target, Self::Error>;
}
