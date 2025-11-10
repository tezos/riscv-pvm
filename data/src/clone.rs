// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Cloning functionality

/// Persistent cloneable state
///
/// This trait is used to clone the persistent state of a value.
///
/// Our PVM data types sometimes contain ephemeral state that helps with running things faster.
/// However, a regular clone of such a type would also clone the ephemeral state, which is not
/// always desirable. This trait allows to clone only the persistent state, thereby discarding any
/// ephemeral state.
pub trait CloneState {
    /// Clone only the persistent state contained in this value.
    fn clone_state(&self) -> Self;
}

impl<T: CloneState, const N: usize> CloneState for [T; N] {
    fn clone_state(&self) -> Self {
        self.each_ref().map(CloneState::clone_state)
    }
}

// We don't implement `Box<T>` directly, to avoid bringing potentially large `T`s onto the stack.
impl<T: CloneState, const N: usize> CloneState for Box<[T; N]> {
    fn clone_state(&self) -> Self {
        let boxed_slice = self
            .as_ref()
            .iter()
            .map(CloneState::clone_state)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let Ok(this) = boxed_slice.try_into() else {
            unreachable!("The length is still the same, hence this conversion should not fail")
        };

        this
    }
}

#[cfg(test)]
mod tests {
    use super::CloneState;

    /// This test assures that cloning for boxed arrays works without stack overflow.
    /// You can validate the test by removing the `CloneState for Box<[T; N]>` implementation and
    /// replacing it with a `CloneState for Box<T>` implementation. This should cause a stack
    /// overflow during cloning.
    #[test]
    fn boxed_array_clone_safely() {
        #[derive(Debug, PartialEq, Eq)]
        struct TestElem([u8; 128]);

        impl CloneState for TestElem {
            fn clone_state(&self) -> Self {
                Self(self.0)
            }
        }

        const LEN: usize = 1_000_000;

        let original: Box<[TestElem; LEN]> =
            Box::new(std::array::from_fn(|_| TestElem([3u8; 128])));
        let cloned: Box<[TestElem; LEN]> = original.clone_state();
        assert_eq!(cloned, original);
    }
}
