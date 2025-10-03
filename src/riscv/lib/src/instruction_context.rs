// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! The instruction context forms the building blocks used for executing RISC-V instructions.
//!
//! By providing these building blocks for various execution formats, the same implementation can
//! be used for both interpretation and compilation of instructions.

pub(super) mod arithmetic;
pub(super) mod comparable;
pub(crate) mod value;

use arithmetic::Arithmetic;
use comparable::Comparable;
use rustc_apfloat::Float;
use rustc_apfloat::Status;
use rustc_apfloat::StatusAnd;
use rustc_apfloat::ieee::Double;

pub use self::value::StoreLoadInt;
use crate::exceptions::Exception;
use crate::instruction_context::value::PhiValue;
use crate::interpreter::float::RoundingMode;
use crate::machine_state::MachineCoreState;
use crate::machine_state::ProgramCounterUpdate;
use crate::machine_state::csregisters::CSRegister;
use crate::machine_state::instruction::Args;
use crate::machine_state::memory::BadMemoryAccess;
use crate::machine_state::memory::Memory;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::registers::FValue;
use crate::machine_state::registers::XValue;
use crate::machine_state::registers::XValue32;
use crate::parser::instruction::InstrWidth;
use crate::state_backend::ManagerReadWrite;
use crate::state_context::StateContext;

/// Type of function that may be used to lower [`Instructions`] to IR.
///
/// [`Instructions`]: crate::machine_state::instruction::Instruction
pub type IcbLoweringFn<I> = unsafe fn(&Args, &mut I) -> IcbFnResult<I>;

/// Result of lowering an instruction.
pub type IcbFnResult<I> = <I as ICB>::IResult<ProgramCounterUpdate<<I as ICB>::XValue>>;

/// Instruction Context Builder contains operations required to
/// execute RISC-V instructions.
#[expect(clippy::upper_case_acronyms, reason = "ICB looks cooler than Icb")]
pub(crate) trait ICB
where
    // These constraints let us tie a type-equality knot.
    //
    // We want to allow handling of arbitrary value types via [`StateContext`]'s associated type
    // `Value`, whilst adding functionality on a subset of those values (e.g. 64-bit and 32-bit
    // integers, and floating-point numbers).
    //
    // There are two ways one can achieve this:
    //
    // 1. Define associated types with those additional constraints (those bring the additional
    //    functionality into scope). Then force each specific `Self::Value<_>` to be equal to the
    //    new associated types.
    //
    // 2. Constrain the instantiated `Self::Value<_>` directly.
    //
    // The second approach would be nicer, if it weren't for Rust's inability to express constraints
    // on associated types like super-trait relationships. This means, if you use `ICB` as a
    // constraint, you also need to express all its inherited constraints on the associated types
    // of `StateContext` in the trait bounds. This results in massive amount of boilerplate being
    // added everywhere. Imagine how many ICB-style instruction implementations there are that
    // mention `ICB` in their trait bounds.
    // Because of this, we use the first approach, which is more verbose with respect to the number
    // of types being declared, but is equally as powerful.
    Self: StateContext<Value<u64> = Self::XValue>
        + StateContext<Value<u32> = Self::XValue32>
        + StateContext<Value<FValue> = Self::FValue>,
{
    /// A 64-bit value stored in [`XRegisters`].
    ///
    /// [`XRegisters`]: crate::machine_state::registers::XRegisters
    type XValue: Arithmetic<Self> + Comparable<Self, Result = Self::Bool>;

    /// A 64-bit floating-point value stored in [`FRegisters`].
    ///
    /// [`FRegisters`]: crate::machine_state::registers::FRegisters
    type FValue;

    /// Construct an [`ICB::XValue`] from an `imm: i64`.
    fn xvalue_of_imm(&mut self, imm: i64) -> Self::XValue;

    /// Construct an [`ICB::XValue32`] from an `imm: i32`.
    fn xvalue32_of_imm(&mut self, imm: i32) -> Self::XValue32;

    /// Perform a read of the program counter.
    fn pc_read(&mut self) -> Self::XValue;

    /// Type for boolean operations.
    type Bool;

    /// Perform a logical `and` operation of two [`ICB::Bool`] values.
    fn bool_and(&mut self, lhs: Self::Bool, rhs: Self::Bool) -> Self::Bool;

    /// A 32-bit value to be used only in word-width operations.
    type XValue32: Arithmetic<Self> + Comparable<Self, Result = Self::Bool>;

    /// Convert an [`XValue`] to a [`XValue32`].
    fn narrow(&mut self, value: Self::XValue) -> Self::XValue32;

    /// Sign-extend an [`XValue32`] to an [`XValue`].
    fn extend_signed(&mut self, value: Self::XValue32) -> Self::XValue;

    /// Zero-extend an [`XValue32`] to an [`XValue`].
    #[expect(dead_code, reason = "Will Be Used Soon™")]
    fn extend_unsigned(&mut self, value: Self::XValue32) -> Self::XValue;

    /// Multiply two [`XValue`] values and return the high 64 bits of the result, with
    /// the appropriate sign-extension passed in as 2 boolean arguments.
    fn mul_high(
        &mut self,
        lhs: Self::XValue,
        rhs: Self::XValue,
        mul_high_type: MulHighType,
    ) -> Self::XValue;

    /// Convert a boolean value to an xvalue.
    ///
    /// Coerces to the following:
    /// - `true -> 1`
    /// - `false -> 0`
    fn xvalue_from_bool(&mut self, value: Self::Bool) -> Self::XValue;

    /// Branching instruction.
    ///
    /// If `condition` is true, the branch will be taken. The PC update
    /// will be to the address returned by `take_branch`.
    ///
    /// If false, the PC update is to the next instruction.
    fn branch(
        &mut self,
        condition: Self::Bool,
        offset: i64,
        instr_width: InstrWidth,
    ) -> ProgramCounterUpdate<Self::XValue>;

    /// Run the IR code produced by the `true_branch` if `cond` is true. If the condition is false,
    /// the IR code following this call will be executed instead.
    fn if_then<OnTrue>(&mut self, cond: Self::Bool, true_branch: OnTrue) -> Self::IResult<()>
    where
        OnTrue: FnOnce(&mut Self) -> Self::IResult<()>;

    /// Take a branch based on the given condition and return to a common line of execution.
    ///
    /// This is used for situations where we have a common execution path following branching.
    /// The `cond` is the condition to branch on, and the `true_branch` and `false_branch` are the
    /// functions to execute for the left and right branches, respectively.
    ///
    /// Semantically, this function returns the caller into the context of the common
    /// execution path with the resulting value of the branch that was taken.
    fn if_then_else<Phi: PhiValue, OnTrue, OnFalse>(
        &mut self,
        cond: Self::Bool,
        true_branch: OnTrue,
        false_branch: OnFalse,
    ) -> Phi::IcbValue<Self>
    where
        OnTrue: FnOnce(&mut Self) -> Phi::IcbValue<Self>,
        OnFalse: FnOnce(&mut Self) -> Phi::IcbValue<Self>;

    /// Representation for the manipulation of fallible operations.
    type IResult<Value>;

    /// Wrap a value as a fallible value.
    fn ok<Value>(&mut self, val: Value) -> Self::IResult<Value>;

    /// Raise an exception, returning a fallible value.
    fn raise_exception<In>(&mut self, exception: Exception) -> Self::IResult<In>;

    /// Map the fallible-value into a fallible-value of a different type.
    fn map<Value, Next, F>(res: Self::IResult<Value>, f: F) -> Self::IResult<Next>
    where
        F: FnOnce(Value) -> Next;

    /// Run a fallible operation over the fallible-value as input.
    fn and_then<Value, Next, F>(res: Self::IResult<Value>, f: F) -> Self::IResult<Next>
    where
        F: FnOnce(Value) -> Self::IResult<Next>;

    /// Write value to main memory, at the given address.
    ///
    /// The value is truncated to the width given by [`LoadStoreWidth`].
    fn main_memory_store<V: StoreLoadInt>(
        &mut self,
        phys_address: Self::XValue,
        value: Self::XValue,
    ) -> Self::IResult<()>;

    /// Read value from main memory, at the given address.
    ///
    /// The value is truncated to the width given by [`LoadStoreWidth`].
    fn main_memory_load<V: StoreLoadInt>(
        &mut self,
        phys_address: Self::XValue,
    ) -> Self::IResult<Self::XValue>;

    /// Take an `XValue` and convert it to a 64-bit float with the dynamic rounding mode in the `frm` field of the
    /// `fcsr` register, returning the result as an `FValue`.
    fn f64_from_x64_unsigned_dynamic(&mut self, xval: Self::XValue) -> Self::FValue;

    /// Take an `XValue` and a static rounding mode, and convert it to a 64-bit float
    /// with the given rounding mode, returning the resulting `FValue`.
    fn f64_from_x64_unsigned_static(
        &mut self,
        xval: Self::XValue,
        rm: RoundingMode,
    ) -> Self::FValue;

    /// Read the value from a Control and Status register.
    fn csr_read(&mut self, reg: CSRegister) -> Self::XValue;

    /// Write a value to a Control and Status register.
    fn csr_write(&mut self, reg: CSRegister, value: Self::XValue);
}

impl<MC: MemoryConfig, M: ManagerReadWrite> ICB for MachineCoreState<MC, M> {
    type XValue = XValue;

    type FValue = FValue;

    #[inline(always)]
    fn xvalue_of_imm(&mut self, imm: i64) -> Self::XValue {
        imm as u64
    }

    fn xvalue32_of_imm(&mut self, imm: i32) -> Self::XValue32 {
        imm as u32
    }

    #[inline(always)]
    fn pc_read(&mut self) -> Self::XValue {
        self.hart.pc.read()
    }

    type Bool = bool;

    #[inline(always)]
    fn bool_and(&mut self, lhs: Self::Bool, rhs: Self::Bool) -> Self::Bool {
        lhs && rhs
    }

    type XValue32 = XValue32;

    #[inline(always)]
    fn narrow(&mut self, value: Self::XValue) -> Self::XValue32 {
        value as u32
    }

    #[inline(always)]
    fn extend_signed(&mut self, value: Self::XValue32) -> Self::XValue {
        value as i32 as u64
    }

    #[inline(always)]
    fn extend_unsigned(&mut self, value: Self::XValue32) -> Self::XValue {
        value as u64
    }

    #[inline(always)]
    fn mul_high(
        &mut self,
        lhs: Self::XValue,
        rhs: Self::XValue,
        mul_high_type: MulHighType,
    ) -> Self::XValue {
        let (lhs, rhs) = match mul_high_type {
            MulHighType::Signed => (lhs as i64 as i128 as u128, rhs as i64 as i128 as u128),
            MulHighType::Unsigned => (lhs as u128, rhs as u128),
            MulHighType::SignedUnsigned => (lhs as i64 as i128 as u128, rhs as u128),
        };
        let result = lhs.wrapping_mul(rhs);

        (result >> 64) as u64
    }

    #[inline(always)]
    fn xvalue_from_bool(&mut self, value: Self::Bool) -> Self::XValue {
        value as XValue
    }

    #[inline(always)]
    fn branch(
        &mut self,
        predicate: Self::Bool,
        offset: i64,
        instr_width: InstrWidth,
    ) -> ProgramCounterUpdate<Self::XValue> {
        if predicate {
            let pc = self.pc_read();
            let address = pc.wrapping_add_signed(offset);
            ProgramCounterUpdate::Set(address)
        } else {
            ProgramCounterUpdate::Next(instr_width)
        }
    }

    #[inline]
    fn if_then<OnTrue>(&mut self, cond: Self::Bool, true_branch: OnTrue) -> Self::IResult<()>
    where
        OnTrue: FnOnce(&mut Self) -> Self::IResult<()>,
    {
        if cond {
            return true_branch(self);
        }

        Ok(())
    }

    #[inline(always)]
    fn if_then_else<Phi: PhiValue, OnTrue, OnFalse>(
        &mut self,
        cond: Self::Bool,
        true_branch: OnTrue,
        false_branch: OnFalse,
    ) -> Phi::IcbValue<Self>
    where
        OnTrue: FnOnce(&mut Self) -> Phi::IcbValue<Self>,
        OnFalse: FnOnce(&mut Self) -> Phi::IcbValue<Self>,
    {
        if cond {
            true_branch(self)
        } else {
            false_branch(self)
        }
    }

    type IResult<In> = Result<In, Exception>;

    #[inline(always)]
    fn ok<In>(&mut self, val: In) -> Self::IResult<In> {
        Ok(val)
    }

    #[inline]
    fn raise_exception<In>(&mut self, exception: Exception) -> Self::IResult<In> {
        Err(exception)
    }

    #[inline(always)]
    fn map<In, Out, F>(res: Self::IResult<In>, f: F) -> Self::IResult<Out>
    where
        F: FnOnce(In) -> Out,
    {
        res.map(f)
    }

    #[inline(always)]
    fn and_then<In, Out, F>(res: Self::IResult<In>, f: F) -> Self::IResult<Out>
    where
        F: FnOnce(In) -> Self::IResult<Out>,
    {
        res.and_then(f)
    }

    #[inline(always)]
    fn main_memory_store<V: StoreLoadInt>(
        &mut self,
        address: Self::XValue,
        value: Self::XValue,
    ) -> Self::IResult<()> {
        self.main_memory
            .write(address, V::from_xvalue(value))
            .map_err(|_: BadMemoryAccess| Exception::StoreAMOAccessFault)
    }

    #[inline(always)]
    fn main_memory_load<V: StoreLoadInt>(
        &mut self,
        address: Self::XValue,
    ) -> Self::IResult<Self::XValue> {
        self.main_memory
            .read(address)
            .map(V::to_xvalue)
            .map_err(|_: BadMemoryAccess| Exception::LoadAccessFault)
    }

    fn f64_from_x64_unsigned_static(
        &mut self,
        xval: Self::XValue,
        rm: RoundingMode,
    ) -> Self::FValue {
        let extended = xval as u128;

        let StatusAnd { status, value } = Double::from_u128_r(extended, rm.into());

        if status != Status::OK {
            self.hart.csregisters.import_float_exception_flags(status);
        }

        value.into()
    }

    fn f64_from_x64_unsigned_dynamic(&mut self, xval: Self::XValue) -> Self::FValue {
        let extended = xval as u128;
        let rm: RoundingMode = self.hart.csregisters.frm.read();

        let StatusAnd { status, value } = Double::from_u128_r(extended, rm.into());

        if status != Status::OK {
            self.hart.csregisters.import_float_exception_flags(status);
        }

        FValue::from(value)
    }

    fn csr_read(&mut self, reg: CSRegister) -> Self::XValue {
        self.hart.csregisters.read(reg)
    }

    fn csr_write(&mut self, reg: CSRegister, value: Self::XValue) {
        self.hart.csregisters.write(reg, value);
    }
}

/// Operators for producing a boolean from two values.
pub enum Predicate {
    Equal,
    NotEqual,
    LessThanSigned,
    LessThanUnsigned,
    LessThanOrEqualSigned,
    GreaterThanSigned,
    GreaterThanOrEqualSigned,
    GreaterThanOrEqualUnsigned,
}

/// The type of shift operation to perform.
pub enum Shift {
    /// Logical left shift. Zeroes are shifted into the least significant bits.
    Left,
    /// Logical right shift. Zeroes are shifted into the most significant bits.
    RightUnsigned,
    /// Arithmetic right shift. Sign-bits (ones) are shifted into the most significant bits.
    RightSigned,
}

/// The type of X64 mul_high operation to perform.
pub enum MulHighType {
    Signed,
    Unsigned,
    SignedUnsigned,
}

/// Supported value widths for loading from/storing to main memory for XRegisters.
///
/// **NB** This type may be passed over C-FFI. See [state_access] for more
/// information.
///
/// For now, the approach taken chooses to pass enums as integers, and parse
/// them back into the Enum variant on the rust side - to avoid potential UB
/// should an incorrect discriminant be parsed. We therefore choose explicit
/// constants for each - so that we know very precisely what values are expected.
///
/// [state_access]: crate::jit::state_access
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum LoadStoreWidth {
    Byte = Self::BYTE_WIDTH,
    Half = Self::HALF_WIDTH,
    Word = Self::WORD_WIDTH,
    Double = Self::DOUBLE_WIDTH,
}

impl LoadStoreWidth {
    const BYTE_WIDTH: u8 = std::mem::size_of::<u8>() as u8;
    const HALF_WIDTH: u8 = std::mem::size_of::<u16>() as u8;
    const WORD_WIDTH: u8 = std::mem::size_of::<u32>() as u8;
    const DOUBLE_WIDTH: u8 = std::mem::size_of::<u64>() as u8;
}
