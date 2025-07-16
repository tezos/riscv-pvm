# RISC-V PVM Project Instructions

## Build and Test Commands
- Build: `make -C src/riscv build`
- Test: `make -C src/riscv test`
- Lint: `make -C src/riscv check`

## Rust Documentation Requirements

**Scope:** All `trait`, `fn`, `static`, `const`, `enum`, `union`, `struct`, type aliases, and macros.

**Visibility-based requirements:**
- **Mandatory:** Exported items (`pub`, `pub(crate)`, `pub(super)`)
- **Encouraged:** Private items
- **Encouraged:** Module documentation using `//!`

**Quality standards:**
- Describe what the item does, not how it works
- Use sentence case; punctuation only for complete sentences
- Include examples for complex public functions
- Document panics, errors, and safety requirements
- Minimum one complete sentence for public items

**Examples:**
```rust
// ✅ Good documentation
/// Creates a new parser with the given input buffer.
/// 
/// # Examples
/// ```
/// let parser = Parser::new("input text");
/// ```
/// 
/// # Panics
/// Panics if the input is empty.
pub fn new(input: &str) -> Self { ... }

// ❌ Poor documentation
/// new parser
pub fn new(input: &str) -> Self { ... }

// ✅ Private item (encouraged) - complete sentence
/// Validates internal state consistency.
fn validate(&self) -> bool { ... }

// ✅ Private item (encouraged) - fragment
/// Internal state validator
fn validate(&self) -> bool { ... }
```

**Edge cases:**
- Trait definitions require documentation
- Associated functions/constants require their own documentation based on visibility
- Re-exports require documentation if they change the API surface
- Generic parameters do not require documentation

**Enforcement:**
- Missing public documentation: blocking review issue
- Missing private documentation: advisory suggestion
- Poor quality documentation: advisory with improvement suggestions

## Rust Trait Implementation Order
**Rationale:** This ordering improves readability by placing type-specific methods before external trait implementations, following Rust community conventions.

**Required ordering:**
1. Type definition (`struct`, `enum`, `union`)
2. Inherent implementation (`impl TypeName`)
3. Trait implementations (`impl TraitName for TypeName`)

**Critical rule:** ALL trait implementations must appear AFTER the inherent implementation block, regardless of file location or context.

**Rules:**
- Inherent `impl` blocks must directly follow type definitions
- Empty inherent `impl` blocks may be omitted
- Trait implementations must appear after inherent implementations
- Trait implementations should be grouped together (adjacent, same visibility)
- Within trait groups, order alphabetically by trait name

**Examples:**
```rust
// ✅ Correct ordering
struct MyType {
    value: i32,
}

impl MyType {
    fn new(value: i32) -> Self {
        Self { value }
    }
}

impl Debug for MyType { ... }
impl Display for MyType { ... }

// ❌ Incorrect - trait impl before inherent impl
impl Debug for MyType { ... }
impl MyType { ... }

// ❌ Incorrect - trait impl before struct definition
impl<T: Debug> Debug for Container<T> { ... }

struct Container<T> { ... }

impl<T> Container<T> { ... }
```

**Detection pattern:** Search for `impl TraitName for TypeName` appearing before `impl TypeName` in the same file.

**Enforcement:** Violations of trait-before-inherent ordering are blocking review issues.
