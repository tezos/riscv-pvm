# Error handling guidelines

This is a work-in-progress document/overview of how we handle errors in the RISC-V codebase.

## `panic!` vs `Result`

This section builds off of the [To panic! or not to panic!](https://doc.rust-lang.org/book/ch09-03-to-panic-or-not-to-panic.html) chapter of the rustbook.

Sometimes, it's unclear as to whether code should `panic` on an error case, or instead return a `Result`. There are a number of circumstances, many of which are covered in the chapter of the rustbook linked above. The main source of unclearness is likely to be the handling of *unrecoverable errors*.

There are two main kinds of unrecoverable error:
- errors caused by fundamental logic issues in the code and/or library
- errors that may be caused by users of an API, or by external state of some kind

### Logic errors

In addition to the 'local proof-in-code' circumstances mentioned above, there are other cases where we believe a property to be upheld, but the 'proof' is spread through several layers of module or code. This is especially the case when it comes to certain performance criticial code that makes use of `unsafe`. In this instance, use of `panic!` (or especially `assert!`) is
fully justified (and indeed implicitly done so in many places when indexing into arrays or vecs).

### API usage/External state

There are other cases, however, where errors may occur as a result of something the code in question cannot itself guard against. This could be an unexpected (but nevertheless valid) usage of the API of a module, or external state that _should_ uphold certain properties, but cannot be guaranteed to for all time. In such instances, especially in library code, the preference should be for returning errors, that users of the library can choose either to handle, or indeed exit the process _iff_ the error is indeed unrecoverable.

In general, for errors that are forseeable, we should avoid making the decision for the consumers of our code.