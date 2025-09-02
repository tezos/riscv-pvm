# Commit Message Guidelines

This project follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) 
specification for commit messages.

## Enforcement

Commit message format is automatically enforced through our GitHub Actions workflow
[`.github/workflows/commitlint.yml`](../../.github/workflows/commitlint.yml) on all pull requests.

## Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

## Examples

```
feat: add transaction validation
fix(vm): resolve stack overflow in execution engine
docs: update smart contract deployment guide
refactor!: change block header structure
```

## Types

The core types `feat` and `fix` have fixed meaning as per specification.

Other unspecified types may be used. We define our own meaning for those:

- **docs**: Documentation changes
- **refactor**: Code change for practical reason that don't change the overall semantics
- **style**: Code change for stylistic reason
- **perf**: Performance improvements
- **test**: Changes related to tests
    - e.g. adding a new test, fixing an existing one
    - don't use this for bug fixes
- **chore**: Misc changes that primarily affect other developers 
  - e.g. Cargo changes, Makefile changes, scripts, auxiliary tools, internal stuff
  - don't use this for refactorings, or style changes
- **ci**: Continuous integration stuff

None of these additional types may change the overall semantics of our main PVM library.

## Description

Use imperative mood: "add xyz" instead of "added xyz".

Pick a brief description that highlights what applying the commit does. 

Ensure the information is understandable by others, not just you.

If beneficial, use the message body to expand on the title. You could provide more information on
"why" and "how".
