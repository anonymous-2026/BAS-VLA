# Public Scope

This document records what the repository currently includes and intentionally leaves out.

## Included

- BAS-VLA method code that is stable enough to preserve, including the public
  breaking core and preserving-auxiliary modules
- benchmark/task configuration files that are already reusable
- public OpenPI + LIBERO evaluation and launch scripts
- lightweight repository notes
- protocol notes that help avoid unsupported public setups

## Excluded

- raw experiment records
- run-specific bookkeeping
- historical working notes
- external method comparison records
- result-heavy review documents
- unsupported wrappers and benchmark hacks

## Inclusion rule

Something should only be copied into this repository if at least one of the following is true:

1. it is part of the method that is expected to remain in the public repository
2. it is a stable configuration that users of the future repository will need
3. it defines an important public protocol constraint
4. it is required to explain why a public release intentionally excludes a fragile path
