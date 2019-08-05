This directory contains some dfa inputs that are always safe.
These inputs share the same structure:
 * two states, one initial (safe), one final (unsafe)
 * no inputs
 * one output
 * three transitions
   * safe action: from initial (safe) -> initial (safe)
   * mistake n: from initial (safe) -> final (unsafe)
   * from final (unsafe) -> final (unsafe)
All dfas have a different action encoding: `mistake_1`, `mistake_2`, `mistake_3`.
