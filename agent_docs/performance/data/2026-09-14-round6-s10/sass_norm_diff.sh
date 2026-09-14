#!/bin/bash
# Normalise the anonymous-namespace mangling before diffing: nvcc encodes the
# absolute source path and the PID of the temp file into every
# `_GLOBAL__N__<hash>_<file>_<hash>_<pid>` symbol, so two compiles of the SAME
# source from two different directories differ in the names and in nothing else.
set -u
OUT=/workspace/output/round6/s10/sass
norm() { sed -E 's/_GLOBAL__N__[0-9a-f]+_/_GLOBAL__N__X_/g; s/__nv_static_[0-9]+__[0-9a-f]+_/__nv_static_X__X_/g; s/_cu_[0-9a-f]+_[0-9]+/_cu_X_X/g' "$1"; }
for b in ortho_potential affine_body_bdf1_kinetic arap abd_linear_subsystem; do
  [ -s "$OUT/$b.base.sass" ] || { echo "MISSING $b"; continue; }
  norm "$OUT/$b.base.sass" > "$OUT/$b.base.norm"
  norm "$OUT/$b.head.sass" > "$OUT/$b.head.norm"
  nf=$(grep -c 'Function : ' "$OUT/$b.head.norm")
  ni=$(grep -cE '^\s+/\*[0-9a-f]+\*/' "$OUT/$b.head.norm")
  if diff -q "$OUT/$b.base.norm" "$OUT/$b.head.norm" >/dev/null; then
    echo "SASS IDENTICAL  $b   ($nf functions, $ni SASS instructions)"
  else
    echo "SASS DIFFERS    $b   ($nf functions)"
    diff "$OUT/$b.base.norm" "$OUT/$b.head.norm" | head -30
  fi
done
