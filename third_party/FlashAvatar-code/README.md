# FlashAvatar FLAME Assets

This directory vendors the `flame/` folder from:

- Repository: https://github.com/USTC3DV/FlashAvatar-code
- Commit: `2d4cb25ca9166b795c03692aec12114d595eff15`

RuiHeadStudio currently uses only `flame/FlameMesh.obj` from this vendor copy.
Its face topology keeps the standard FLAME 5023 vertices and adds mouth-closure
faces used by FlashAvatar for interior mouth coverage.

This topology change is intended for new training runs. Existing RuiHeadStudio
checkpoints store per-Gaussian face bindings and local coordinates from the
previous FLAME face topology, so resuming old checkpoints with this branch is
not guaranteed to be semantically compatible.

The remaining files are kept with the vendored folder for provenance. They are
not imported by the current RuiHeadStudio pipeline.

License note: these files may include FLAME-derived assets. Review FLAME and
FlashAvatar redistribution terms before publishing this repository or packaging
these assets outside internal experiments.
