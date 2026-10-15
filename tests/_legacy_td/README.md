# Frozen numerical reference

These six modules preserve the learning calculations, their private validation, and the four
online window schedules from commit `dc2ecdc4099f68b17e6b1826ec6fcaeb9b25593d`.
Only import wiring was changed: references import each other rather than `examples`/`runpy`;
environment setup and CLI experiments are omitted. They never delegate to the new Learners.

`test_promoted_learners.py` starts the old reference and public API from independent identical
models/seeds and compares targets, signed errors, every table write, actual online episode
statistics, environment RNG and Torch RNG, including truncation tails.
This duplication exists only as a frozen regression oracle, outside the installed package.
Production examples delegate to the public Learners. Do not update these references merely
to make a changed algorithm pass.
