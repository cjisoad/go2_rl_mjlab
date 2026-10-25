# Approved Go2 migration design

User approved on 2026-10-23: adapt all four methods to the installed mjlab/RSL-RL environment, retain Tracking plus its humanoid assets and Go2 Flat/Rough training, remove unrelated robots and deployment programs. Standup uses current V2.1. User clarified backflip uses V1.6 (not the originally requested 1.6.1).

DreamWaQ and AMP include their algorithm implementations and required demonstrations. Each method is a separate registered task. Preserve source mathematical semantics while adapting scene, contacts, actuator and observation APIs to mjlab. No Isaac runtime dependency or alternate RSL package import hacks.

Acceptance: source provenance recorded, meaningful numerical tests, actual WSL CUDA simulation and short training updates, checkpoint restore and policy inference/export. Short validation does not establish training convergence or sim-to-real quality.
