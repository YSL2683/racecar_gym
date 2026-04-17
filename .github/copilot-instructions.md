# Copilot Instructions

## Build, test, and lint commands

- Install the package in editable mode from the repo root: `pip install -e .`
- Wrapper tests use `unittest`: `PYTHONPATH=. python -m unittest discover tests/wrappers`
- Run a single wrapper test: `PYTHONPATH=. python -m unittest tests.wrappers.action_repeat.ActionRepeatTest.test_single_agent_action_repeat`
- PettingZoo API checks are executable modules, not `unittest` cases:
  - `PYTHONPATH=. python -m tests.pettingzoo_api.api_test`
  - `PYTHONPATH=. python -m tests.pettingzoo_api.parallel_api_test`
- No dedicated lint configuration or lint command is checked into this repository.

## High-level architecture

- `racecar_gym.envs.scenarios` is the composition layer. It loads a scenario YAML into `ScenarioSpec`, builds `Agent` objects, and delegates world/vehicle construction to the Bullet providers.
- Scenario YAML in `scenarios/*.yml` only chooses the world, agent ids, vehicle name, enabled sensors/actuators, and task parameters. The actual vehicle and scene implementations live under `models/vehicles/<name>/` and `models/scenes/<track>/`.
- `racecar_gym.bullet.providers` is the main factory layer. It validates that requested sensors and actuators exist in the selected vehicle config, wraps sensors with fixed-timestep sampling, resolves scene asset paths, and auto-downloads missing track assets on first use.
- `racecar_gym.bullet.world.World` owns the PyBullet simulation loop, starting-position strategies, map-derived race state (`progress`, `checkpoint`, `lap`, `rank`, collisions), and rendering.
- The Gymnasium and PettingZoo APIs share the same simulation core. `SingleAgentRaceEnv` and `MultiAgentRaceEnv` both drive the scenario/world objects directly, while `racecar_gym.envs.pettingzoo_api` is a thin `ParallelEnv` adapter over `MultiAgentRaceEnv`.
- Gym environment registration happens as an import side effect in `racecar_gym.envs.gym_api`. Importing that module registers `SingleAgent<Track>-v0` and `MultiAgent<Track>-v0` for every YAML file in `scenarios/`, plus the generic `SingleAgentRaceEnv-v0` and `MultiAgentRaceEnv-v0`.

## Key conventions

- Multi-agent observations and actions are `Dict` spaces keyed by agent id. Single-agent environments expose just the single agent space, but they still inject a scalar `time` field into observations during `reset()` and `step()`.
- Tasks are string-selected from a registry. To add a new task, implement a `Task` subclass and register it in `racecar_gym/tasks/__init__.py`; scenarios refer to it by `task_name`.
- Reset behavior is controlled by `options["mode"]` and the exact supported values come from `racecar_gym/bullet/world.py`: `grid`, `random`, `random_bidirectional`, and `random_ball`.
- Rendering mode is not cosmetic. `human` enables the PyBullet GUI and follow-camera updates; non-human modes run in DIRECT/EGL and return RGB arrays through `world.render(...)`.
- Scenario files should only request sensors and actuators already declared in the selected vehicle YAML. The loader enforces this as a strict subset check.
- Track assets may be missing in a fresh checkout. `load_world(...)` downloads a scene zip automatically when the expected `models/scenes/<track>/<track>.yml` file is absent.
- Tests are split across styles: wrapper tests are `unittest` modules, while PettingZoo compliance checks are standalone executable scripts under `tests/pettingzoo_api/`.
