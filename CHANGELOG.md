# Changelog

## Unreleased

### Fixed

- Use NumPy 1.26.4 for compatibility with MuJoCo array bindings.
- Spawn parallel simulation workers and create their shared buffers with the same multiprocessing context.
- Load evaluation data from configured external directories and exclude it from source distributions and wheels.
- Use config_libero_plus.yaml for LIBERO-plus paths.
- Check assets and task files before starting rollouts.
- Read task instructions when constructing the selected suite.
- Resolve base-task and perturbed initial-state filenames consistently.
