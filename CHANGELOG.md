# Changelog

## Unreleased

### Fixed

- Pin NumPy 1.26.4 for MuJoCo 2.3.7 rendering and array compatibility.
- Spawn parallel simulation workers and create their shared buffers with the same multiprocessing context.

- Load evaluation data from configured external directories and exclude it from source distributions and wheels.
- Use config_libero_pro.yaml for LIBERO-Pro paths.
- Check assets and task files before starting rollouts.
- Resolve perturbation data paths through the simulator configuration.
- Dataset utilities share the benchmark path resolver.
