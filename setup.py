from setuptools import setup

# replace tensorflow with tensorflow-gpu on GPU-compatible machine
setup(
    name="weight_uncertainty_nn",
    version="0.0.1",
    install_requires=["numpy", "pandas", "tqdm", "matplotlib", "tensorflow"]
)