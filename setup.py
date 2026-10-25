"""Install the training tasks and local robot/motion resources."""

from setuptools import find_packages, setup

setup(
    name="go2_rl_mjlab",
    version="0.1.0",
    packages=find_packages(include=("src", "src.*")),
    include_package_data=True,
    exclude_package_data={"": ["**/__pycache__/*", "**/*.pyc", "**/*.pyo"]},
    package_data={"src": ["assets/**/*"]},
    install_requires=["mjlab==1.2.0", "mujoco-warp==3.5.0"],
)
