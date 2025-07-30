from setuptools import find_packages, setup

setup(
    name="triangle_rasterization",
    version="0.0.1",
    description="PyTorch-compatible differentiable triangle rasterization.",
    author="David Charatan",
    author_email="charatan@mit.edu",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["slangtorch", "torch"],
)
