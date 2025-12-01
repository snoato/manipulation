from setuptools import setup, find_packages

setup(
    name="manipulation",
    version="0.1.0",
    description="Robotics manipulation package built on top of MuJoCo and MINK",
    author="Daniel Swoboda, Chair of Machine Learning and Reasoning (i6), RWTH Aachen University",
    packages=find_packages(),
    install_requires=[
        "mujoco>=3.0.0",
        "numpy>=1.20.0",
        "mink>=0.0.1",
        "loop-rate-limiters>=1.0.0",
    ],
    python_requires=">=3.8",
    package_data={
        "manipulation": [
            "environments/assets/franka_emika_panda/**/*",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
