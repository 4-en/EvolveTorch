# setup.py

from setuptools import setup, find_packages



deps = [
    "torch",
    "matplotlib",
    "pygame",
    "tqdm"
]

setup(
    name="EvolveTorch",
    version="0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=deps,
    entry_points={
        'console_scripts': [
            'evolvetorch-test = evolvetorch.test:main',
        ]
    },
)


