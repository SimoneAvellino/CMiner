from setuptools import setup, find_packages


def parse_requirements(file):
    with open(file, "r") as f:
        return [line.strip() for line in f if line.strip()]


setup(
    name="CMiner",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=parse_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "CMiner = CMiner.main:main_function",
        ],
    },
)
