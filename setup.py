"""Setup script for NTRO-SRM (Sentinel-2 Multi-Spectral Super-Resolution Mapping)."""

from pathlib import Path
from setuptools import find_packages, setup

# Read the contents of README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""

# Read requirements
req_path = Path(__file__).parent / "requirements.txt"
install_requires = []
if req_path.is_file():
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            install_requires.append(line)

setup(
    name="ntro-srm",
    version="0.1.0",
    description="Deep Learning Based Super-Resolution Mapping from Medium Resolution Satellite Imageries (Sentinel-2 10m -> 2.5m GSD)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NTRO-SRM Contributors",
    license="MIT",
    url="https://github.com/your-username/NTRO-SRM",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/NTRO-SRM/issues",
        "Documentation": "https://github.com/your-username/NTRO-SRM#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "ntro-srm-web=ntro_srm.web.app:run_server",
        ],
    },
)
