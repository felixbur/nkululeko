with open("nkululeko/constants.py", "r") as f:
    for line in f:
        if line.startswith("VERSION"):
            VERSION = line.split("=")[1].strip().strip("\"'")
            break

from setuptools import find_packages, setup

setup(
    name="nkululeko",
    version=VERSION,
    description="Machine learning audio prediction experiments based on templates",
    author="Felix Burkhardt",
    author_email="fxburk@gmail.com",
    url="https://github.com/felixbur/nkululeko",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "audeer>=1.0.0",
        "audformat>=1.3.1",
        "audinterface>=1.0.0",
        "audiofile>=1.0.0",
        "audiomentations==0.31.0",
        "audmetric>=1.0.0",
        "audonnx>=0.7.0",
        "confidence-intervals>=0.0.2",
        "datasets>=2.0.0",
        "imageio>=2.0.0",
        "matplotlib>=3.0.0",
        "numpy>=1.20.0",
        "opensmile>=2.0.0",
        "pandas>=1.0.0",
        "praat-parselmouth>=0.4.0",
        "scikit_learn>=1.0.0",
        "scipy>=1.0.0",
        "seaborn>=0.11.0",
        "sounddevice>=0.4.0",
        "transformers>=4.0.0",
        "umap-learn>=0.5.0",
        "xgboost>=1.0.0",
        "pylatex>=1.0.0",
    ],
    extras_require={
        "torch": [
            "torch>=1.0.0",
            "torchvision>=0.10.0",
            "torchaudio>=0.10.0",
        ],
        "torch-cpu": [
            "torch>=1.0.0",
            "torchvision>=0.10.0",
            "torchaudio>=0.10.0",
        ],
        "torch-nightly": [
            "torch",
            "torchvision",
            "torchaudio",
        ],
        "spotlight": [
            "renumics-spotlight>=1.6.13",
            "sliceguard>=0.0.35",
        ],
        "tensorflow": [
            "tensorflow>=2.0.0",
            "tensorflow_hub>=0.12.0",
        ],
        "all": [
            "torch>=1.0.0",
            "torchvision>=0.10.0",
            "torchaudio>=0.10.0",
            "renumics-spotlight>=0.1.0",
            "sliceguard>=0.1.0",
            "tensorflow>=2.0.0",
            "tensorflow_hub>=0.12.0",
            "shap>=0.40.0",
            "imblearn>=0.0.0",
            "cylimiter>=0.0.1",
            "audtorch>=0.0.1",
            "splitutils>=0.0.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "nkululeko.explore=nkululeko.explore:main",
            "nkululeko.nkululeko=nkululeko.nkululeko:main",
            "nkululeko.augment=nkululeko.augment:main",
            "nkululeko.demo=nkululeko.demo:main",
            "nkululeko.export=nkululeko.export:main",
            "nkululeko.predict=nkululeko.predict:main",
            "nkululeko.resample=nkululeko.resample:main",
            "nkululeko.segment=nkululeko.segment:main",
            "nkululeko.testing=nkululeko.testing:main",
            "nkululeko.ensemble=nkululeko.ensemble:main",
        ],
    },
)
