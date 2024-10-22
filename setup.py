# from nkululeko.constants import VERSION
from setuptools import setup

setup(
    use_scm_version=True,
    # version=VERSION,
    entry_points={
        "console_scripts": [
            # 'nkululeko=nkululeko.nkululeko:main',
            "nkululeko.explore=nkululeko.explore:main",
            "nkululeko.nkululeko=nkululeko.nkululeko:main",
            "nkululeko.augment=nkululeko.augment:main",
            "nkululeko.demo=nkululeko.demo:main",
            "nkululeko.export=nkululeko.export:main",
            "nkululeko.predict=nkululeko.predict:main",
            "nkululeko.resample=nkululeko.resample:main",
            "nkululeko.segment=nkululeko.segment:main",
            "nkululeko.test=nkululeko.test:main",
            "nkululeko.ensemble=nkululeko.ensemble:main",
        ],
    },
)
