import importlib
import os
import pkgutil

import pytest

import nkululeko.autopredict


def _autopredict_module_names():
    pkg_path = os.path.dirname(nkululeko.autopredict.__file__)
    names = []
    for module in pkgutil.iter_modules([pkg_path]):
        if module.name == "tests":
            continue
        if module.ispkg:
            continue
        names.append(module.name)
    return sorted(names)


@pytest.mark.parametrize("module_name", _autopredict_module_names())
def test_autopredict_module_imports(module_name):
    full_name = f"{nkululeko.autopredict.__name__}.{module_name}"
    try:
        importlib.import_module(full_name)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("nkululeko."):
            raise
        pytest.skip(f"Optional dependency missing for {full_name}: {exc}")
    except ImportError as exc:
        pytest.skip(f"Optional dependency missing for {full_name}: {exc}")
