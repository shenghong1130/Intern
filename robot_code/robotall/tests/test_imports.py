import importlib


def test_package_imports_and_exports():
    pkg = importlib.import_module("robotall")
    assert callable(pkg.act)
    assert callable(pkg.register_robot)
    assert callable(pkg.send_request)
