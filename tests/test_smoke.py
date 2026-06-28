"""Smoke test: the package and its subpackages import cleanly."""


def test_package_imports():
    import cs336
    import cs336.data
    import cs336.models
    import cs336.tokenizer
    import cs336.training
    import cs336.utils

    assert cs336 is not None
