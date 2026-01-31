"""Unit tests for __version__.py."""

import diffpy.snmf  # noqa


def test_package_version():
    """Ensure the package version is defined and not set to the initial
    placeholder."""
    assert hasattr(diffpy.snmf, "__version__")
    assert diffpy.snmf.__version__ != "0.0.0"
