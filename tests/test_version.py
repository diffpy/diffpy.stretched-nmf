"""Unit tests for __version__.py."""

import diffpy.stretched_nmf  # noqa


def test_package_version():
    """Ensure the package version is defined and not set to the initial
    placeholder."""
    assert hasattr(diffpy.stretched_nmf, "__version__")
    assert diffpy.stretched_nmf.__version__ != "0.0.0"
