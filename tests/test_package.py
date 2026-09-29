import hub


def test_package_is_importable():
    assert hub.__name__ == "hub"
