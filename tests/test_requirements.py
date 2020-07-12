"""Test availability of required packages."""

import unittest
import os
from pathlib import Path

import pkg_resources

parent_path = Path(__file__).parent.parent
requirements_path_os = os.path.join(parent_path,"requirements.txt")
print("Parent       Path = {}".format(parent_path))
print("Requirements Path = {}".format(requirements_path_os))
_REQUIREMENTS_PATH = Path(requirements_path_os)


class TestRequirements(unittest.TestCase):
    """Test availability of required packages."""

    def test_requirements(self):
        """Test that each required package is available."""
        # Ref: https://stackoverflow.com/a/45474387/
        requirements = pkg_resources.parse_requirements(_REQUIREMENTS_PATH.open())
        for requirement in requirements:
            requirement = str(requirement)
            print("Check Requirement ==> {}".format(requirement))
            with self.subTest(requirement=requirement):
                pkg_resources.require(requirement)
