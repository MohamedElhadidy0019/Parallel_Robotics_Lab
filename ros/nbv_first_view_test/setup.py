import os
from glob import glob
from setuptools import setup, find_packages

package_name = "nbv_first_view_test"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="student",
    maintainer_email="student@todo.todo",
    description="Reach the inspection start pose and show the first RGB-D frame in Rerun.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "first_view_node = nbv_first_view_test.node:main",
        ],
    },
)
