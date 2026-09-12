import os
from glob import glob
from setuptools import setup, find_packages

package_name = "steve_sim_prep"

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths

extra_files = []
for f in package_files("models"):
    dest = os.path.join("share", package_name, os.path.dirname(f))
    extra_files.append((dest, [f]))

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ] + extra_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="djyjyh",
    maintainer_email="dev@todo.todo",
    description="Standalone decoupled simulation environment preparer for Steve robot in Gazebo",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "preparer_node = steve_sim_prep.preparer_node:main",
        ],
    },
)
