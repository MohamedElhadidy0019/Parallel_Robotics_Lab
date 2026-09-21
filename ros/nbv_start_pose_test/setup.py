from setuptools import find_packages, setup

setup(
    name="nbv_start_pose_test",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/nbv_start_pose_test"]),
        ("share/nbv_start_pose_test", ["package.xml", "README.md"]),
        ("share/nbv_start_pose_test/launch", ["launch/start_pose.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="student",
    maintainer_email="student@todo.todo",
    description="Standalone live-URDF viewer and cuRobo start-pose test",
    license="Apache-2.0",
    entry_points={"console_scripts": ["start_pose_node = nbv_start_pose_test.node:main"]},
)
