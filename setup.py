from glob import glob
from setuptools import setup, find_packages

setup(
    name="line_follower",
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/line_follower"]),
        ("share/line_follower", ["package.xml"]),
        ("share/line_follower/config", glob("config/*.yaml")),
        ("share/line_follower/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Shanmukh Narayana Chaganti",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
        "camera_sim = line_follower.nodes.camera_sim_node:main",
        "line_perception = line_follower.nodes.perception_node:main",
        "line_control = line_follower.nodes.control_node:main",
    ]},
)
