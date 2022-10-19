# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name="lti_llm_client",
    version="0.0.1",
    author="Patrick Fernandes",
    author_email="pfernand@cs.cmu.edu",
    url="",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.7",
    setup_requires=[],
    install_requires=[
        'requests'
    ],
)