#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

setup(
    name='bank_customer_complaint_analysis',
    packages=find_packages(
        include=['bank_customer_complaint_analysis', 'bank_customer_complaint_analysis.*']
    ),
    test_suite='tests',
    version="0.1.0",
)
