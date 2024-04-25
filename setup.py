#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/12/2023 5:03 PM
# @Author  : xiaomanl
# @File    : setup.py
# @Software: PyCharm
from setuptools import setup, find_packages

setup(
    name='performance-prediction',
    version='1.0.0',
    description='performance prediction',
    author='Xiaoman',
    author_email='xiaoman.liu@intel.com',
    entry_points={
    'console_scripts': ['performance-prediction=module:function'],
    },
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.2.0',
        'tqdm>=4.60.0',
        'protobuf>=3.17.3',
        'PyYAML>=5.4.1',
        'scikit_learn>=0.24.2',
        'packaging>=21.0',
        'jsonlines>=2.0.0',
        'xgb>=1.4.2',
        'plotly>=5.3.1',
        'cufflinks>=0.17.3',
        'pydotplus',
        'kaleido',
        'dtreeviz',
        "tensorflow",
        "openpyxl",
        "mysql-connector-python"
    ],
)
#
# setup(
#     name='performance-prediction',
#     version='original',
#     description='performance prediction',
#     long_description = open('README.md', encoding='utf-8').read(),
#     long_description_content_type = 'text/markdown',
#     author='Intel',
#     author_email='xiaoman.liu@intel.com',
#     url='https://github.com/xiaoman-liu/performance-prediction',
#     license='MIT',
#     project_urls={
#         'Data of models': 'https://github.com/xiaoman-liu/performance-prediction',
#     },
#     classifiers = [
#             'License :: OSI Approved :: MIT License',
#             'Operating System :: MacOS :: MacOS X',
#             'Operating System :: Microsoft :: Windows :: Windows 11',
#             'Operating System :: POSIX :: Linux',
#             'Programming Language :: Python :: 3 :: Only',
#             'Topic :: Scientific/Engineering :: Artificial Intelligence',
#         ],
#     packages=find_packages(),
#     package_data={
#         'HPP': [
#             'config/*.yaml',
#             'dataset/*.csv',
#         ],
#     },
#     entry_points={
#         'console_scripts': ['performance-prediction=module:function'],
#     },
#     install_requires=[
#         'numpy',
#         'pandas',
#         'tqdm',
#         'protobuf',
#         'PyYAML',
#         'scikit_learn',
#         'packaging',
#         'jsonlines',
#         'xgb',
#         'plotly',
#         'cufflinks'
#     ],
# )
# # #        'numpy>=1.20.0',
# #         'pandas>=1.2.0',
# #         'tqdm>=4.60.0',
# #         'protobuf>=3.17.3',
# #         'PyYAML>=5.4.1',
# #         'scikit_learn>=0.24.2',
# #         'packaging>=21.0',
# #         'jsonlines>=2.0.0',
# #         'xgb>=1.4.2',
# #         'plotly>=5.3.1',
# #         'cufflinks>=0.17.3'
#pip install virtualenv
#virtualenv myenv
#source myenv/bin/activate
#pip install .