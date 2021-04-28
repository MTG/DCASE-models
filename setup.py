import setuptools
from importlib.machinery import SourceFileLoader

version = SourceFileLoader('dcase_models.version',
                           'dcase_models/version.py').load_module()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DCASE-models",
    version=version.version,
    author="Pablo Zinemanas",
    author_email="pablo.zinemanas@upf.edu",
    description="Python library for rapid prototyping of environmental sound analysis systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pzinemanas/DCASE-models",
    download_url='http://github.com/pzinemanas/DCASE-models/releases',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy==1.17.4',
        'pandas==0.25.3',
        'SoundFile==0.10.3.post1',
        'PyYAML==5.2',
        'numba==0.48.0',
        'librosa==0.7.1',
        'openl3==0.3.1',
        'scikit-learn==0.22.1',
        'keras==2.2.4',
       # 'tensorflow-gpu==1.12.0',
        'autopool==0.1.0',
        'wget==3.2',
        'sox==1.3.7',
        'sed_eval==0.2.1',
    ],
    extras_require={
        'docs': ['numpydoc', 'sphinx!=1.3.1', 'sphinx_rtd_theme'],
        'tests': ['pytest >= 5.4.3', 'pytest-cov >= 2.11.1'],
        'visualization': [
            'plotly==4.5.0',
            'dash==1.8.0',
            'dash_bootstrap_components==0.8.1',
            'dash_audio_components==1.2.0',
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
