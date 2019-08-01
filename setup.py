from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='morpher',
      version='0.0.1',
      description='Modeling of Outcome and Risk Prediction in Health Research',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
      ],
      url='https://github.com/hpi-dhc/morpher',
      author='Harry Cruz',
      author_email='harrycruz@gmail.com',
      license='MIT',
      packages=['morpher', 'morpher.jobs'],
      tests_require=['pytest'],
      setup_requires=['pytest-runner'],
      install_requires=[
          'markdown',
          'scikit-learn',
          'pandas',
          'numpy',
          'lime',
          'scipy',
          'matplotlib==3.1.0',
          'sklearn_pandas',
          'fancyimpute',
          'imbalanced-learn',
          'statsmodels',
          'jsonpickle==1.2'
      ],
      #specifying a given version: 'chardet==3.0.4',
      zip_safe=False)
