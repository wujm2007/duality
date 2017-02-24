from setuptools import setup


setup(name='duality',
      version='0.1',
      description='the kernel part of a virtual character',
      url='http://github.com/ravenSanstete/duality',
      author='morino',
      author_email='sansmori@gmail.com',
      license='MIT',
      packages=['duality'],
      install_requires=[
          'nltk',
          'spacy',
          'vaderSentiment',
          'numpy',
          'python-Levenshtein',
          'tensorflow',
          'keras',
          'pandas',
          'pycurl'
      ],
      zip_safe=False
)
