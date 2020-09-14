from setuptools import setup, find_packages
import sys
from setuptools.command.install import install as _install

# class Install(_install):
    # def run(self):
        # _install.do_egg_install(self)
        # import nltk
        # nltk.download("punkt")

setup(
    name='visibility',
    version='0.2',
    # cmdclass={'install':Install},
    # package_data={
        # "utils":["utils/*.json"]
        # "utils":["utils/*.json"],
        # ".":["*.json",'captions_train2014.json']
    # },
    # include_package_data=True,
    data_files=[('csv',['vocabulary.csv'])],
    install_requires=[
        'opencv-python',
        'numpy',
        # 'tensorflow==1.14.0',
        'tqdm',
        'matplotlib',
        'scikit-image',
        # 'scikit-learn',
        # 'wget'
        # 'nltk'
        # 'pickle'
    ],
    scripts=['base_model.py','nn.py','config.py','dataset.py','misc.py','vocabulary.py','visibility.py','model.py','coco.py'],
    # setup_requires=['nltk']
)

# if 'install' in sys.argv:
# import nltk
# nltk.download('punkt')
