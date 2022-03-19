from setuptools import setup
from pathlib import Path


source_root = Path(".")
with (source_root / "requirements.txt").open(encoding="utf8") as f:
    requirements = f.readlines()

setup(
    name="erpack", 
    version="0.0.1",
    author="Eric Aderne",
    author_email="eeaderne@gmail.com",
    description='Data Science Package',
    url="git@github.com:estader/ericpackage",
    install_requires=requirements,
    license='unlicense',
    package_dir= {
        'erpack': '',
        'erpack.automl': 'erpack/automl',
        'erpack.data_cleaning': 'erpack/data_cleaning',
        'erpack.error_analysis': 'erpack/error_analysis',
        'erpack.exploratory_analysis': 'erpack/exploratory_analysis',
        'erpack.feature_engineering': 'erpack/feature_engineering',
        'erpack.feature_selection': 'erpack/feature_selection',
        'erpack.monitoring': 'erpack/monitoring',
        'erpack.production_pipeline': 'erpack/production_pipeline',
        'erpack.quality_check': 'erpack/quality_check',
        'erpack.result_interpretation': 'erpack/result_interpretation',
        'erpack.train_test_split': 'erpack/train_test_split',},
    packages = ['erpack',
                'erpack.automl',
                'erpack.data_cleaning',
                'erpack.error_analysis',
                'erpack.exploratory_analysis',
                'erpack.feature_engineering',
                'erpack.feature_selection',
                'erpack.monitoring',
                'erpack.production_pipeline',
                'erpack.quality_check',
                'erpack.result_interpretation',
                'erpack.train_test_split'],
    zip_safe=False,
    python_requires='>=3.6'
)



