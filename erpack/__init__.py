"""Inicialização do pacote."""

from .quality_check import quality_check
from .train_test_split import train_test_split
from .data_cleaning import data_cleaning
from .feature_engineering import feature_engineering
from .exploratory_analysis import exploratory_analysis
from .feature_selection import feature_selection
from .automl import automl
from .error_analysis import error_analysis
from .result_interpretation import result_interpretation
from .monitoring import monitoring
from .production_pipeline import production_pipeline

__version__ = '1.0.0'
__dist_name__ = 'erpack - Python Data Science Package made by Eric Aderne'
