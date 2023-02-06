import os
import numpy as np
import pandas as pd
import numpy.testing as npt
import pytest

from ..core import load_subjects


def get_df_subjects():
    data = [
        {'subject': '11-001', 'birth_date': '1960-01-01', 'gender': 'F'},
        {'subject': '12-012', 'birth_date': '1959-01-01', 'gender': 'M'},
        {'subject': '13-003', 'birth_date': '1943-01-01', 'gender': 'F'},
        {'subject': '14-111', 'birth_date': '1966-01-01', 'gender': 'M'}
    ]
    return pd.DataFrame(data).set_index('subject')


def get_df_data():
    data = [
        {'date': '2010-03-05', 'th': 101},
        {'date': '2014-06-01', 'th': 144},
        {'date': '2012-04-15', 'th': 45},
        {'date': '2013-01-24', 'th': 665},
        {'date': '2010-06-19', 'th': 123},
    ]
    return pd.DataFrame(data)


def test_load_subjects():
    with pytest.raises(TypeError):
        load_subjects([1, 2, 3])

    csvfile = "test_subjects.csv"
    dfsave = get_df_subjects()
    dfsave.to_csv(csvfile)
    dfload = load_subjects(csvfile)
    npt.assert_equal(dfsave.equals(dfload), True)
    os.remove(csvfile)
