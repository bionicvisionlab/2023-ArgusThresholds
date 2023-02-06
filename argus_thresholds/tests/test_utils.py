from datetime import datetime
import numpy as np
import numpy.testing as npt
from .. import utils


def test_str2date():
    for Ymd in [(1985, 9, 30), (2000, 1, 1), (2018, 8, 31)]:
        date = datetime(*Ymd)
        npt.assert_equal(date, utils.str2date(date.strftime("%Y-%m-%d")))
    npt.assert_equal(np.isnan(utils.str2date(np.nan)), True)


def test_str2timestamp():
    # Warning: This will give different results on Windows because of mktime
    npt.assert_equal(utils.str2timestamp("1990-01-01"), 631152000.0)
    npt.assert_equal(utils.str2timestamp("2011-12-01"), 1322697600.0)
    npt.assert_equal(np.isnan(utils.str2timestamp(np.nan)), True)


def test_years_btw_dates():
    npt.assert_equal(utils.years_btw_dates(datetime(2018, 3, 14),
                                           datetime(2018, 3, 14)), 0)
    npt.assert_equal(utils.years_btw_dates(datetime(2018, 3, 14),
                                           datetime(1999, 3, 14)), 19)
    npt.assert_equal(utils.years_btw_dates(datetime(2018, 3, 14),
                                           datetime(1999, 3, 11)), 19)
    npt.assert_equal(utils.years_btw_dates(datetime(2018, 3, 14),
                                           datetime(1999, 3, 15)), 18)
    npt.assert_equal(utils.years_btw_dates(datetime(2018, 3, 14),
                                           datetime(2019, 3, 14)), -1)
    npt.assert_equal(np.isnan(utils.years_btw_dates(datetime(2018, 3, 14),
                                                    np.nan)), True)


def test_days_btw_dates():
    npt.assert_equal(utils.days_btw_dates(datetime(2018, 3, 14),
                                          datetime(2018, 3, 14)), 0)
    npt.assert_equal(utils.days_btw_dates(datetime(2018, 3, 14),
                                          datetime(2018, 3, 1)), 13)
    npt.assert_equal(utils.days_btw_dates(datetime(2018, 3, 14),
                                          datetime(2018, 3, 13)), 1)
    npt.assert_equal(utils.days_btw_dates(datetime(2018, 3, 14),
                                          datetime(2018, 2, 27)), 15)
    npt.assert_equal(utils.days_btw_dates(datetime(2018, 3, 14),
                                          datetime(2017, 12, 30)), 74)
    npt.assert_equal(utils.days_btw_dates(datetime(2018, 3, 14),
                                          datetime(2018, 3, 20)), -6)
    npt.assert_equal(np.isnan(utils.days_btw_dates(datetime(2018, 3, 14),
                                                   np.nan)), True)
