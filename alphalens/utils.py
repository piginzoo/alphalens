#
# Copyright 2018 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import re
import warnings

from IPython.display import display
from pandas.tseries.offsets import CustomBusinessDay, Day, BusinessDay
from scipy.stats import mode


class NonMatchingTimezoneError(Exception):
    pass


class MaxLossExceededError(Exception):
    pass


def rethrow(exception, additional_message):
    """
    Re-raise the last exception that was active in the current scope
    without losing the stacktrace but adding an additional message.
    This is hacky because it has to be compatible with both python 2/3
    """
    e = exception
    m = additional_message
    if not e.args:
        e.args = (m,)
    else:
        e.args = (e.args[0] + m,) + e.args[1:]
    raise e


def non_unique_bin_edges_error(func):
    """
    Give user a more informative error in case it is not possible
    to properly calculate quantiles on the input dataframe (factor)
    """
    message = """

    An error occurred while computing bins/quantiles on the input provided.
    This usually happens when the input contains too many identical
    values and they span more than one quantile. The quantiles are choosen
    to have the same number of records each, but the same value cannot span
    multiple quantiles. Possible workarounds are:
    1 - Decrease the number of quantiles
    2 - Specify a custom quantiles range, e.g. [0, .50, .75, 1.] to get unequal
        number of records per quantile
    3 - Use 'bins' option instead of 'quantiles', 'bins' chooses the
        buckets to be evenly spaced according to the values themselves, while
        'quantiles' forces the buckets to have the same number of records.
    4 - for factors with discrete values use the 'bins' option with custom
        ranges and create a range for each discrete value
    Please see utils.get_clean_factor_and_forward_returns documentation for
    full documentation of 'bins' and 'quantiles' options.

"""

    def dec(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if 'Bin edges must be unique' in str(e):
                rethrow(e, message)
            raise
    return dec


@non_unique_bin_edges_error
def quantize_factor(factor_data,
                    quantiles=5,
                    bins=None,
                    by_group=False,
                    no_raise=False,
                    zero_aware=False):
    """
    Computes period wise factor quantiles.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
            -----------------------------------------------------
                date    |    asset   |  factor |  forward returns
            -----------------------------------------------------
                        |   AAPL     |   0.5   |  0.12
                        -----------------------------------------
                        |   BA       |  -1.1   |  0.02
                        -----------------------------------------
            2014-01-01  |   CMG      |   1.7   |  0.031
                        -----------------------------------------

    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Only one of 'quantiles' or 'bins' can be not-None
    by_group : bool, optional
        If True, compute quantile buckets separately for each group.
    no_raise: bool, optional
        If True, no exceptions are thrown and the values for which the
        exception would have been thrown are set to np.NaN
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.

    Returns
    -------
    factor_quantile : pd.Series
        Factor quantiles indexed by date and asset.
            -------------------------------------------
                date    |    asset   |  factor_quantile
            -------------------------------------------
                        |   AAPL     |   1
                        -------------------------------
                        |   BA       |   1
                        -------------------------------
            2014-01-01  |   CMG      |   2
                        -------------------------------
    """
    if not ((quantiles is not None and bins is None) or
            (quantiles is None and bins is not None)):
        raise ValueError('Either quantiles or bins should be provided')

    if zero_aware and not (isinstance(quantiles, int)
                           or isinstance(bins, int)):
        msg = ("zero_aware should only be True when quantiles or bins is an"
               " integer")
        raise ValueError(msg)

    def quantile_calc(x, _quantiles, _bins, _zero_aware, _no_raise):
        try:
            if _quantiles is not None and _bins is None and not _zero_aware:
                return pd.qcut(x, _quantiles, labels=False) + 1
            elif _quantiles is not None and _bins is None and _zero_aware:
                pos_quantiles = pd.qcut(x[x >= 0], _quantiles // 2,
                                        labels=False) + _quantiles // 2 + 1
                neg_quantiles = pd.qcut(x[x < 0], _quantiles // 2,
                                        labels=False) + 1
                return pd.concat([pos_quantiles, neg_quantiles]).sort_index()
            elif _bins is not None and _quantiles is None and not _zero_aware:
                return pd.cut(x, _bins, labels=False) + 1
            elif _bins is not None and _quantiles is None and _zero_aware:
                pos_bins = pd.cut(x[x >= 0], _bins // 2,
                                  labels=False) + _bins // 2 + 1
                neg_bins = pd.cut(x[x < 0], _bins // 2,
                                  labels=False) + 1
                return pd.concat([pos_bins, neg_bins]).sort_index()
        except Exception as e:
            if _no_raise:
                return pd.Series(index=x.index)
            raise e

    grouper = [factor_data.index.get_level_values('date')]
    if by_group:
        grouper.append('group')

    factor_quantile = factor_data.groupby(grouper)['factor'] \
        .apply(quantile_calc, quantiles, bins, zero_aware, no_raise)
    factor_quantile.name = 'factor_quantile'

    return factor_quantile.dropna()


def infer_trading_calendar(factor_idx, prices_idx):
    """
    TODO：这里面的代码里提到了holiday，不知道会不会受美国节假日影响，不过看代码似乎没有
    Infer the trading calendar from factor and price information.

    Parameters
    ----------
    factor_idx : pd.DatetimeIndex
        The factor datetimes for which we are computing the forward returns
    prices_idx : pd.DatetimeIndex
        The prices datetimes associated withthe factor data

    Returns
    -------
    calendar : pd.DateOffset
    """
    full_idx = factor_idx.union(prices_idx)

    traded_weekdays = []
    holidays = []

    days_of_the_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for day, day_str in enumerate(days_of_the_week):

        weekday_mask = (full_idx.dayofweek == day)

        # drop days of the week that are not traded at all
        if not weekday_mask.any():
            continue
        traded_weekdays.append(day_str)

        # look for holidays
        used_weekdays = full_idx[weekday_mask].normalize()
        all_weekdays = pd.date_range(full_idx.min(), full_idx.max(),
                                     freq=CustomBusinessDay(weekmask=day_str)
                                     ).normalize()
        _holidays = all_weekdays.difference(used_weekdays)
        _holidays = [timestamp.date() for timestamp in _holidays]
        holidays.extend(_holidays)

    traded_weekdays = ' '.join(traded_weekdays)
    return CustomBusinessDay(weekmask=traded_weekdays, holidays=holidays)


def compute_forward_returns(factor,
                            prices,
                            periods=(1, 5, 10),
                            filter_zscore=None,
                            cumulative_returns=True):
    """
    Finds the N period forward returns (as percent change) for each asset
    provided.

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.

        - See full explanation in utils.get_clean_factor_and_forward_returns

    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float, optional
        Sets forward returns greater than X standard deviations
        from the the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier(统计异常值) filtering incorporates(体现) lookahead(先行的) bias.
    cumulative_returns : bool, optional
        If True, forward returns columns will contain cumulative returns.
        Setting this to False is useful if you want to analyze how predictive
        a factor is for a single forward day.

    Returns
    -------
    forward_returns : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by timestamp (level 0) and asset
        (level 1), containing the forward returns for assets.
        Forward returns column names follow the format accepted by
        pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc).
        'date' index freq property (forward_returns.index.levels[0].freq)
        will be set to a trading calendar (pandas DateOffset) inferred
        from the input data (see infer_trading_calendar for more details).
    """

    # level0是日期
    factor_dateindex = factor.index.levels[0]
    if factor_dateindex.tz != prices.index.tz:
        raise NonMatchingTimezoneError("The timezone of 'factor' is not the "
                                       "same as the timezone of 'prices'. See "
                                       "the pandas methods tz_localize and "
                                       "tz_convert.")
    # 感觉就是把因子和价格的日期的并集整理出来
    freq = infer_trading_calendar(factor_dateindex, prices.index)

    factor_dateindex = factor_dateindex.intersection(prices.index)

    if len(factor_dateindex) == 0:
        raise ValueError("Factor and prices indices don't match: make sure "
                         "they have the same convention in terms of datetimes "
                         "and symbol-names")

    # chop prices down to only the assets we care about (= unique assets in
    # `factor`).  we could modify `prices` in place, but that might confuse
    # the caller.
    prices = prices.filter(items=factor.index.levels[1])

    raw_values_dict = {}
    column_list = [] # 由timedelta_to_string()生成的诸如"10_D, 20D"的列名

    # periods可能是 [10,20,60] 之类的
    for period in sorted(periods):
        if cumulative_returns:
            returns = prices.pct_change(period) # 这个确实是基于股票是列的基础之上，用pct_change函数算出收益率来
        else:
            returns = prices.pct_change()

        forward_returns = returns.shift(-period).reindex(factor_dateindex)
        """
        shift(-3)，这样做是为了让收益的日期，和原有的prices，错开对齐
        当前         shift后     pct_change值  
                    2021.1.1    NAN
                    2021.1.3    NAN
                    2021.1.4    NAN
        ---------------------------
        2021.1.1    2021.1.5    0.1
        2021.1.3    2021.1.6    0.2
        2021.1.4                NAN
        2021.1.5                NAN
        2021.1.6                NAN
        """

        # 大于几个标准差的就设成nan，这样可以过滤一些异常高、低的收益率
        # 不过我觉得没必要用
        if filter_zscore is not None:
            mask = abs(
                forward_returns - forward_returns.mean()
            ) > (filter_zscore * forward_returns.std())
            forward_returns[mask] = np.nan

        # TODO：没太懂，感觉是看你传入的10、20、60之类的间隔是不是合理，暂时忽略
        # Find the period length, which will be the column name. We'll test
        # several entries in order to find out the most likely period length
        # (in case the user passed inconsinstent data)
        #
        days_diffs = []
        for i in range(30):
            if i >= len(forward_returns.index):
                break
            p_idx = prices.index.get_loc(forward_returns.index[i])
            if p_idx is None or p_idx < 0 or (
                    p_idx + period) >= len(prices.index):
                continue
            start = prices.index[p_idx]
            end = prices.index[p_idx + period]
            period_len = diff_custom_calendar_timedeltas(start, end, freq)
            days_diffs.append(period_len.components.days)

        # scipy.stats.mode函数，寻找出现次数最多的成员
        delta_days = period_len.components.days - mode(days_diffs).mode[0]
        period_len -= pd.Timedelta(days=delta_days)
        label = timedelta_to_string(period_len)

        column_list.append(label)

        raw_values_dict[label] = np.concatenate(forward_returns.values)

    df = pd.DataFrame.from_dict(raw_values_dict)
    df.set_index(
        # 用factor的日期  X(笛卡尔乘)  价格的列(股票)
        pd.MultiIndex.from_product( # from_product()从多个集合的笛卡尔积创建MultiIndex对象。
            [factor_dateindex, prices.columns],
            names=['date', 'asset']
        ),
        inplace=True
    )
    df = df.reindex(factor.index)

    # now set the columns correctly
    df = df[column_list]
    df.index.levels[0].freq = freq
    # exit()
    df.index.set_names(['date', 'asset'], inplace=True)

    return df


def backshift_returns_series(series, N):
    """Shift a multi-indexed series backwards by N observations in
    the first level.

    This can be used to convert backward-looking returns into a
    forward-returns series.
    """
    ix = series.index
    dates, sids = ix.levels
    date_labels, sid_labels = map(np.array, ix.labels)

    # Output date labels will contain the all but the last N dates.
    new_dates = dates[:-N]

    # Output data will remove the first M rows, where M is the index of the
    # last record with one of the first N dates.
    cutoff = date_labels.searchsorted(N)
    new_date_labels = date_labels[cutoff:] - N
    new_sid_labels = sid_labels[cutoff:]
    new_values = series.values[cutoff:]

    assert new_date_labels[0] == 0

    new_index = pd.MultiIndex(
        levels=[new_dates, sids],
        labels=[new_date_labels, new_sid_labels],
        sortorder=1,
        names=ix.names,
    )

    return pd.Series(data=new_values, index=new_index)


def demean_forward_returns(factor_data, grouper=None):
    """
    Convert forward returns to returns relative to mean
    period wise all-universe or group returns.
    group-wise normalization incorporates the assumption of a
    group neutral portfolio constraint and thus allows the
    factor to be evaluated across groups.

    For example, if AAPL 5 period return is 0.1% and mean 5 period
    return for the Technology stocks in our universe was 0.5% in the
    same period, the group adjusted 5 period return for AAPL in this
    period is -0.4%.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        Forward returns indexed by date and asset.
        Separate column for each forward return window.
    grouper : list
        If True, demean according to group.

    Returns
    -------
    adjusted_forward_returns : pd.DataFrame - MultiIndex
        DataFrame of the same format as the input, but with each
        security's returns normalized by group.
    """

    factor_data = factor_data.copy()

    if not grouper:
        grouper = factor_data.index.get_level_values('date')

    cols = get_forward_returns_columns(factor_data.columns)
    factor_data[cols] = factor_data.groupby(grouper)[cols] \
        .transform(lambda x: x - x.mean())

    return factor_data


def print_table(table, name=None, fmt=None):
    """
    Pretty print a pandas DataFrame.

    Uses HTML output if running inside Jupyter Notebook, otherwise
    formatted text output.

    Parameters
    ----------
    table : pd.Series or pd.DataFrame
        Table to pretty-print.
    name : str, optional
        Table name to display in upper left corner.
    fmt : str, optional
        Formatter to use for displaying table elements.
        E.g. '{0:.2f}%' for displaying 100 as '100.00%'.
        Restores original setting after displaying.
    """
    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    if isinstance(table, pd.DataFrame):
        table.columns.name = name

    prev_option = pd.get_option('display.float_format')
    if fmt is not None:
        pd.set_option('display.float_format', lambda x: fmt.format(x))

    display(table)

    if fmt is not None:
        pd.set_option('display.float_format', prev_option)


def get_clean_factor(factor,
                     forward_returns,
                     groupby=None,
                     binning_by_group=False,
                     quantiles=5,
                     bins=None,
                     groupby_labels=None,
                     max_loss=0.35,
                     zero_aware=False):
    """
    格式化因子，和，股票的收益率。
    quantiles，会按照横截面排序成几个，比如5，就是1~5。
    股票收益率，会按照1，5，10，分别列出，这个应该是之前计算好的。

    Formats the factor data, forward return data, and group mappings into a
    DataFrame that contains aligned MultiIndex indices of timestamp and asset.
    The returned data will be formatted to be suitable for Alphalens functions.

    It is safe to skip a call to this function and still make use of Alphalens
    functionalities as long as the factor data conforms to the format returned
    from get_clean_factor_and_forward_returns and documented here

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.
        ::
            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   0.5
                        -----------------------
                        |   BA       |  -1.1
                        -----------------------
            2014-01-01  |   CMG      |   1.7
                        -----------------------
                        |   DAL      |  -0.1
                        -----------------------
                        |   LULU     |   2.7
                        -----------------------

    forward_returns : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by timestamp (level 0) and asset
        (level 1), containing the forward returns for assets.
        Forward returns column names must follow the format accepted by
        pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc).
        'date' index freq property must be set to a trading calendar
        (pandas DateOffset), see infer_trading_calendar for more details.
        This information is currently used only in cumulative returns
        computation
        ::
            ---------------------------------------
                       |       | 1D  | 5D  | 10D
            ---------------------------------------
                date   | asset |     |     |
            ---------------------------------------
                       | AAPL  | 0.09|-0.01|-0.079
                       ----------------------------
                       | BA    | 0.02| 0.06| 0.020
                       ----------------------------
            2014-01-01 | CMG   | 0.03| 0.09| 0.036
                       ----------------------------
                       | DAL   |-0.02|-0.06|-0.029
                       ----------------------------
                       | LULU  |-0.03| 0.05|-0.009
                       ----------------------------

    groupby : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    binning_by_group : bool
        If True, compute quantile buckets separately for each group.
        This is useful when the factor values range vary considerably
        across gorups so that it is wise to make the binning group relative.
        You should probably enable this if the factor is intended
        to be analyzed for a group neutral portfolio
    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    分箱个数
    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Chooses the buckets to be evenly spaced according to the values
        themselves. Useful when the factor contains discrete values.
        Only one of 'quantiles' or 'bins' can be not-None
    行业的代码
    groupby_labels : dict
        A dictionary keyed by group code with values corresponding
        to the display name for each group.
    最多可以扔掉百分之多少无效的数据
    max_loss : float, optional，默认max_loss=0.35
        Maximum percentage (0.00 to 1.00) of factor data dropping allowed,
        computed comparing the number of items in the input factor index and
        the number of items in the output DataFrame index.
        Factor data can be partially dropped due to being flawed itself
        (e.g. NaNs), not having provided enough price data to compute
        forward returns for all factor values, or because it is not possible
        to perform binning.
        Set max_loss=0 to avoid Exceptions suppression.
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.
        'quantiles' is None.

    Returns
    -------
    merged_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.

        - forward returns column names follow the format accepted by
          pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc)

        - 'date' index freq property (merged_data.index.levels[0].freq) is the
          same as that of the input forward returns data. This is currently
          used only in cumulative returns computation
        ::
           -------------------------------------------------------------------
                      |       | 1D  | 5D  | 10D  |factor|group|factor_quantile
           -------------------------------------------------------------------
               date   | asset |     |     |      |      |     |
           -------------------------------------------------------------------
                      | AAPL  | 0.09|-0.01|-0.079|  0.5 |  G1 |      3
                      --------------------------------------------------------
                      | BA    | 0.02| 0.06| 0.020| -1.1 |  G2 |      5
                      --------------------------------------------------------
           2014-01-01 | CMG   | 0.03| 0.09| 0.036|  1.7 |  G2 |      1
                      --------------------------------------------------------
                      | DAL   |-0.02|-0.06|-0.029| -0.1 |  G3 |      5
                      --------------------------------------------------------
                      | LULU  |-0.03| 0.05|-0.009|  2.7 |  G1 |      2
                      --------------------------------------------------------
    """

    initial_amount = float(len(factor.index))

    factor_copy = factor.copy()
    factor_copy.index = factor_copy.index.rename(['date', 'asset'])
    factor_copy = factor_copy[np.isfinite(factor_copy)]

    merged_data = forward_returns.copy()
    merged_data['factor'] = factor_copy

    if groupby is not None:
        if isinstance(groupby, dict):
            diff = set(factor_copy.index.get_level_values(
                'asset')) - set(groupby.keys())
            if len(diff) > 0:
                raise KeyError(
                    "Assets {} not in group mapping".format(
                        list(diff)))

            ss = pd.Series(groupby)
            groupby = pd.Series(index=factor_copy.index,
                                data=ss[factor_copy.index.get_level_values(
                                    'asset')].values)

        if groupby_labels is not None:
            diff = set(groupby.values) - set(groupby_labels.keys())
            if len(diff) > 0:
                raise KeyError(
                    "groups {} not in passed group names".format(
                        list(diff)))

            sn = pd.Series(groupby_labels)
            groupby = pd.Series(index=groupby.index,
                                data=sn[groupby.values].values)

        merged_data['group'] = groupby.astype('category')

    merged_data = merged_data.dropna()

    # 保留的条数
    fwdret_amount = float(len(merged_data.index))

    no_raise = False if max_loss == 0 else True
    
    # 传入factor+forward_returns，返回每只股票每期的分箱的排名
    # 每行是 日期+股票，多了一列"factor_quantile"，是他在当期的排名分箱号
    quantile_data = quantize_factor(
        merged_data,
        quantiles,
        bins,
        binning_by_group,
        no_raise,
        zero_aware
    )

    merged_data['factor_quantile'] = quantile_data

    merged_data = merged_data.dropna()

    binning_amount = float(len(merged_data.index))

    # 丢弃的百分比
    tot_loss = (initial_amount - binning_amount) / initial_amount
    # 保留的百分比
    fwdret_loss = (initial_amount - fwdret_amount) / initial_amount
    # bin分箱时候，丢弃的百分比
    bin_loss = tot_loss - fwdret_loss

    # print("Dropped %.1f%% entries from factor data: %.1f%% in forward "
    #       "returns computation and %.1f%% in binning phase "
    #       "(set max_loss=0 to see potentially suppressed Exceptions)." %
    #       (tot_loss * 100, fwdret_loss * 100, bin_loss * 100))

    print("在分箱阶段,丢弃了 %.1f%% 的无效因子：分箱前丢弃%.1f%% + 分箱时丢弃%.1f%%" %
          (tot_loss * 100, fwdret_loss * 100, bin_loss * 100))

    # 如果丢掉的太多，就不玩了
    if tot_loss > max_loss:
        message = ("丢掉了%.1f%%的无效数据，超过可容忍的%.1f%%最大阈值" % (tot_loss * 100, max_loss * 100))
        raise MaxLossExceededError(message)

    return merged_data


def get_clean_factor_and_forward_returns(factor,
                                         prices,
                                         groupby=None,
                                         binning_by_group=False,
                                         quantiles=5,
                                         bins=None,
                                         periods=(1, 5, 10),
                                         filter_zscore=20,
                                         groupby_labels=None,
                                         max_loss=0.35,
                                         zero_aware=False,
                                         cumulative_returns=True):
    """
    入参需要强调一下，格式不一样的：
    factor的索引是 [日期,股票]，列只有1个，就是因子值
    prices的索引是 [日期]， 而列是所有的股票。
    cumulative_returns，也不是从最开始的累计收益，而是periods提到的周期内的累计收益。

    Formats the factor data, pricing data, and group mappings into a DataFrame
    that contains aligned MultiIndex indices of timestamp and asset. The
    returned data will be formatted to be suitable for Alphalens functions.

    It is safe to skip a call to this function and still make use of Alphalens
    functionalities as long as the factor data conforms to the format returned
    from get_clean_factor_and_forward_returns and documented here

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.
        ::
            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   0.5
                        -----------------------
                        |   BA       |  -1.1
                        -----------------------
            2014-01-01  |   CMG      |   1.7
                        -----------------------
                        |   DAL      |  -0.1
                        -----------------------
                        |   LULU     |   2.7
                        -----------------------

    prices : pd.DataFrame
        A wide form Pandas DataFrame indexed by timestamp with assets
        in the columns.
        Pricing data must span the factor analysis time period plus an
        additional buffer window that is greater than the maximum number
        of expected periods in the forward returns calculations.
        It is important to pass the correct pricing data in depending on
        what time of period your signal was generated so to avoid lookahead
        bias, or  delayed calculations.
        'Prices' must contain at least an entry for each timestamp/asset
        combination in 'factor'. This entry should reflect the buy price
        for the assets and usually it is the next available price after the
        factor is computed but it can also be a later price if the factor is
        meant to be traded later (e.g. if the factor is computed at market
        open but traded 1 hour after market open the price information should
        be 1 hour after market open).
        'Prices' must also contain entries for timestamps following each
        timestamp/asset combination in 'factor', as many more timestamps
        as the maximum value in 'periods'. The asset price after 'period'
        timestamps will be considered the sell price for that asset when
        computing 'period' forward returns.
        ::
            ----------------------------------------------------
                        | AAPL |  BA  |  CMG  |  DAL  |  LULU  |
            ----------------------------------------------------
               Date     |      |      |       |       |        |
            ----------------------------------------------------
            2014-01-01  |605.12| 24.58|  11.72| 54.43 |  37.14 |
            ----------------------------------------------------
            2014-01-02  |604.35| 22.23|  12.21| 52.78 |  33.63 |
            ----------------------------------------------------
            2014-01-03  |607.94| 21.68|  14.36| 53.94 |  29.37 |
            ----------------------------------------------------

    groupby : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    binning_by_group : bool
        If True, compute quantile buckets separately for each group.
        This is useful when the factor values range vary considerably
        across gorups so that it is wise to make the binning group relative.
        You should probably enable this if the factor is intended
        to be analyzed for a group neutral portfolio
    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Chooses the buckets to be evenly spaced according to the values
        themselves. Useful when the factor contains discrete values.
        Only one of 'quantiles' or 'bins' can be not-None
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float, optional
        Sets forward returns greater than X standard deviations
        from the the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
    groupby_labels : dict
        A dictionary keyed by group code with values corresponding
        to the display name for each group.
    max_loss : float, optional
        Maximum percentage (0.00 to 1.00) of factor data dropping allowed,
        computed comparing the number of items in the input factor index and
        the number of items in the output DataFrame index.
        Factor data can be partially dropped due to being flawed itself
        (e.g. NaNs), not having provided enough price data to compute
        forward returns for all factor values, or because it is not possible
        to perform binning.
        Set max_loss=0 to avoid Exceptions suppression.
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.
    cumulative_returns : bool, optional
        If True, forward returns columns will contain cumulative returns.
        Setting this to False is useful if you want to analyze how predictive
        a factor is for a single forward day.

    Returns
    -------
    merged_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - forward returns column names follow  the format accepted by
          pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc)
        - 'date' index freq property (merged_data.index.levels[0].freq) will be
          set to a trading calendar (pandas DateOffset) inferred from the input
          data (see infer_trading_calendar for more details). This is currently
          used only in cumulative returns computation
        ::
           -------------------------------------------------------------------
                      |       | 1D  | 5D  | 10D  |factor|group|factor_quantile
           -------------------------------------------------------------------
               date   | asset |     |     |      |      |     |
           -------------------------------------------------------------------
                      | AAPL  | 0.09|-0.01|-0.079|  0.5 |  G1 |      3
                      --------------------------------------------------------
                      | BA    | 0.02| 0.06| 0.020| -1.1 |  G2 |      5
                      --------------------------------------------------------
           2014-01-01 | CMG   | 0.03| 0.09| 0.036|  1.7 |  G2 |      1
                      --------------------------------------------------------
                      | DAL   |-0.02|-0.06|-0.029| -0.1 |  G3 |      5
                      --------------------------------------------------------
                      | LULU  |-0.03| 0.05|-0.009|  2.7 |  G1 |      2
                      --------------------------------------------------------

    See Also
    --------
    utils.get_clean_factor
        For use when forward returns are already available.
    """
    # 计算N期之后，这个股票对应的收益率
    # 这个获得的是各个股票的N日（prices）间隔的收益率
    forward_returns = compute_forward_returns(
        factor,
        prices,
        periods,
        filter_zscore,
        cumulative_returns, # 这个参数有迷惑性，不是从开始每期每期的累计收益，而是找个周期的累计收益
                            # 如果是false，计算的是每一天的，true是这10天的，而不是从最最开始的时候的累计
    )

    # 会把factor, forward_returns合成1个df，且，算出每个横截面（日子）对应的排名（根据quantiles）
    # TODO groupby没搞定？？？
    factor_data = get_clean_factor(factor, forward_returns, groupby=groupby,
                                   groupby_labels=groupby_labels,
                                   quantiles=quantiles, bins=bins,
                                   binning_by_group=binning_by_group,
                                   max_loss=max_loss, zero_aware=zero_aware)

    return factor_data


def rate_of_return(period_ret, base_period):
    """
    Convert returns to 'one_period_len' rate of returns: that is the value the
    returns would have every 'one_period_len' if they had grown at a steady
    rate

    我理解这个是把一段时间的收益率，还原到每一天去，比如我知道了5天的收益率，我来问你每天的收益率
      --> period_ret.add(1).pow(conversion_factor).sub(1)
      --> period_ret.add(1).pow(1/5).sub(1)

    Parameters
    ----------
    period_ret: pd.DataFrame
        DataFrame containing returns values with column headings representing
        the return period.
    base_period: string
        The base period length used in the conversion
        It must follow pandas.Timedelta constructor format (e.g. '1 days',
        '1D', '30m', '3h', '1D1h', etc)

    Returns
    -------
    pd.DataFrame
        DataFrame in same format as input but with 'one_period_len' rate of
        returns values.
    """
    # period_ret 就是你回测期的总的回报率
    period_len = period_ret.name  # periold_len pdb值：'1D', '5D'
    conversion_factor = (pd.Timedelta(base_period) / # base_period就是你的回测期，用的是column的name，即 '1D','5D'
                         pd.Timedelta(period_len))  #
    """
    (Pdb) pp period_ret, period_len, base_period
        (factor_quantile
        1.0    0.008426
        2.0   -0.009757
        3.0   -0.001937
        4.0   -0.006727
        5.0    0.010852
        Name: 5D, dtype: float64,
         '5D',
         '1D')
    """
    return period_ret.add(1).pow(conversion_factor).sub(1)


def std_conversion(period_std, base_period):
    """
    one_period_len standard deviation (or standard error) approximation

    Parameters
    ----------
    period_std: pd.DataFrame
        DataFrame containing standard deviation or standard error values
        with column headings representing the return period.
    base_period: string
        The base period length used in the conversion
        It must follow pandas.Timedelta constructor format (e.g. '1 days',
        '1D', '30m', '3h', '1D1h', etc)

    Returns
    -------
    pd.DataFrame
        DataFrame in same format as input but with one-period
        standard deviation/error values.
    """
    period_len = period_std.name
    conversion_factor = (pd.Timedelta(period_len) /
                         pd.Timedelta(base_period))
    return period_std / np.sqrt(conversion_factor)


def get_forward_returns_columns(columns, require_exact_day_multiple=False):
    """
    Utility that detects and returns the columns that are forward returns
    """

    # If exact day multiples are required in the forward return periods,
    # drop all other columns (e.g. drop 3D12h).
    if require_exact_day_multiple:
        pattern = re.compile(r"^(\d+([D]))+$", re.IGNORECASE)
        valid_columns = [(pattern.match(col) is not None) for col in columns]

        if sum(valid_columns) < len(valid_columns):
            warnings.warn(
                "Skipping return periods that aren't exact multiples"
                + " of days."
            )
    else:
        pattern = re.compile(r"^(\d+([Dhms]|ms|us|ns]))+$", re.IGNORECASE)
        valid_columns = [(pattern.match(col) is not None) for col in columns]

    return columns[valid_columns]


def timedelta_to_string(timedelta):
    """
    Utility that converts a pandas.Timedelta to a string representation
    compatible with pandas.Timedelta constructor format

    Parameters
    ----------
    timedelta: pd.Timedelta

    Returns
    -------
    string
        string representation of 'timedelta'
    """
    c = timedelta.components
    format = ''
    if c.days != 0:
        format += '%dD' % c.days
    if c.hours > 0:
        format += '%dh' % c.hours
    if c.minutes > 0:
        format += '%dm' % c.minutes
    if c.seconds > 0:
        format += '%ds' % c.seconds
    if c.milliseconds > 0:
        format += '%dms' % c.milliseconds
    if c.microseconds > 0:
        format += '%dus' % c.microseconds
    if c.nanoseconds > 0:
        format += '%dns' % c.nanoseconds
    return format


def timedelta_strings_to_integers(sequence):
    """
    Converts pandas string representations of timedeltas into integers of days.

    Parameters
    ----------
    sequence : iterable
        List or array of timedelta string representations, e.g. ['1D', '5D'].

    Returns
    -------
    sequence : list
        Integer days corresponding to the input sequence, e.g. [1, 5].
    """
    return list(map(lambda x: pd.Timedelta(x).days, sequence))


def add_custom_calendar_timedelta(input, timedelta, freq):
    """
    Add timedelta to 'input' taking into consideration custom frequency, which
    is used to deal with custom calendars, such as a trading calendar

    Parameters
    ----------
    input : pd.DatetimeIndex or pd.Timestamp
    timedelta : pd.Timedelta
    freq : pd.DataOffset (CustomBusinessDay, Day or BusinessDay)

    Returns
    -------
    pd.DatetimeIndex or pd.Timestamp
        input + timedelta
    """
    if not isinstance(freq, (Day, BusinessDay, CustomBusinessDay)):
        raise ValueError("freq must be Day, BDay or CustomBusinessDay")
    days = timedelta.components.days
    offset = timedelta - pd.Timedelta(days=days)
    return input + freq * days + offset


def diff_custom_calendar_timedeltas(start, end, freq):
    """
    Compute the difference between two pd.Timedelta taking into consideration
    custom frequency, which is used to deal with custom calendars, such as a
    trading calendar

    Parameters
    ----------
    start : pd.Timestamp
    end : pd.Timestamp
    freq : CustomBusinessDay (see infer_trading_calendar)
    freq : pd.DataOffset (CustomBusinessDay, Day or BDay)

    Returns
    -------
    pd.Timedelta
        end - start
    """
    if not isinstance(freq, (Day, BusinessDay, CustomBusinessDay)):
        raise ValueError("freq must be Day, BusinessDay or CustomBusinessDay")

    weekmask = getattr(freq, 'weekmask', None)
    holidays = getattr(freq, 'holidays', None)

    if weekmask is None and holidays is None:
        if isinstance(freq, Day):
            weekmask = 'Mon Tue Wed Thu Fri Sat Sun'
            holidays = []
        elif isinstance(freq, BusinessDay):
            weekmask = 'Mon Tue Wed Thu Fri'
            holidays = []

    if weekmask is not None and holidays is not None:
        # we prefer this method as it is faster
        actual_days = np.busday_count(np.array(start).astype('datetime64[D]'),
                                      np.array(end).astype('datetime64[D]'),
                                      weekmask, holidays)
    else:
        # default, it is slow
        actual_days = pd.date_range(start, end, freq=freq).shape[0] - 1
        if not freq.onOffset(start):
            actual_days -= 1

    timediff = end - start
    delta_days = timediff.components.days - actual_days
    return timediff - pd.Timedelta(days=delta_days)
