#
# Copyright 2017 Quantopian, Inc.
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

import os
import sys
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

from . import performance as perf
from . import plotting
from . import utils


def plot_image(no=None, factor_name=None):
    """
    plot各类因子的分析图
    :param no:
    :param factor_name:
    :return:
    """

    # 获得调用者的函数名
    caller_name = sys._getframe(1).f_code.co_name

    if not factor_name: factor_name = 'Unkown'

    factor_dir = "/debug/{}".format(factor_name)
    if not os.path.exists(factor_dir): os.makedirs(factor_dir)
    if no:
        file_name = os.path.join(factor_dir, "{}_{}.jpg".format(caller_name, no))
        plt.savefig(file_name)
    else:
        file_name = os.path.join(factor_dir, "{}.jpg".format(caller_name))
        plt.savefig(file_name)


class GridFigure(object):
    """
    It makes life easier with grid plots
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.fig = plt.figure(figsize=(14, rows * 7))
        self.gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.3)
        self.curr_row = 0
        self.curr_col = 0

    def next_row(self):
        if self.curr_col != 0:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, :])
        self.curr_row += 1
        return subplt

    def next_cell(self):
        if self.curr_col >= self.cols:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, self.curr_col])
        self.curr_col += 1
        return subplt

    def close(self):
        plt.close(self.fig)
        self.fig = None
        self.gs = None


@plotting.customize
def create_summary_tear_sheet(
        factor_data, long_short=True, group_neutral=False,factor_name=None
):
    """
    Creates a small summary tear sheet with returns, information, and turnover
    analysis.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool
        Should this computation happen on a long short portfolio? if so, then
        mean quantile returns will be demeaned across the factor universe.
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
    """

    # Returns Analysis
    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
    )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )

    compstd_quant_daily = std_quant_daily.apply(
        utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
    )

    alpha_beta = perf.factor_alpha_beta(
        factor_data, demeaned=long_short, group_adjust=group_neutral
    )

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_quant_rateret_bydate,
        factor_data["factor_quantile"].max(),
        factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )

    periods = utils.get_forward_returns_columns(factor_data.columns)
    periods = list(map(lambda p: pd.Timedelta(p).days, periods))

    fr_cols = len(periods)
    vertical_sections = 2 + fr_cols * 3
    gf = GridFigure(rows=vertical_sections, cols=1)

    plotting.plot_quantile_statistics_table(factor_data,factor_name)

    plotting.plot_returns_table(
        alpha_beta, mean_quant_rateret, mean_ret_spread_quant,factor_name
    )

    plotting.plot_quantile_returns_bar(
        mean_quant_rateret,
        by_group=False,
        ylim_percentiles=None,
        ax=gf.next_row(),
        factor_name=factor_name
    )

    # Information Analysis
    ic = perf.factor_information_coefficient(factor_data)
    ic_result = plotting.plot_information_table(ic,factor_name)

    # Turnover Analysis
    quantile_factor = factor_data["factor_quantile"]

    quantile_turnover = {
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in range(1, int(quantile_factor.max()) + 1)
            ],
            axis=1,
        )
        for p in periods
    }

    autocorrelation = pd.concat(
        [
            perf.factor_rank_autocorrelation(factor_data, period)
            for period in periods
        ],
        axis=1,
    )

    plotting.plot_turnover_table(autocorrelation, quantile_turnover,factor_name)

    # plt.show()
    plot_image(factor_name=factor_name)
    gf.close()


@plotting.customize
def create_returns_tear_sheet(
        factor_data, long_short=True, group_neutral=False, by_group=False, factor_name=None
):
    """
    Creates a tear sheet for returns analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool # long多，short空
        Should this computation happen on a long short portfolio? if so, then
        mean quantile returns will be demeaned降低 across the factor universe体系.
        Additionally factor values will be demeaned across the factor universe
        when factor weighting the portfolio for cumulative returns plots
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
        Additionally each group will weight the same in cumulative returns
        plots
    by_group : bool
        If True, display graphs separately for each group.
    """
    # 名字具有迷惑性，可不是计算因子的收益率，而是因子作用下的资产的收益率
    # 这个名字具备迷惑性，不是因子的收益率，而是股票的收益率计算
    factor_returns = perf.factor_returns(
        factor_data, long_short, group_neutral
    )

    # 这个很重要，是按照分组quantile，来算每组的收益率
    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data,
        by_group=False,  # group其实就是行业
        demeaned=long_short,  # 多空，其实，就是减不减均值
        group_adjust=group_neutral,  # 搞不搞行业中性化
    )

    # ??? 感觉是算累计收益率似的？没太明白
    mean_quant_rateret = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
    )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    #
    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )

    compstd_quant_daily = std_quant_daily.apply(
        utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
    )

    alpha_beta = perf.factor_alpha_beta(
        factor_data, factor_returns, long_short, group_neutral
    )

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_quant_rateret_bydate,
        factor_data["factor_quantile"].max(),
        factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )

    fr_cols = len(factor_returns.columns)
    vertical_sections = 2 + fr_cols * 3
    gf = GridFigure(rows=vertical_sections, cols=1)

    plotting.plot_returns_table(
        alpha_beta, mean_quant_rateret, mean_ret_spread_quant,factor_name
    )

    plotting.plot_quantile_returns_bar(
        mean_quant_rateret,
        by_group=False,
        ylim_percentiles=None,
        ax=gf.next_row(),
        factor_name=factor_name
    )

    plotting.plot_quantile_returns_violin(
        mean_quant_rateret_bydate, ylim_percentiles=(1, 99), ax=gf.next_row()
    )

    trading_calendar = factor_data.index.levels[0].freq
    if trading_calendar is None:
        trading_calendar = pd.tseries.offsets.BDay()
        warnings.warn(
            "'freq' not set in factor_data index: assuming business day",
            UserWarning,
        )

    # Compute cumulative returns from daily simple returns, if '1D'
    # returns are provided.
    if "1D" in factor_returns:
        title = (
                "Factor Weighted "
                + ("Group Neutral " if group_neutral else "")
                + ("Long/Short " if long_short else "")
                + "Portfolio Cumulative Return (1D Period)"
        )

        plotting.plot_cumulative_returns(
            factor_returns["1D"], period="1D", title=title, ax=gf.next_row(),factor_name=factor_name
        )

        plotting.plot_cumulative_returns_by_quantile(
            mean_quant_ret_bydate["1D"], period="1D", ax=gf.next_row(),factor_name=factor_name
        )

    ax_mean_quantile_returns_spread_ts = [
        gf.next_row() for x in range(fr_cols)
    ]
    plotting.plot_mean_quantile_returns_spread_time_series(
        mean_ret_spread_quant,
        std_err=std_spread_quant,
        bandwidth=0.5,
        ax=ax_mean_quantile_returns_spread_ts,
        factor_name=factor_name
    )

    # plt.show()
    plot_image(factor_name=factor_name)
    gf.close()

    if by_group:
        (
            mean_return_quantile_group,
            mean_return_quantile_group_std_err,
        ) = perf.mean_return_by_quantile(
            factor_data,
            by_date=False,
            by_group=True,
            demeaned=long_short,
            group_adjust=group_neutral,
        )

        mean_quant_rateret_group = mean_return_quantile_group.apply(
            utils.rate_of_return,
            axis=0,
            base_period=mean_return_quantile_group.columns[0],
        )

        num_groups = len(
            mean_quant_rateret_group.index.get_level_values("group").unique()
        )

        vertical_sections = 1 + (((num_groups - 1) // 2) + 1)
        gf = GridFigure(rows=vertical_sections, cols=2)

        ax_quantile_returns_bar_by_group = [
            gf.next_cell() for _ in range(num_groups)
        ]
        plotting.plot_quantile_returns_bar(
            mean_quant_rateret_group,
            by_group=True,
            ylim_percentiles=(5, 95),
            ax=ax_quantile_returns_bar_by_group,
        )
        # plt.show()
        plot_image(factor_name=factor_name,no="group")
        gf.close()

    """
    列出此函数中的数据结果，供参考
    factor_returns 因子和股票收益率整合数据，是整个因子的期间收益率，
    比如5D的-0.011991，就是2020-01-02的5天后（2020-01-08）的收益率
                       1D        5D       10D
    date
    2020-01-02 -0.003905 -0.011991 -0.011562
    2020-01-03 -0.009487 -0.011938 -0.013898
    2020-01-06  0.006794  0.009132 -0.019035
    2020-01-07 -0.012808  0.000650 -0.030828
    2020-01-08  0.007658  0.010118 -0.019045
    ...              ...       ...       ...
    2020-11-11  0.003248  0.042907  0.059134
    2020-11-12  0.024770  0.039765  0.048045
    2020-11-13  0.000885  0.025681  0.029234
    2020-11-16  0.002162  0.021977  0.028535
    2020-11-17  0.010308  0.014305  0.017877
    
    ----------------------------------------------
    mean_quant_ret 股票收益率分层数据，这个是分层后的**平均**收益率，你看，日期没有了
                            1D        5D       10D
    factor_quantile
    1                0.000469  0.002528  0.004002
    2               -0.000308 -0.003135 -0.007179
    3               -0.000171  0.001305  0.003913
    4               -0.000999 -0.002981 -0.001240
    5                0.000324  0.000144 -0.002900
    ----------------------------------------------
    mean_quant_rateret 股票收益率分层，这个
                            1D        5D       10D
    factor_quantile
    1                0.000469  0.000505  0.000400
    2               -0.000308 -0.000628 -0.000720
    3               -0.000171  0.000261  0.000391
    4               -0.000999 -0.000597 -0.000124
    5                0.000324  0.000029 -0.000290
    ----------------------------------------------
    std_quantile 分层收益的标准差
                            1D        5D       10D
    factor_quantile
    1                0.001078  0.002393  0.003333
    2                0.001037  0.002560  0.003752
    3                0.001233  0.003377  0.005057
    4                0.001050  0.002698  0.004203
    5                0.001185  0.002479  0.003305
    ---------------------------------------------------------
    mean_quant_ret_bydate 分层的收益率的每期数据，<------------------------- 这个最重要  
                                       1D        5D       10D
    factor_quantile date
    1               2020-01-02 -0.006622  0.032152  0.031694
                    2020-01-03  0.015522  0.051757  0.057915
                    2020-01-06  0.007057  0.012068  0.030915
                    2020-01-07  0.020194  0.013098  0.027879
                    2020-01-08 -0.004523  0.005561 -0.007309
    ...                              ...       ...       ...
    5               2020-11-11  0.005672  0.093278  0.105856
                    2020-11-12  0.050313  0.082732  0.080722
                    2020-11-13 -0.001784  0.052973  0.039285
                    2020-11-16  0.013559  0.046653  0.046796
                    2020-11-17  0.019855  0.009947  0.019335
    ---------------------------------------------------------
    std_quant_daily 分层的收益率的每天数据的标准差
                                       1D        5D       10D
    factor_quantile date
    1               2020-01-02  0.003118  0.000365  0.031198
                    2020-01-03  0.004643  0.000287  0.004013
                    2020-01-06  0.016330  0.000010  0.001747
                    2020-01-07  0.009811  0.030267  0.009582
                    2020-01-08  0.004602  0.018311  0.015435
    ...                              ...       ...       ...
    5               2020-11-11  0.000135  0.083704  0.044860
                    2020-11-12  0.050961  0.070027  0.033575
                    2020-11-13  0.013132  0.021369  0.042332
                    2020-11-16  0.033913  0.033358  0.034859
                    2020-11-17  0.004259  0.023549  0.065362
    ---------------------------------------------------------
    mean_quant_rateret_bydate 分层的收益率的每天(细化到每天)数据，感觉这数据没啥用
                                       1D        5D       10D
    factor_quantile date
    1               2020-01-02 -0.006622  0.006349  0.003125
                    2020-01-03  0.015522  0.010144  0.005646
                    2020-01-06  0.007057  0.002402  0.003049
                    2020-01-07  0.020194  0.002606  0.002754
                    2020-01-08 -0.004523  0.001110 -0.000733
    ...                              ...       ...       ...
    5               2020-11-11  0.005672  0.017996  0.010113
                    2020-11-12  0.050313  0.016025  0.007793
                    2020-11-13 -0.001784  0.010377  0.003861
                    2020-11-16  0.013559  0.009161  0.004584
                    2020-11-17  0.019855  0.001982  0.001917
    ---------------------------------------------------------
    compstd_quant_daily 不知
                                       1D        5D       10D
    factor_quantile date
    1               2020-01-02  0.003118  0.000163  0.009866
                    2020-01-03  0.004643  0.000129  0.001269
                    2020-01-06  0.016330  0.000005  0.000553
                    2020-01-07  0.009811  0.013536  0.003030
                    2020-01-08  0.004602  0.008189  0.004881
    ...                              ...       ...       ...
    5               2020-11-11  0.000135  0.037433  0.014186
                    2020-11-12  0.050961  0.031317  0.010617
                    2020-11-13  0.013132  0.009557  0.013386
                    2020-11-16  0.033913  0.014918  0.011023
                    2020-11-17  0.004259  0.010531  0.020669
    ---------------------------------------------------------
    alpha_beta alpha和beta的数据
                       1D        5D       10D
    Ann. alpha -0.024242 -0.019153  0.006152
    beta        0.055238  0.147883  0.110754
    ----------------------------------------------
    mean_ret_spread_quant 不知
                       1D        5D       10D
    date
    2020-01-02 -0.004850 -0.010227 -0.004903
    2020-01-03 -0.021946 -0.011993 -0.007355
    2020-01-06 -0.000324 -0.000850 -0.006359
    2020-01-07 -0.041991 -0.000730 -0.006911
    2020-01-08  0.018839  0.003096 -0.003578
    ...              ...       ...       ...
    2020-11-11  0.011971  0.029559  0.017420
    2020-11-12  0.060745  0.025684  0.014513
    2020-11-13  0.012465  0.022505  0.010746
    2020-11-16  0.011377  0.016024  0.009023
    2020-11-17  0.044467  0.010615  0.006572
    ----------------------------------------------
    std_spread_quant 不知
                       1D        5D       10D
    date
    2020-01-02  0.004321  0.004466  0.010705
    2020-01-03  0.004715  0.000145  0.002116
    2020-01-06  0.019092  0.002682  0.022002
    2020-01-07  0.009894  0.017105  0.041916
    2020-01-08  0.005620  0.011916  0.029102
    ...              ...       ...       ...
    2020-11-11  0.008611  0.037669  0.016698
    2020-11-12  0.051327  0.031321  0.012660
    2020-11-13  0.019179  0.011875  0.016759
    2020-11-16  0.034720  0.015902  0.012054
    2020-11-17  0.004286  0.014652  0.021769
    ----------------------------------------------
    """

    # print("factor_returns 因子和股票收益率整合数据\n",factor_returns)
    # print("mean_quant_ret 股票收益率分层数据\n",mean_quant_ret)
    # print("mean_quant_rateret 股票收益率分层（？？？）数据\n",mean_quant_rateret)
    # print("std_quantile 分层收益的标准差\n",std_quantile)
    # print("mean_quantile_ret_bydate 分层的收益率的每期数据，这个最重要\n",mean_quant_ret_bydate)# !!!
    # print("std_quant_daily 分层的收益率的每天数据的标准差\n",std_quant_daily)
    # # 我靠，这个是平均每天的收益率（用于计算5天、10天他们的，因为他们的收益率都是10天，所以要搞出来每天的）
    # print("mean_quant_rateret_bydate 分层的收益率的每天(细化到每天)数据，感觉这数据没啥用\n",mean_quant_rateret_bydate)
    # print("compstd_quant_daily 不知\n",compstd_quant_daily)
    # print("alpha_beta alpha和beta的数据\n",alpha_beta)
    # print("mean_ret_spread_quant 不知\n",mean_ret_spread_quant)
    # print("std_spread_quant 不知\n",std_spread_quant)

    return factor_returns, mean_quant_ret_bydate


@plotting.customize
def create_information_tear_sheet(
        factor_data, group_neutral=False, by_group=False, factor_name=None
):
    """
    Creates a tear sheet for information analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    group_neutral : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, display graphs separately for each group.
    """

    ic = perf.factor_information_coefficient(factor_data, group_neutral)

    ic_result = plotting.plot_information_table(ic,factor_name)

    columns_wide = 2
    fr_cols = len(ic.columns)
    rows_when_wide = ((fr_cols - 1) // columns_wide) + 1
    vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    ax_ic_ts = [gf.next_row() for _ in range(fr_cols)]
    plotting.plot_ic_ts(ic, ax=ax_ic_ts,factor_name=factor_name)

    ax_ic_hqq = [gf.next_cell() for _ in range(fr_cols * 2)]
    plotting.plot_ic_hist(ic, ax=ax_ic_hqq[::2])
    plotting.plot_ic_qq(ic, ax=ax_ic_hqq[1::2])

    if not by_group:
        mean_monthly_ic = perf.mean_information_coefficient(
            factor_data,
            group_adjust=group_neutral,
            by_group=False,
            by_time="M",
        )
        ax_monthly_ic_heatmap = [gf.next_cell() for x in range(fr_cols)]
        plotting.plot_monthly_ic_heatmap(
            mean_monthly_ic, ax=ax_monthly_ic_heatmap,factor_name=factor_name
        )

    if by_group:
        mean_group_ic = perf.mean_information_coefficient(
            factor_data, group_adjust=group_neutral, by_group=True
        )

        plotting.plot_ic_by_group(mean_group_ic, ax=gf.next_row(),factor_name=factor_name)

    # plt.show()
    plot_image(factor_name=factor_name)
    gf.close()
    return ic, ic_result


@plotting.customize
def create_turnover_tear_sheet(factor_data, turnover_periods=None,factor_name=None):
    """
    Creates a tear sheet for analyzing the turnover properties of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    turnover_periods : sequence[string], optional
        Periods to compute turnover analysis on. By default periods in
        'factor_data' are used but custom periods can provided instead. This
        can be useful when periods in 'factor_data' are not multiples of the
        frequency at which factor values are computed i.e. the periods
        are 2h and 4h and the factor is computed daily and so values like
        ['1D', '2D'] could be used instead
    """

    if turnover_periods is None:
        input_periods = utils.get_forward_returns_columns(
            factor_data.columns, require_exact_day_multiple=True,
        ).to_numpy()  # https://github.com/quantopian/alphalens/issues/379 bugfix by piginzoo,2021.12.17
        turnover_periods = utils.timedelta_strings_to_integers(input_periods)
    else:
        turnover_periods = utils.timedelta_strings_to_integers(
            turnover_periods,
        )

    quantile_factor = factor_data["factor_quantile"]

    quantile_turnover = {
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in quantile_factor.sort_values().unique().tolist()
            ],
            axis=1,
        )
        for p in turnover_periods
    }

    autocorrelation = pd.concat(
        [
            perf.factor_rank_autocorrelation(factor_data, period)
            for period in turnover_periods
        ],
        axis=1,
    )

    plotting.plot_turnover_table(autocorrelation, quantile_turnover,factor_name)

    fr_cols = len(turnover_periods)
    columns_wide = 1
    rows_when_wide = ((fr_cols - 1) // 1) + 1
    vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    for period in turnover_periods:
        if quantile_turnover[period].isnull().all().all():
            continue
        plotting.plot_top_bottom_quantile_turnover(
            quantile_turnover[period], period=period, ax=gf.next_row(),factor_name=factor_name
        )

    for period in autocorrelation:
        if autocorrelation[period].isnull().all():
            continue
        plotting.plot_factor_rank_auto_correlation(
            autocorrelation[period], period=period, ax=gf.next_row(),factor_name=factor_name
        )

    # plt.show()
    plot_image(factor_name=factor_name)
    gf.close()


@plotting.customize
def create_full_tear_sheet(factor_data,
                           long_short=True,
                           group_neutral=False,
                           by_group=False):
    """
    Creates a full tear sheet for analysis and evaluating single
    return predicting (alpha) factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool
        Should this computation happen on a long short portfolio?
        - See tears.create_returns_tear_sheet for details on how this flag
        affects returns analysis
    group_neutral : bool
        Should this computation happen on a group neutral portfolio?
        - See tears.create_returns_tear_sheet for details on how this flag
        affects returns analysis
        - See tears.create_information_tear_sheet for details on how this
        flag affects information analysis
    by_group : bool
        If True, display graphs separately for each group.
    """

    plotting.plot_quantile_statistics_table(factor_data,factor_name)
    create_returns_tear_sheet(
        factor_data, long_short, group_neutral, by_group, set_context=False
    )
    create_information_tear_sheet(
        factor_data, group_neutral, by_group, set_context=False
    )
    create_turnover_tear_sheet(factor_data, set_context=False)


@plotting.customize
def create_event_returns_tear_sheet(factor_data,
                                    returns,
                                    avgretplot=(5, 15),
                                    long_short=True,
                                    group_neutral=False,
                                    std_bar=True,
                                    by_group=False,
                                    factor_name=None):
    """
    Creates a tear sheet to view the average cumulative returns for a
    factor within a window (pre and post event).

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, the factor
        quantile/bin that factor value belongs to and (optionally) the group
        the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    returns : pd.DataFrame
        A DataFrame indexed by date with assets in the columns containing daily
        returns.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    avgretplot: tuple (int, int) - (before, after)
        If not None, plot quantile average cumulative returns
    long_short : bool
        Should this computation happen on a long short portfolio? if so then
        factor returns will be demeaned across the factor universe
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
    std_bar : boolean, optional
        Show plots with standard deviation bars, one for each quantile
    by_group : bool
        If True, display graphs separately for each group.
    """

    before, after = avgretplot

    avg_cumulative_returns = perf.average_cumulative_return_by_quantile(
        factor_data,
        returns,
        periods_before=before,
        periods_after=after,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    num_quantiles = int(factor_data["factor_quantile"].max())

    vertical_sections = 1
    if std_bar:
        vertical_sections += ((num_quantiles - 1) // 2) + 1
    cols = 2 if num_quantiles != 1 else 1
    gf = GridFigure(rows=vertical_sections, cols=cols)
    plotting.plot_quantile_average_cumulative_return(
        avg_cumulative_returns,
        by_quantile=False,
        std_bar=False,
        ax=gf.next_row(),
        factor_name=factor_name
    )
    if std_bar:
        ax_avg_cumulative_returns_by_q = [
            gf.next_cell() for _ in range(num_quantiles)
        ]
        plotting.plot_quantile_average_cumulative_return(
            avg_cumulative_returns,
            by_quantile=True,
            std_bar=True,
            ax=ax_avg_cumulative_returns_by_q,
            factor_name=factor_name
        )

    # plt.show()
    plot_image(factor_name=factor_name)
    gf.close()

    if by_group:
        groups = factor_data["group"].unique()
        num_groups = len(groups)
        vertical_sections = ((num_groups - 1) // 2) + 1
        gf = GridFigure(rows=vertical_sections, cols=2)

        avg_cumret_by_group = perf.average_cumulative_return_by_quantile(
            factor_data,
            returns,
            periods_before=before,
            periods_after=after,
            demeaned=long_short,
            group_adjust=group_neutral,
            by_group=True,
        )

        for group, avg_cumret in avg_cumret_by_group.groupby(level="group"):
            avg_cumret.index = avg_cumret.index.droplevel("group")
            plotting.plot_quantile_average_cumulative_return(
                avg_cumret,
                by_quantile=False,
                std_bar=False,
                title=group,
                ax=gf.next_cell(),
                factor_name=factor_name
            )

        # plt.show()
        plot_image(factor_name=factor_name,no="group")
        gf.close()


@plotting.customize
def create_event_study_tear_sheet(factor_data,
                                  returns,
                                  avgretplot=(5, 15),
                                  rate_of_ret=True,
                                  n_bars=50,
                                  factor_name=None):
    """
    Creates an event study tear sheet for analysis of a specific event.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single event, forward returns for each
        period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
    returns : pd.DataFrame, required only if 'avgretplot' is provided
        A DataFrame indexed by date with assets in the columns containing daily
        returns.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    avgretplot: tuple (int, int) - (before, after), optional
        If not None, plot event style average cumulative returns within a
        window (pre and post event).
    rate_of_ret : bool, optional
        Display rate of return instead of simple return in 'Mean Period Wise
        Return By Factor Quantile' and 'Period Wise Return By Factor Quantile'
        plots
    n_bars : int, optional
        Number of bars in event distribution plot
    """

    long_short = False

    plotting.plot_quantile_statistics_table(factor_data,factor_name)

    gf = GridFigure(rows=1, cols=1)
    plotting.plot_events_distribution(
        events=factor_data["factor"], num_bars=n_bars, ax=gf.next_row(),factor_name=factor_name
    )
    # plt.show()
    plot_image(factor_name=factor_name)
    gf.close()

    if returns is not None and avgretplot is not None:
        create_event_returns_tear_sheet(
            factor_data=factor_data,
            returns=returns,
            avgretplot=avgretplot,
            long_short=long_short,
            group_neutral=False,
            std_bar=True,
            by_group=False,
        )

    factor_returns = perf.factor_returns(
        factor_data, demeaned=False, equal_weight=True
    )

    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data, by_group=False, demeaned=long_short
    )
    if rate_of_ret:
        mean_quant_ret = mean_quant_ret.apply(
            utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
        )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data, by_date=True, by_group=False, demeaned=long_short
    )
    if rate_of_ret:
        mean_quant_ret_bydate = mean_quant_ret_bydate.apply(
            utils.rate_of_return,
            axis=0,
            base_period=mean_quant_ret_bydate.columns[0],
        )

    fr_cols = len(factor_returns.columns)
    vertical_sections = 2 + fr_cols * 1
    gf = GridFigure(rows=vertical_sections + 1, cols=1)

    plotting.plot_quantile_returns_bar(
        mean_quant_ret, by_group=False, ylim_percentiles=None, ax=gf.next_row()
    )

    plotting.plot_quantile_returns_violin(
        mean_quant_ret_bydate, ylim_percentiles=(1, 99), ax=gf.next_row(),factor_name=factor_name
    )

    trading_calendar = factor_data.index.levels[0].freq
    if trading_calendar is None:
        trading_calendar = pd.tseries.offsets.BDay()
        warnings.warn(
            "'freq' not set in factor_data index: assuming business day",
            UserWarning,
        )

    # plt.show()
    plot_image(factor_name=factor_name,no="detail")
    gf.close()
