import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Chapter 10 Simple Linear Regression and Correlation""")
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import altair as alt
    from scipy import stats

    import inspect


    def get_source(func) -> str:
        """Display a function's source code as markdown"""
        source = inspect.getsource(func)
        return f"""```python
    {source}
    ```"""
    return alt, get_source, mo, pl, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 10.1 A Probabilistic Model for Simple Linear Regression""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
    _Linear regression analysis_ begins by fitting a straight line, $y = \beta_0 + \beta_1 x$, to a set of paired data $\{(x_i, y_i), i = 1, 2, \ldots , n\}$ on two numerical variables $x$ and $y$. The linear regression model

    $$
    Y_i = \beta_0 + \beta_1 x_i + \epsilon_i\ (i=1,2, \ldots, n)
    $$

    has these basic assumptions:

    1. The predictor variable $x$ is regarded as nonrandom because it is assumed to be set by the investigator.
    2. The mean of $Y_i$ is a linear function of $x_i$.
    3. The errors $\epsilon_i$ are i.i.d. normal.
    """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Ex 10.1

    Tell whether the following mathematical models are theoretical and deterministic or empirical and probabilistic.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "1. Maxwell's equations of electromagnetism.": "theoretical and deterministic",
            "2. An econometric model of the U.S. economy.": "empirical and probabilistic",
            "3. A credit scoring model for the probability of a credit applicant being a good risk as a function of selected variables, e.g., income, outstanding debts, etc.": "empirical and probabilistic",
        },
        multiple=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Ex 10.2

    Tell whether the following mathematical models are theoretical and deterministic or empirical and probabilistic.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            '''1. An item response model for the probability of a correct response to an item on a "true-false" test as a function of the item's intrinsic difficulty.''': "empirical and probabilistic",
            "2. The Cobb-Douglas production function, which relates the output of a firm to its capital and labor inputs.": "empirical and probabilistic",
            "3. Kepler's laws of planetary motion.": "theoretical and deterministic",
        },
        multiple=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Ex 10.3

    Give an example of an experimental study in which the explanatory variable is controlled at fixed values, while the response variable is random. Also, give an example of an observational study in which both variables are uncontrolled and random.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Controlled explanatory variable at fixed values: temperatures in the day as a function of the hours 1h, 2h, ...
    - Both uncontrolled: humidity as a function of temperature.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 10.2 Fitting the Simple Linear Regression Model""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
    The _least squares(LS) estimates_ $\hat{\beta}_0$ and $\hat{\beta}_1$ minimize $Q = \sum_{i=1}^n [y_i - (\beta_0 + \beta_1 x_i) ]^2$ and are given by

    $$
    \begin{align*}
    \hat{\beta}_0 &= \bar{y} - \hat{\beta}_1 \bar{x},\\
    \hat{\beta}_1 &= \frac{S_{xy}}{S_{xx}}
    \end{align*}
    $$

    where $S_{xy} = \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})$ and $S_{xx} = \sum_{i=1}^n (x_i - \bar{x})^2$. The *fitted values* are given by $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$ and the *residuals* by $e_i = y_i - \hat{y}_i$.

    The total sum of squares (SST), regression sum of squares (SSR) and error sum of squares (SSE) are defined as $\mathrm{SST} = \sum_{i=1}^n (y_i - \bar{y})^2$, $\mathrm{SSR} = \sum_{i=1}^n (\hat{y}_i - \bar{y})^2$, and $\mathrm{SSE} = \sum_{i=1}^n (y_i - \hat{y}_i)^2$. These sums of squares satisfy the identity $\mathrm{SST} = \mathrm{SSR} + \mathrm{SSE}$. A measure of goodness of fit of the least squares line is the *coefficient of determination*,

    $$
    r^2 = \frac{\mathrm{SSR}}{\mathrm{SST}} = 1 - \frac{\mathrm{SSE}}{\mathrm{SST}}
    $$

    which represents the proportion of variation in $y$ that is accounted for by regression on $x$. The *correlation coefficient* $r$ equals $\pm\sqrt{r^2}$, where $\mathrm{sign}(r) = \mathrm{sign}(\hat{\beta}_1)$. In fact, $r = \hat{\beta}_1 (s_x / s_y)$, where $s_x$ and $s_y$ are the sample standard deviations of $x$ and $y$, respectively.

    The *probabilistic model* for linear regression assumes that $y_i$ is the observed value of r.v. $Y \thicksim N(\mu_i, \sigma^2)$, where $\mu_i = \beta_0 + \beta_1 x_i$ and the $Y_i$ are independent. An unbiased estimate of $\sigma^2$ is provided by $s^2 = \mathrm{SSE}/(n - 2)$ with $n-2$ d.f.
    """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(get_source, mo, pl):
    def linreg(x: pl.Expr, y: pl.Expr, x_star: float|None = None) -> pl.Expr:
        """
        Gives results of simple linear regression and estimation
        by directly translating textbook formulas to polars expressions.

        Input:
        - x, y: observations
        - x_star: input point for estimation

        Output:
        β0/1: intercept/slope of regression line
        r2: coefficient of determination
        s2: mean square error estimate of σ^2
        f: F statistic of H0: β1 = 0. f=t^2
        se_β0/1: standard error of β0/1
        se_est_ci/pi: standard error of the estimation confidence/prediction interval
        """
        n = x.len()
        sxx = ((x - x.mean()) ** 2).sum()
        syy = ((y - y.mean()) ** 2).sum()
        sxy = ((x - x.mean()) * (y - y.mean())).sum()
        β1 = sxy / sxx
        β0 = y.mean() - x.mean() * β1
        r2 = sxy**2 / (sxx * syy)
        s2 = (syy - β1**2 * sxx) / (n - 2)
        f = β1**2 * sxx / s2
        se_β0 = (s2 * (x**2).sum() / (n * sxx)).sqrt()
        se_β1 = (s2 / sxx).sqrt()
        se_est_ci = (s2 * (1 / n + (x_star - x.mean()) ** 2 / sxx)).sqrt()
        se_est_pi = (s2 * (1 + 1 / n + (x_star - x.mean()) ** 2 / sxx)).sqrt()
        return pl.struct(
            β0.alias("β0"),
            β1.alias("β1"),
            r2.alias("r2"),
            s2.alias("s2"),
            f.alias("f"),
            se_β0.alias("se_β0"),
            se_β1.alias("se_β1"),
            se_est_ci.alias("se_est_ci"),
            se_est_pi.alias("se_est_pi"),
        )


    mo.callout(
        mo.md(rf""" 
    The following `linreg` function of polars expressions follows directly from the formulas in the book, yielding $\beta_0$ and $\beta_1$ among some others defined in later chapters.

    {get_source(linreg)}

    """),
        kind="info",
    )
    return (linreg,)


@app.cell(hide_code=True)
def _(mo, pl):
    df_ex4 = pl.read_json("../SDAEI-Tamhane/ch10/Ex10-4.json").explode(pl.all())


    def _rename_df(df: pl.DataFrame) -> pl.DataFrame:
        return df.rename(
            {
                "No": "Obs.\nNo.",
                "LAST": "Duration of Eruption\n(LAST)",
                "NEXT": "Time Between Eruptions\n(NEXT)",
            }
        )


    mo.md(
        f"""
    ### Ex 10.4

    The time between eruptions of Old Faithful geyser in Yellowstone National Park is random but is related to the duration of the last eruption. The table below shows these times for 21 consecutive eruptions.

    {
            mo.ui.table(
                _rename_df(df_ex4),
                label="Old Faithful Eruptions: Duration and Time Between Eruptions",
                show_column_summaries=False,
                selection=None,
                show_data_types=False,
            )
        }

    Let us see how well we can predict the time to next eruption, given the length of time of the last eruption."""
    )
    return (df_ex4,)


@app.cell(hide_code=True)
def _(alt, df_ex4, mo):
    _chart = df_ex4.plot.scatter(
        alt.X("LAST").scale(domain=[1, 5.5]),
        alt.Y("NEXT").scale(domain=[30, 100]),
    )

    mo.md(
        f"""
    /// details | (a) Make a scatter plot of NEXT vs. LAST. Does the relationship appear to be approximately linear?

    {mo.as_html(_chart)}

    Yes, the points appear approximately linear.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(alt, df_ex4, linreg, mo, pl):
    _res = df_ex4.select(linreg(pl.col("LAST"), pl.col("NEXT"))).item()

    _chart_scatter = df_ex4.plot.scatter(
        alt.X("LAST").scale(domain=[1, 5.5]),
        alt.Y("NEXT").scale(domain=[30, 100]),
    )

    _chart_regression = _chart_scatter.transform_regression(
        "LAST", "NEXT"
    ).mark_line(color="red")

    mo.md(
        rf"""
    /// details | (b) Fit a least squares regression line. Use it to predict the time to the next eruption if the last eruption lasted 3 minutes.

    When applied to the dataset, `linreg` gives $\beta_0$ = {_res["β0"]:.2f} and $\beta_1$ = {_res["β1"]:.2f} . If the last eruption lasted 3 minutes, the time to the next eruption would be in 

    $$
    \beta_0 + \beta_1 \cdot 3 =
        {_res["β0"]:.2f} + {_res["β1"]:.2f} \cdot 3 = {_res["β0"] + _res["β1"] * 3:.2f}
    $$

    minutes.

    {mo.as_html(_chart_scatter + _chart_regression)}

    ///"""
    )
    return


@app.cell(hide_code=True)
def _(df_ex4, linreg, mo, pl):
    _res = df_ex4.select(linreg(pl.col("LAST"), pl.col("NEXT"))).item()

    mo.md(
        rf"""
    /// details | (c) What proportion of variability in NEXT is accounted for by LAST? Does it suggest that LAST is a good predictor of NEXT?

    Also using the `linreg` function, $r^2$ = {_res['r2']:.2f}, suggesting that `LAST` is a pretty good predictor of `NEXT`.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex4, linreg, mo, pl):
    _res = df_ex4.select(linreg(pl.col("LAST"), pl.col("NEXT"))).item()


    mo.md(
        rf"""
    /// details | (d) Calculate the mean square error estimate of $\sigma$.

    Also using the `linreg` function, $s^2$ = {_res['s2']:.2f}, and the mean square error estimate of $\sigma$ is $s$ = {_res['s2'] ** 0.5:.2f}.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, pl):
    df_ex5 = pl.read_json("../SDAEI-Tamhane/ch10/Ex10-5.json").explode(pl.all())

    mo.md(
        f"""
    ### Ex 10.5

    The data below show Olympic triple jump winning distances for men in meters for the years 1896 to 1992 (there were no Olympic games in 1916, 1940, and 1944).

    {
            mo.ui.table(
                df_ex5,
                label="Men's Olympic Triple Jump Winning Distance (in meters)",
                show_column_summaries=False,
                selection=None,
                show_data_types=False
            )
        }
    """
    )
    return (df_ex5,)


@app.cell(hide_code=True)
def _(alt, df_ex5, mo, pl):
    _chart = df_ex5.with_columns(pl.col("Year").cast(int).cast(str)).plot.scatter(
        alt.X("Year:T"), # .scale(domain=[(1890,1,1), (2000,1,1)]),
        alt.Y("Distance").scale(domain=[13, 19]),
    )

    mo.md(
        f"""
    /// details | (a) Make a scatter plot of the length of the jump by year. Does the relationship appear to be approximately linear?

    {mo.as_html(_chart)}

    Yes, the points appear approximately linear.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(alt, df_ex5, linreg, mo, pl):
    _res = df_ex5.select(linreg(pl.col("Year"), pl.col("Distance"))).item()

    _chart_scatter = df_ex5.with_columns(pl.col("Year").cast(int).cast(str)).plot.scatter(
        alt.X("Year:T"), # .scale(domain=[(1890,1,1), (2000,1,1)]),
        alt.Y("Distance").scale(domain=[13, 19]),
    )

    _chart_regression = _chart_scatter.transform_regression(
        "Year", "Distance"
    ).mark_line(color="red")

    mo.md(
        rf"""
    /// details |  (b) Fit a least squares regression line.

    Using `linreg`, $\beta_0$ = {_res['β0']:.2f} and $\beta_1$ = {_res['β1']:.3f}.

    {mo.as_html(_chart_scatter + _chart_regression)}

    ///"""
    )
    return


@app.cell(hide_code=True)
def _(df_ex5, linreg, mo, pl):
    _res = df_ex5.select(linreg(pl.col("Year"), pl.col("Distance"))).item()


    mo.md(
        rf"""
    /// details | (c) Calculate the mean square error estimate of $\sigma$.

    Also using the `linreg` function, $s^2$ = {_res['s2']:.3f}, and the mean square error estimate of $\sigma$ is $s$ = {_res['s2'] ** 0.5:.3f}.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, pl):
    df_ex6 = pl.read_json("../SDAEI-Tamhane/ch10/Ex10-6.json").explode(pl.all())

    mo.md(
        f"""
    ### Ex 10.6

    The following data give the barometric pressure (in inches of mercury) and the boiling point (in °F) of water in the Alps.

    {
            mo.ui.table(
                df_ex6,
                label="Boiling Point of Water in the Alps",
                show_column_summaries=False,
                selection=None,
                show_data_types=False
            )
        }

    """
    )
    return (df_ex6,)


@app.cell(hide_code=True)
def _(alt, df_ex6, mo):
    _chart = df_ex6.plot.scatter(
        alt.X("Pressure").scale(domain=[20, 31]),
        alt.Y("Temp").scale(domain=[192, 215]),
    )

    mo.md(
        f"""
    /// details | (a) Make a scatter plot of the boiling point by barometric pressure. Does the relationship appear to be approximately linear?

    {mo.as_html(_chart)}

    Yes, the relationship is approximately linear.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(alt, df_ex6, linreg, mo, pl):
    _res = df_ex6.select(linreg(pl.col("Pressure"), pl.col("Temp"))).item()

    _chart_scatter = df_ex6.plot.scatter(
        alt.X("Pressure").scale(domain=[20, 31]),
        alt.Y("Temp").scale(domain=[192, 215]),
    )

    _chart_regression = _chart_scatter.transform_regression(
        "Pressure", "Temp"
    ).mark_line(color="red")

    mo.md(
        rf"""
    /// details | (b) Fit a least squares regression line. What proportion of variation in the boiling point is accounted for by linear regression on the barometric pressure?

    Using `linreg`, $\beta_0$ = {_res['β0']:.2f}, $\beta_1$ = {_res['β1']:.3f}, and $r^2$ = {_res['r2']:.3f}. That is, {_res['r2'] * 100:.1f}% percent of variation in the boiling point is accounted for by linear regression on the barometric pressure.

    {mo.as_html(_chart_scatter + _chart_regression)}

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex6, linreg, mo, pl):
    _res = df_ex6.select(linreg(pl.col("Pressure"), pl.col("Temp"))).item()

    mo.md(
        rf"""
    /// details | (c) Calculate the mean square error estimate of $\sigma$.

    Also using the `linreg` function, $s^2$ = {_res["s2"]:.3f}, and the mean square error estimate of $\sigma$ is $s$ = {_res["s2"] ** 0.5:.3f}.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, pl):
    df_ex7 = pl.read_json("../SDAEI-Tamhane/ch10/Ex10-7.json").explode(pl.all())

    mo.md(
        f"""
    ### Ex 10.7

    The following table shows Olympic 100 meter backstroke winning times for women for the years 1924 to 1992 (there were no Olympic games in 1940 and 1944).

    {
            mo.ui.table(
                df_ex7,
                label="Women's Olympic 100 Meter Backstroke Winning Times (in seconds)",
                show_column_summaries=False,
                selection=None,
                show_data_types=False,
            )
        }
    """
    )
    return (df_ex7,)


@app.cell(hide_code=True)
def _(alt, df_ex7, mo, pl):
    _chart = df_ex7.with_columns(pl.col("Year").cast(int).cast(str)).plot.scatter(
        alt.X("Year:T"),
        alt.Y("Time").scale(domain=[55, 90]),
    )

    mo.md(
        f"""
    /// details | (a) Make a scatter plot of the winning times by year. Does the relationship appear to be approximately linear?

    {mo.as_html(_chart)}

    Yes, the relationship is approximately linear.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(alt, df_ex7, linreg, mo, pl):
    _res = df_ex7.select(linreg(pl.col("Year"), pl.col("Time"))).item()

    _chart_scatter = df_ex7.with_columns(pl.col("Year").cast(int).cast(str)).plot.scatter(
        alt.X("Year:T"),
        alt.Y("Time").scale(domain=[55, 90]),
    )

    _chart_regression = _chart_scatter.transform_regression(
        "Year", "Time"
    ).mark_line(color="red")

    mo.md(
        rf"""
    /// details | (b) Fit a least squares regression line.

    Using `linreg`, $\beta_0$ = {_res['β0']:.1f} and $\beta_1$ = {_res['β1']:.3f}.

    {mo.as_html(_chart_scatter + _chart_regression)}

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex7, linreg, mo, pl):
    _res = df_ex7.select(linreg(pl.col("Year"), pl.col("Time"))).item()

    mo.md(
        rf"""
    /// details | (c) Calculate the mean square error estimate of $\sigma$.

    Also using the `linreg` function, $s^2$ = {_res["s2"]:.3f}, and the mean square error estimate of $\sigma$ is $s$ = {_res["s2"] ** 0.5:.3f}.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Ex 10.8

    Often the conditions of the problem dictate that the intercept coefficient $\beta_0$ must be zero, e.g., the sales revenue as a function of the number of units sold or the gas mileage of a car as a function of the weight of the car. This is called _regression through the origin_. Show that the LS estimate of the slope coefficient $\beta_1$ when fitting the straight line $y = \beta_1 x$ based on the data $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$ is

    $$
    \beta_1 = \frac{\sum x_i y_i}{\sum x_i^2}.
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To find the minumum of the sum of errors

    $$
    Q = \sum (\beta_1 x_i - y_i)^2,
    $$

    we take its derivative with respect to $\beta_1$.

    $$ \begin{align*} 
    \frac{\partial Q}{\partial \beta_1} &= \sum 2x_i(\beta_1 x_i - y_i)\\
    &= 2\sum \beta_1 x_i^2 - x_i y_i
    \end{align*}
    $$

    Therefore the minimum is taken at

    $$
    \beta_1 = \frac{\sum x_i^2}{\sum x_i y_i}.
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 10.3 Statistical Inference for Simple Linear Regression""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
    The estimated standard errors of $\hat{\beta}_0$ and $\hat{\beta}_1$ equal

    $$
    \mathrm{SE}(\hat{\beta}_0) = s\sqrt{\frac{\sum x_i^2}{n\,S_{xx}}} \quad \text{and}\quad \mathrm{SE}(\hat{\beta}_1) = \frac{s}{\sqrt{S_{xx}}}.
    $$

    These are used to construct confidence intervals and perform hypothesis tests on $\beta_0$ and $\beta_1$. For example, a $100(1-\alpha)$% confidence interval on $\beta_1$ is given by

    $$
    \hat{\beta_1} \pm t_{n-2, \alpha/2}\,\textrm{SE}(\hat{\beta_1}).
    $$

    A common use of the fitted regression model is to _predict_ $Y^*$ for specified $x = x^*$ or to _estimate_ $\mu^* = \textrm{E}(Y^*)$. In both cases we have

    $$
    \hat{Y}^* = \hat{\mu}^* = \beta_0 + \beta_1 x^*.
    $$

    However, a $100(1-\alpha)$% _prediction interval_ for $Y^*$ is wider than a $100(1-\alpha)$% confidence interval for $\mu^*$, because $Y^*$ is an r.v., while $\mu^*$ is a fixed constant.
    """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Ex 10.9

    Refer to Exercise 10.5.
    """
    )
    return


@app.cell
def _(df_ex5, linreg, mo, pl, stats):
    _res = df_ex5.select(linreg(pl.col("Year"), pl.col("Distance"))).item()

    _t = _res['β1'] / _res['se_β1']
    _pval = stats.t.sf(_t, df_ex5.height-2)

    mo.md(
        rf"""
    /// details | (a) Is there a significant increasing linear trend in the triple jump distance? Test at $\alpha = .05$.

    This is a test of $H_0: \beta_1 \le 0$ and the $t$-statistic is {_t:.2f} with a one-sided $P$-value of {_pval:.2e} < $\alpha$. So yes there is a significant increasing trend.


    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex5, linreg, mo, pl, stats):
    _x = 2004
    _α = 0.05
    _res = df_ex5.select(
        linreg(pl.col("Year"), pl.col("Distance"), x_star=_x)
    ).item()
    _n = df_ex5.height

    _y = _res["β0"] + _res["β1"] * _x
    _t_star = stats.t.ppf(1 - _α / 2, _n - 2).item() # critical value
    [_pi_low, _pi_high] = [
        _y - _t_star * _res["se_est_pi"],
        _y + _t_star * _res["se_est_pi"],
    ]

    mo.md(
        rf"""
    /// details | (b) Calculate a 95% PI for the winning jump in 2004. Do you think this prediction is reliable? Why or why not? Would a 95% CI for the winning jump in 2004 have a meaningful interpretation? Explain.

    A 95% PI for the winning jump in 2004 is [{_pi_low:.2f}, {_pi_high:.2f}], but it is not reliable since we are extrapolating. In this case, a CI is not meaningful because there will be at most a single winning jump in 2004. 

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Ex 10.10

    Refer to Exercise 10.6.
    """
    )
    return


@app.cell
def _(df_ex6, linreg, mo, pl, stats):
    _x = 28
    _α = 0.05
    _res = df_ex6.select(
        linreg(pl.col("Pressure"), pl.col("Temp"), x_star=_x)
    ).item()
    _n = df_ex6.height

    _y = _res["β0"] + _res["β1"] * _x
    _t_star = stats.t.ppf(1 - _α / 2, _n - 2).item() # critical value
    [_ci_low, _ci_high] = [
        _y - _t_star * _res["se_est_ci"],
        _y + _t_star * _res["se_est_ci"],
    ]


    mo.md(
        fr"""
    /// details | (a) Calculate a 95% CI for the boiling point if the barometric pressure is 28 inches of mercury. Interpret your CI.

    The said CI is calculated to be [{_ci_low:.2f}, {_ci_high:.2f}]. That is to say, there is a 95% chance that this interval includes the boiling point at 28 inches of of mercury on the true regression line.

    ///
    """
    )
    return


@app.cell
def _(df_ex6, linreg, mo, pl, stats):
    _x = 31
    _α = 0.05
    _res = df_ex6.select(
        linreg(pl.col("Pressure"), pl.col("Temp"), x_star=_x)
    ).item()
    _n = df_ex6.height

    _y = _res["β0"] + _res["β1"] * _x
    _t_star = stats.t.ppf(1 - _α / 2, _n - 2).item() # critical value
    [_low, _high] = [
        _y - _t_star * _res["se_est_ci"],
        _y + _t_star * _res["se_est_ci"],
    ]


    mo.md(
        rf"""
    /// details | (b) Calculate a 95% CI for the boiling point if the barometric pressure is 31 inches of mercury. Compare this with the CI of (a).

    The said CI is calculated to be [{_low:.2f}, {_high:.2f}]. It is much wider than (a) at 28 inches of mercury and should be treated as unreliable because we are extrapolating outside the data domain.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Ex 10.11

    Refer to the Old Faithful data in Exercise 10.4.
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex4, linreg, mo, pl, stats):
    _x = 3
    _α = 0.05
    _res = df_ex4.select(
        linreg(pl.col("LAST"), pl.col("NEXT"), x_star=_x)
    ).item()
    _n = df_ex4.height

    _y = _res["β0"] + _res["β1"] * _x
    _t_star = stats.t.ppf(1 - _α / 2, _n - 2).item() # critical value
    [_low, _high] = [
        _y - _t_star * _res["se_est_pi"],
        _y + _t_star * _res["se_est_pi"],
    ]


    mo.md(
        rf"""
    /// details | (a) Calculate a 95% PI for the time to the next eruption if the last eruption lasted 3 minutes.

    The said PI is calculated to be [{_low:.2f}, {_high:.2f}].
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex4, linreg, mo, pl, stats):
    _x = 3
    _α = 0.05
    _res = df_ex4.select(
        linreg(pl.col("LAST"), pl.col("NEXT"), x_star=_x)
    ).item()
    _n = df_ex4.height

    _y = _res["β0"] + _res["β1"] * _x
    _t_star = stats.t.ppf(1 - _α / 2, _n - 2).item() # critical value
    [_low, _high] = [
        _y - _t_star * _res["se_est_ci"],
        _y + _t_star * _res["se_est_ci"],
    ]

    mo.md(
        rf"""
    /// details | (b) Calculate a 95% CI for the mean time to the next eruption for a last eruption lasting 3 minutes. Compare this CI with the PI obtained in (a).

    The said CI is calculated to be [{_low:.2f}, {_high:.2f}] which is a lot narrower than the PI in (a).

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex4, linreg, mo, pl, stats):
    _x = 1
    _α = 0.05
    _res = df_ex4.select(
        linreg(pl.col("LAST"), pl.col("NEXT"), x_star=_x)
    ).item()
    _n = df_ex4.height

    _y = _res["β0"] + _res["β1"] * _x
    _t_star = stats.t.ppf(1 - _α / 2, _n - 2).item() # critical value
    [_low, _high] = [
        _y - _t_star * _res["se_est_pi"],
        _y + _t_star * _res["se_est_pi"],
    ]

    mo.md(
        rf"""
    /// details | (c) Repeat (a) if the last eruption lasted 1 minute. Do you think this prediction is reliable? Why or why not?

    The PI for a 1 minute last eruption is [{_low:.2f}, {_high:.2f}] which is unreliable because we are extrapolating outside of the data domain.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Ex 10.12

    Refer to Exercise 10.7.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | (a) Calculate a 95% PI for the winning time in 2004. Do you think this prediction is reliable? Why or why not?

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | (b) Use the regression equation to find the year in which the winning time would break 1 minute. Given that the Olympics are every four years, during which Olympics would this happen?

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 10.4 Regression Diagnosis""")
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""Residuals are key to checking the model assumptions such as normality of the $Y_i$, linearity of the regression model, constant variance $\sigma^2$, and independence of the $Y_i$. Residuals are also useful for detecting _outliers_ and _influential observations_. Many of these diagnostic checks are done by plotting residuals in appropriate ways."""
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 10.5 *Correlation Analysis""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    _Correlation analysis_ assumes that the data $\{(x_i, y_i), i = 1, 2, \ldots , n\}$ form a random sample from a _bivariate normal distribution_ with correlation coefficient $\rho$. An estimate of $\rho$ is the sample correlation coefficient $r$. An exact test of $H_0: \rho = 0$ is a $t$-test with $n-2$ d.f. based on the test statistic

    $$
    t = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}}.
    $$

    This equals $t = \hat{\beta}_1 / \textrm{SE}(\hat{\beta}_1)$ which is used to test $H_0: \beta_1 = 0$ in the related regression model. In other cases only approximate large sample inferences are available. These inferences use the parameterization

    $$
    \psi = \frac{1}{2}\log_e \left(\frac{1+\rho}{1-\rho}\right).
    $$

    The sample estimate $\hat{\psi}$ of $\psi$, obtained by substituting $\hat{\rho} = r$ in the above expression, is approximately normally distributed with mean=$\frac{1}{2}\log_e (\frac{1+\rho}{1-\rho})$ and variance=$\frac{1}{n-3}$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 10.6 Pitfalls of Regression and Correlation Analysis""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


if __name__ == "__main__":
    app.run()
