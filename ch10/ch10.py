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
    return alt, get_source, mo, np, pl, stats


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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
def _(df_ex7, linreg, mo, pl, stats):
    _x = 2004
    _α = 0.05
    _res = df_ex7.select(
        linreg(pl.col("Year"), pl.col("Time"), x_star=_x)
    ).item()
    _n = df_ex7.height

    _y = _res["β0"] + _res["β1"] * _x
    _t_star = stats.t.ppf(1 - _α / 2, _n - 2).item() # critical value
    [_low, _high] = [
        _y - _t_star * _res["se_est_pi"],
        _y + _t_star * _res["se_est_pi"],
    ]

    mo.md(
        rf"""
    /// details | (a) Calculate a 95% PI for the winning time in 2004. Do you think this prediction is reliable? Why or why not?

    The specified PI is calculated to be [{_low:.2f}, {_high:.2f}]. However, this prediction is unreliable because we are extrapolating outside the data range (latest available year was 1996).

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex7, linreg, mo, pl):
    _y = 60
    _res = df_ex7.select(linreg(pl.col("Year"), pl.col("Time"))).item()

    _x = (_y - _res["β0"]) / _res["β1"]


    mo.md(
        rf"""
    /// details | (b) Use the regression equation to find the year in which the winning time would break 1 minute. Given that the Olympics are every four years, during which Olympics would this happen?

    The year {_x:.0f} (by inverse regression).

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, pl):
    df_ex13 = pl.read_json("../SDAEI-Tamhane/ch10/Ex10-13.json").explode(pl.all())

    mo.md(
        rf"""
    ### Ex 10.13

    The U.S. infant mortality rates (IMR) (per 1000 live births) for both sexes and all races for the years 1981-1990 (coded as years 1-10) were as follows:

    {
            mo.ui.table(
                df_ex13,
                show_column_summaries=False,
                selection=None,
                show_data_types=False,
            )
        }

    The MINITAB regression output is shown below.

    ```text
    MINITAB Output for Regression of Infant Mortality Rate Versus Year
    MTB regress 'IMR' 1 'Year'
    The regression equation is
    IMR = 12.0 - 0.270 Year
    Predictor     Coef     Stdev   t-ratio       p
    Constant   12.0333    0.0851    141.35   0.000
    Year      -0.26970   0.01372    -19.66   0.000
    s = 0.1246   R-sq = 98.0%   R-sq(adj) = 97.7%
    Analysis of Variance
    SOURCE      DF       ss        MS         F         p
    Regression   1   6.0008    6.0008    386.39     0.000
    Error        8   0.1242    0.0155
    Total        9   6.1250
    ```

    Answer the following questions based on this output.
    """
    )
    return (df_ex13,)


@app.cell(hide_code=True)
def _(df_ex13, mo, stats):
    _β1 = 0.2697
    _β10 = 0.3
    _se = 0.01372
    _n = df_ex13.height
    _t = (_β1 - _β10) / _se
    _pval = stats.t.cdf(_t, _n-2)

    mo.md(
        rf"""
    /// details | (a) What was the average rate of decrease in the infant mortality rate per year during 1981-1990? Suppose that for the rest of the Western world the rate of decrease in IMR was 0.3 deaths per year per 1000 live births. Was the U.S. rate of decrease significantly less than that for the rest of the Western world? Use $\alpha$ = .05.

    Reading off the MINITAB output, the rate of IMR decrease is {_β1} deaths per 1,000 live births per year. 

    Set up $H_0: \beta_1 \ge 0.3$ and use the MINITAB output $\textrm{{SE}}(\hat{{\beta}}_1)$ = {_se} to get a $t$-statistic of {_t:.3f} with $P$-value {_pval:.3f} < $\alpha$. So yes, the US IMR decrease is less than that for the rest of the Western world.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, np, stats):
    _α = 0.05
    _n = 10
    _β0 = 12.0333
    _β1 = -0.26970
    _s = 0.1246
    _ssr = 6.0008
    _sxx = _ssr / _β1**2
    _x = 1995 - 1980
    _x_mean = (1 + 10) / 2
    _y = _β0 + _β1 * _x
    _t_crit = stats.t.ppf(1 - _α / 2, _n - 2)
    _se_est_pi = _s * np.sqrt(1 + 1 / _n + (_x - _x_mean) ** 2 / _sxx)
    _err = _t_crit * _se_est_pi


    mo.md(
        r"""
    /// details | (b) Predict the IMR for the year 1995. Calculate a 95% prediction interval. (Note that $S_{xx}$ can be obtained from the values of $\textrm{Stdev}(\hat{\beta}_1)$ and $s$ given in the MINITAB output.)

    $S_{xx}$ can be obtained as in the note, or simply $S_{xx} = \textrm{SSR}/\hat{\beta}_1^2$, and both $\textrm{SSR}$ and $\hat{\beta}_1$ are in the MINITAB output. Anyways, the calculated IMR for the year 1995 is """
        f""" {_y:.2f} ± {_err:.2f} = [{_y - _err:.2f}, {_y + _err:.2f}].

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Ex 10.14

    For the linear regression model show that the sample mean $\bar{Y}$ and the LS slope estimator $\hat{\beta}_1$ are statistically independent.

    _Hint_: Write $\bar{Y} = \frac{1}{n} \sum Y_i$ and $\hat{\beta}_1 = \sum c_i Y_i$, where $c_i = (x_i - \bar{x}) / S_{xx}$ and satisfy $\sum c_i = 0$. Then show that $\textrm{Cov}(\bar{Y}, \hat{\beta}_1) = 0$ by using the formula 

    $$
    \textrm{Cov}\left(\sum_{i=1}^m a_i X_i, \sum_{j=1}^n b_j Y_j \right) = \sum_{i=1}^m \sum_{j=1}^n a_i b_j\,\textrm{Cov}(X_i,Y_j)
    $$

    for the covariance between two sets of linear combinations of r.v.'s. Finally, use the result that if two normal r.v.'s are uncorrelated, then they are independent.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    By the formula for $\hat{\beta}_1$:

    $$
    \begin{align*}
    \hat{\beta}_1 &= S_{xy}/S_{xx} \\
    &= \frac{1}{S_{xx}} \sum(x_i - \bar{x})(x_i - \bar{y}) \\
    &= \frac{1}{S_{xx}} \left[\sum(x_i - \bar{x})\,y_i -\bar{y}\sum (x_i - \bar{x})\right] \\
    &= \frac{1}{S_{xx}} \sum(x_i - \bar{x})\,y_i.
    \end{align*}
    $$

    Let $c_i = (x_i - \bar{x}) / S_{xx}$, we have $\hat{\beta}_1 = \sum c_i Y_i$ and $\sum c_i = 0$. Now,

    $$
    \begin{align*}
    \textrm{Cov}(\bar{Y}, \hat{\beta}_1) &= \textrm{Cov}\left(\frac{1}{n}\sum_i Y_i, \sum_j c_j Y_j\right) \\
    &= \sum_i \sum_j \frac{c_j}{n}\,\textrm{Cov}(Y_i, Y_j) \\
    &= \frac{1}{n} \sum_i c_i \quad \quad \triangleright\ \textrm{Cov}(Y_i, Y_j) = 1_{i=j}\\
    &= 0.
    \end{align*}
    $$

    On the other hand, $\bar{Y}$ and $\hat{\beta}_1$ are both normal, and for jointly normal random variables, being uncorrelated implies independence.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 10.4 Regression Diagnosis""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""Residuals are key to checking the model assumptions such as normality of the $Y_i$, linearity of the regression model, constant variance $\sigma^2$, and independence of the $Y_i$. Residuals are also useful for detecting _outliers_ and _influential observations_. Many of these diagnostic checks are done by plotting residuals in appropriate ways."""
        ),
        kind="info",
    )
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Ex 10.15

    Often, the probability of response $p\ (0 \le p \le 1)$ is modeled as a function of a stimulus $x$ by the _logistic function_:

    $$
    p = \frac{\exp{(\beta_0 + \beta_1 x)}}{1+\exp{(\beta_o + \beta_1 x)}}.
    $$

    For example, the stimulus is the dose level of a drug and the response is cured or is notcured. Find the linearizing transformation $h(p)$ so that $h(p) = \beta_0 + \beta_1 x$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Taking the inverse on both sides will lead to the linearizing transformation

    $$
    h(p) = \ln{\frac{p}{1-p}} = \beta_0 + \beta_1 x.
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, pl):
    df_ex16 = pl.read_json("../SDAEI-Tamhane/ch10/Ex10-16.json").explode(pl.all())

    mo.md(
        rf"""
    ### Ex 10.16

    A prime is a positive integer that has no integer factors other than 1 and itself (1 is not regarded as a prime). The number of primes in any given interval of whole numbers is highly irregular. However, the proportion of primes less than or equal to any given number $x$ (denoted by $p(x)$) follows a regular pattern as $x$ increases. The following table gives the number and proportion of primes for $x = 10^n$ for $n = 1, 2, \ldots, 10$. The objective of the present exercise is to discover this pattern.

    {
            mo.ui.table(
                df_ex16,
                selection=None,
                show_data_types=False,
            )
        }
    """
    )
    return (df_ex16,)


@app.cell(hide_code=True)
def _(alt, df_ex16, mo, pl):
    _df = df_ex16.with_columns(
        x=10.0 ** pl.col("x").str.extract("\\^(\\d+)").cast(int)
    ).with_columns(
        h1=10000 / pl.col("x"),
        h2=1000 / pl.col("x").sqrt(),
        h3=1 / pl.col("x").log10(),
    )

    _chart = (
        alt.Chart(_df)
        .mark_line(point=True)
        .encode(alt.X(alt.repeat("column"), type="quantitative"), alt.Y("Prop"))
        .properties(width=200)
        .repeat(column=["h1", "h2", "h3"])
    )

    mo.md(
        r"""
    /// details | (a) Plot the proportion of primes, $p(x)$, against $10,000/x$, $1000/\sqrt{x}$, and $1 / \log_{10}x$. Which relationship appears most linear?

    In the charts below, h1 = $10,000/x$, h2 = $1000/\sqrt{x}$, and h3 = $1 / \log_{10}x$. Apparently the relationship between $1 / \log_{10}x$ and $p(x)$ appears most linear.
    """
        f"""
    {mo.as_html(_chart)}
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex16, linreg, mo, pl):
    _res = (
        df_ex16.with_columns(
            x=10.0 ** pl.col("x").str.extract("\\^(\\d+)").cast(int)
        )
        .with_columns(
            h3=1 / pl.col("x").log10(),
        )
        .select(linreg(pl.col("h3"), pl.col("Prop")))
        .item()
    )

    mo.md(
        rf"""
    /// details | (b) Estimate the slope of the line $p(x) = \beta_0 + \beta_1 \cdot 1/\log_{{10}}x$ and show that $\hat{{\beta}}_1 \approx \log_{{10}}e = 0.4343$.

    Using `linreg` and $\alpha$ = 0.05, $\hat{{\beta}}_0 = {_res["β0"]:.4f}$ and $\hat{{\beta}}_1 = {_res["β1"]:.4f} \approx \log_{{10}}e = 0.4343$.


    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | (c) Explain how the relationship found in (b) roughly translates into the _prime number theorem_: For large $x$, $p(x) \approx 1 / \log_e x$.

    TODO
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, pl):
    df_ex17 = pl.read_json("../SDAEI-Tamhane/ch10/Ex10-17.json").explode(pl.all())

    mo.md(
        rf"""
    ### Ex 10.17

    In a memory retention experiment subjects were asked to memorize a list of disconnected items, and then were asked to recall them at various times up to a week later. The proportion $p$ of items recalled at times $t$ (in minutes) is given below.

    {
            mo.ui.table(
                df_ex17,
                show_column_summaries=False,
                selection=None,
                show_data_types=False,
            )
        }
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | (a) Note that $t$ increases almost geometrically throughout. This suggests that a logarithmic transformation of $t$ might linearize the relationship. Plot $p$ vs. $\ln{t}$. Is the relationship approximately linear?

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | (b) Fit a trend line to the plot in (a). From the trend line estimate the time for 50% retention.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, pl):
    df_ex18 = pl.read_json("../SDAEI-Tamhane/ch10/Ex10-18.json").explode(pl.all())

    mo.md(
        rf"""
    ### Ex 10.18

    The following are the average distances of the planets in the solar system from the sun:

    {
            mo.ui.table(
                df_ex18,
                label="(distances are in millions of miles.)",
                show_column_summaries=False,
                selection=None,
                show_data_types=False,
            )
        }

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | (a) How does the distance of a planet from the sun increase with the planet number? Find a transformation of the distance that gives a linear relationship with respect to the planet number.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | (b) Fit a least squares straight line after linearizing the relationship.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | (c) It is speculated that there is a planet beyond Pluto, called Planet X. Predict its distance from the sun.

    ///
    """
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
