import marimo

__generated_with = "0.16.2"
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
    from great_tables import GT, md, html

    import inspect


    def get_source(func) -> str:
        """Display a function's source code as markdown"""
        source = inspect.getsource(func)
        return f"""```python
    {source}
    ```"""


    def df_ex(n: int) -> pl.DataFrame:
        """return dataframe of exercise number n"""
        fname = f"../SDAEI-Tamhane/ch10/Ex10-{n}.json"
        # print(f'loading exercise data from "{fname}"')
        return pl.read_json(fname).explode(pl.all())
    return alt, df_ex, get_source, html, md, mo, np, pl, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 10.1 A Probabilistic Model for Simple Linear Regression""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.ui.tabs(
        {
            "Theory": mo.md(
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
            )
        }
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
    mo.md(
        r"""
    /// details | 1. Maxwell's equations of electromagnetism

    theoretical and deterministic
    ///

    /// details | 2. An econometric model of the U.S. economy.

    empirical and probabilistic
    ///

    /// details | 3. A credit scoring model for the probability of a credit applicant being a good risk as a function of selected variables, e.g., income, outstanding debts, etc.

    empirical and probabilistic
    ///
    """
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
    mo.md(
        r"""
    /// details | 1. An item response model for the probability of a correct response to an item on a "true-false" test as a function of the item's intrinsic difficulty.

    empirical and probabilistic
    ///

    /// details | 2. The Cobb-Douglas production function, which relates the output of a firm to its capital and labor inputs.

    empirical and probabilistic
    ///

    /// details | 3. Kepler's laws of planetary motion.

    theoretical and deterministic
    ///
    """
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
def _(get_source, mo, pl):
    def linreg(x: pl.Expr, y: pl.Expr, x_star: float | None = None) -> pl.Expr:
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
            n.alias("n"),
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


    mo.ui.tabs(
        {
            "Theory": mo.md(
                r"""
    The _least squares(LS) estimates_ $\hat{\beta}_0$ and $\hat{\beta}_1$ minimize $Q = \sum_{i=1}^n [y_i - (\beta_0 + \beta_1 x_i) ]^2$ and are given by

    $$
    \begin{align*}
    \hat{\beta}_1 &= \frac{S_{xy}}{S_{xx}}, \\
    \hat{\beta}_0 &= \bar{y} - \hat{\beta}_1 \bar{x}
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
            "Implementation": mo.md(rf"""
    The `scipy` function [`stats.linregress`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html) can be used to get many results of a simple linear regression.

    The `linreg` function of polars expressions defined below follows directly from the formulas in the book, yielding $\beta_0$ and $\beta_1$ among others. We'll use this homemade function for the exercises.

    {get_source(linreg)}
    """),
        }
    )
    return (linreg,)


@app.cell(hide_code=True)
def _(df_ex, html, md, mo):
    mo.md(
        rf"""
    ### Ex 10.4

    The time between eruptions of Old Faithful geyser in Yellowstone National Park is random but is related to the duration of the last eruption. The table below shows these times for 21 consecutive eruptions.

    {
            mo.center(
                mo.as_html(
                    df_ex(4)
                    .style.tab_options(
                        table_font_size=12,
                        table_width="60%",
                        container_height="50vh",
                    )
                    .tab_header(
                        title="Old Faithful Eruptions: Duration and Time Between Eruptions (in min.)"
                    )
                    .tab_stub(rowname_col="No")
                    .tab_stubhead(label="Obs. No.")
                    .cols_label(
                        LAST=html("Duration of Eruption<br>(LAST)"),
                        NEXT=html("Time Between Eruptions<br>(NEXT)"),
                    )
                    .fmt_number(columns=["No", "NEXT"], drop_trailing_zeros=True)
                    .cols_align(align="center")
                    .tab_source_note(
                        source_note=md(
                            'Source: L. Denby and D. Pregibon ( 1987), "An example of the use of graphics in regression". _The American Statistician_, 41, pp. 33-38.'
                        )
                    )
                )
            )
        }

    Let us see how well we can predict the time to next eruption, given the length of time of the last eruption.
    """
    )
    return


@app.cell(hide_code=True)
def _(alt, df_ex, linreg, mo, pl):
    _df = df_ex(4)
    _res = _df.select(linreg(pl.col("LAST"), pl.col("NEXT"))).item()
    _β0, _β1, _r2, _s2 = _res["β0"], _res["β1"], _res["r2"], _res["s2"]

    _scatter = _df.plot.scatter(
        alt.X("LAST").scale(domain=[1, 5.5]),
        alt.Y("NEXT").scale(domain=[30, 100]),
    )

    _line = _scatter.transform_regression("LAST", "NEXT").mark_line(color="red")


    mo.md(
        rf"""
    /// details | (a) Make a scatter plot of NEXT vs. LAST. Does the relationship appear to be approximately linear?

    {mo.ui.altair_chart(_scatter)}

    Yes, the points appear approximately linear.
    ///

    /// details | (b) Fit a least squares regression line. Use it to predict the time to the next eruption if the last eruption lasted 3 minutes.

    When applied to the dataset, `linreg` gives $\beta_0$ = {_β0:.2f} and $\beta_1$ = {_β1:.2f}. If the last eruption lasted 3 minutes, the time to the next eruption would be in about

    $$
    \beta_0 + \beta_1 \cdot 3 = {_β0:.2f} + {_β1:.2f} \cdot 3 = {_β0 + _β1 * 3:.2f}
    $$

    minutes.

    {mo.as_html(_scatter + _line)}
    ///

    /// details | (c) What proportion of variability in NEXT is accounted for by LAST? Does it suggest that LAST is a good predictor of NEXT?

    $r^2$ = {_r2:.2f}, suggesting that `LAST` is a pretty good predictor of `NEXT`.
    ///

    /// details | (d) Calculate the mean square error estimate of $\sigma$.

    $s^2$ = {_s2:.2f}, and the mean square error estimate of $\sigma$ is $s$ = {_s2**0.5:.2f}.
    ///

                """
    )
    return


@app.cell(hide_code=True)
def _(df_ex, md, mo):
    mo.md(
        f"""
    ### Ex 10.5

    The data below show Olympic triple jump winning distances for men in meters for the years 1896 to 1992 (there were no Olympic games in 1916, 1940, and 1944).

    {
            mo.center(
                mo.as_html(
                    df_ex(5)
                    .style.tab_options(
                        table_font_size=12,
                        table_width="60%",
                        container_height="50vh",
                    )
                    .cols_align("center")
                    .tab_header(
                        title="Men's Olympic Triple Jump Winning Distance (in meters)"
                    )
                    .fmt_integer(columns="Year", use_seps=False)
                    .tab_source_note(
                        source_note=md(
                            "Source: _World Almanac and Book of Facts_ (1995), Mahwah. NJ: Funk & Wagnalls Corporation, p. 860."
                        )
                    )
                )
            )
        }
    """
    )
    return


@app.cell(hide_code=True)
def _(alt, df_ex, linreg, mo, pl):
    _df = df_ex(5)
    _res = _df.select(linreg(pl.col("Year"), pl.col("Distance"))).item()
    _β0, _β1, _r2, _s2 = _res["β0"], _res["β1"], _res["r2"], _res["s2"]

    _scatter = _df.with_columns(pl.col("Year").cast(int).cast(str)).plot.scatter(
        alt.X("Year:T"),
        alt.Y("Distance").scale(domain=[13, 19]),
    )
    _line = _scatter.transform_regression("Year", "Distance").mark_line(
        color="red"
    )

    mo.md(
        rf"""
    /// details | (a) Make a scatter plot of the length of the jump by year. Does the relationship appear to be approximately linear?

    {mo.as_html(_scatter)}

    Yes, the points appear approximately linear.
    ///

    /// details |  (b) Fit a least squares regression line.

    Using `linreg`, $\beta_0$ = {_β0:.2f} and $\beta_1$ = {_β1:.3f}.

    {mo.ui.altair_chart(_line + _scatter)}
    ///

    /// details | (c) Calculate the mean square error estimate of $\sigma$.

    $s^2$ = {_s2:.3f}, and the mean square error estimate of $\sigma$ is $s$ = {_s2**0.5:.3f}.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex, md, mo):
    mo.md(
        f"""
    ### Ex 10.6

    The following data give the barometric pressure (in inches of mercury) and the boiling point (in °F) of water in the Alps.

    {
            mo.center(
                mo.as_html(
                    df_ex(6)
                    .style.tab_options(
                        table_font_size=12,
                        table_width="50%",
                    )
                    .cols_align("center")
                    .tab_header(title="Boiling Point of Water in the Alps")
                    .tab_source_note(
                        source_note=md(
                            "Source: A. C. Atkinson (1985), _Plots, Transformations and Regression_, Oxford: Clarendon Press. p. 4. Reprinted in _Small Data Sets_, pp. 270-271."
                        )
                    )
                )
            )
        }
    """
    )
    return


@app.cell(hide_code=True)
def _(alt, df_ex, linreg, mo, pl):
    _df = df_ex(6)
    _res = _df.select(linreg(pl.col("Pressure"), pl.col("Temp"))).item()

    _scatter = _df.plot.scatter(
        alt.X("Pressure").scale(domain=[20, 31]),
        alt.Y("Temp").scale(domain=[192, 215]),
    )

    _line = _scatter.transform_regression("Pressure", "Temp").mark_line(
        color="red"
    )

    mo.md(
        rf"""
    /// details | (a) Make a scatter plot of the boiling point by barometric pressure. Does the relationship appear to be approximately linear?

    {mo.ui.altair_chart(_scatter)}

    Yes, the relationship is approximately linear.
    ///

    /// details | (b) Fit a least squares regression line. What proportion of variation in the boiling point is accounted for by linear regression on the barometric pressure?

    Using `linreg`, $\beta_0$ = {_res["β0"]:.2f}, $\beta_1$ = {_res["β1"]:.3f}, and $r^2$ = {_res["r2"]:.3f}. That is, {_res["r2"] * 100:.1f}% percent of variation in the boiling point is accounted for by linear regression on the barometric pressure.

    {mo.ui.altair_chart(_line + _scatter)}
    ///

    /// details | (c) Calculate the mean square error estimate of $\sigma$.

    Also using the `linreg` function, $s^2$ = {_res["s2"]:.3f}, and the mean square error estimate of $\sigma$ is $s$ = {_res["s2"] ** 0.5:.3f}.

    ///"""
    )
    return


@app.cell(hide_code=True)
def _(df_ex, md, mo):
    mo.md(
        rf"""
    ### Ex 10.7

    The following table shows Olympic 100 meter backstroke winning times for women for the years 1924 to 1992 (there were no Olympic games in 1940 and 1944).

    {
            mo.center(
                mo.as_html(
                    df_ex(7)
                    .style.tab_options(
                        table_font_size=12,
                        table_width="60%",
                    )
                    .cols_align("center")
                    .tab_header(
                        title="Women's Olympic 100 Meter Backstroke Winning Times (in seconds)"
                    )
                    .fmt_integer(columns="Year", use_seps=False)
                    .tab_source_note(
                        source_note=md(
                            "Source: _The World Almanac and Book of Facts_ (1995). Mahwah, NJ: Funk & Wagnalls Corporation. p. 864."
                        )
                    )
                )
            )
        }
    """
    )
    return


@app.cell(hide_code=True)
def _(alt, df_ex, linreg, mo, pl):
    _df = df_ex(7)
    _res = _df.select(linreg(pl.col("Year"), pl.col("Time"))).item()

    _scatter = _df.with_columns(pl.col("Year").cast(int).cast(str)).plot.scatter(
        alt.X("Year:T"),
        alt.Y("Time").scale(domain=[55, 90]),
    )

    _line = _scatter.transform_regression(
        "Year", "Time"
    ).mark_line(color="red")

    mo.md(
        rf"""
    /// details | (a) Make a scatter plot of the winning times by year. Does the relationship appear to be approximately linear?

    {mo.ui.altair_chart(_scatter)}

    Yes, the relationship is approximately linear.
    ///

    /// details | (b) Fit a least squares regression line.

    Using `linreg`, $\beta_0$ = {_res['β0']:.1f} and $\beta_1$ = {_res['β1']:.3f}.

    {mo.ui.altair_chart(_line + _scatter)}
    ///

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
def _(get_source, linreg, mo, pl, stats):
    def slope_test(
        df: pl.DataFrame,
        x: str,
        y: str,
        b: float = 0,
        alternative: str = "two-sided",
    ) -> tuple[float, float]:
        """
        returns (t-statistic, P-value) of observed data under H0: beta_1 = b

        input:
        - df: the dataset (polars dataframe)
        - x, y: column names for observations
        - b: given constant to compare with beta_1. defaults to 0
        - alternative:
            'two-sided': beta_1 ≠ b which is the default
            'less': beta_1 < b
            'greater': beta_1 > b

        output:
        a tuple (t, pval) containing the t-statisic and the P-value
        """
        res = df.select(linreg(pl.col(x), pl.col(y))).item()
        t = (res["β1"] - b) / res["se_β1"]
        n = res["n"]
        match alternative:
            case "two-sided":
                pval = 2 * stats.t.sf(abs(t), n - 2).item()
            case "greater":
                pval = stats.t.sf(t, n - 2).item()
            case "less":
                pval = stats.t.cdf(t, n - 2).item()
            case _:
                raise ValueError("unknown alternative value.")
        return (t, pval)


    def estimate_interval(
        df: pl.DataFrame,
        x: str,
        y: str,
        x_star: float,
        PI: bool = False,
        α: float = 0.05,
    ) -> list[float]:
        """
        Gives the confidence/prediction interval for estimation.

        Input:
        - df: the dataset (polars dataframe)
        - x, y: column names for observations
        - x_star: input point for estimation
        - PI: defaults to False (calculate a CI)
        - α: significance level

        Output:
        the list [low, high] designating the calculated interval.
        """
        res = df.select(linreg(pl.col(x), pl.col(y), x_star=x_star)).item()
        y_star = res["β0"] + res["β1"] * x_star
        t_crit = stats.t.ppf(1 - α / 2, res["n"] - 2).item()  # critical value
        se = res["se_est_pi"] if PI else res["se_est_ci"]
        return [y_star - t_crit * se, y_star + t_crit * se]


    mo.ui.tabs(
        {
            "Theory": mo.md(
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

    A $100(1-\alpha)\%$ CI for $\mu^*$ is given by

    $$
    \hat{\mu}^* \pm t_{n-2, \alpha/2} s \sqrt{\frac{1}{n} + \frac{(x^* - \bar{x})^2}{S_{xx}}}.
    $$

    A $100(1-\alpha)$% _prediction interval_ (PI) for $Y^*$ is given by

    $$
    \hat{Y}^* \pm t_{n-2, \alpha/2} s \sqrt{1 + \frac{1}{n} + \frac{(x^* - \bar{x})^2}{S_{xx}}}.
    $$

    However, a $100(1-\alpha)$% PI for $Y^*$ is wider than a CI for $\mu^*$, because $Y^*$ is an r.v., while $\mu^*$ is a fixed constant.
    """
            ),
            "Implementation": mo.md(rf"""
    The `scipy` function [`stats.linregress`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html) can also be used to test if $\beta_1 = 0$.

    Here we define a more general `slope_test` to test if $\beta_1$ is differenct than a given slope.

    {get_source(slope_test)}

    We also define a function `estimate_interval` to calculate the CI/PI for estimation.

    {get_source(estimate_interval)}
        """),
        }
    )
    return estimate_interval, slope_test


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
def _(df_ex, estimate_interval, mo, slope_test):
    _df = df_ex(5)

    _t, _pval = slope_test(_df, "Year", "Distance")

    [_low, _high] = estimate_interval(_df, "Year", "Distance", 2004, PI=True)

    mo.md(
        rf"""
    /// details | (a) Is there a significant increasing linear trend in the triple jump distance? Test at $\alpha = .05$.

    This is a test of $H_0: \beta_1 \le 0$ and the $t$-statistic is {_t:.2f} with a one-sided $P$-value of {_pval:.2e} < $\alpha$. So yes there is a significant increasing trend.
    ///

    /// details | (b) Calculate a 95% PI for the winning jump in 2004. Do you think this prediction is reliable? Why or why not? Would a 95% CI for the winning jump in 2004 have a meaningful interpretation? Explain.

    A 95% PI for the winning jump in 2004 is [{_low:.2f}, {_high:.2f}], but it is not reliable since we are extrapolating. In this case, a CI is not meaningful because there will be at most a single winning jump in 2004. 
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
def _(df_ex, estimate_interval, mo):
    _df = df_ex(6)
    [_low, _high] = estimate_interval(_df, "Pressure", "Temp", 28)

    mo.output.append(
        mo.md(
            rf"""
    /// details | (a) Calculate a 95% CI for the boiling point if the barometric pressure is 28 inches of mercury. Interpret your CI.

    The said CI is calculated to be [{_low:.2f}, {_high:.2f}]. That is to say, there is a 95% chance that this interval includes the boiling point at 28 inches of of mercury on the true regression line.
    ///"""
        )
    )

    [_low, _high] = estimate_interval(_df, "Pressure", "Temp", 31)

    mo.output.append(
        mo.md(
            rf"""
    /// details | (b) Calculate a 95% CI for the boiling point if the barometric pressure is 31 inches of mercury. Compare this with the CI of (a).

    The said CI is calculated to be [{_low:.2f}, {_high:.2f}]. It is much wider than (a) at 28 inches of mercury and should be treated as unreliable because we are extrapolating outside the data domain.
    /// """
        )
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
def _(df_ex, estimate_interval, mo):
    _df = df_ex(4)

    [_low, _high] = estimate_interval(_df, x="LAST", y="NEXT", x_star=3, PI=True)

    mo.output.append(
        mo.md(
            rf"""
    /// details | (a) Calculate a 95% PI for the time to the next eruption if the last eruption lasted 3 minutes.

    The said PI is calculated to be [{_low:.2f}, {_high:.2f}].
    ///"""
        )
    )

    [_low, _high] = estimate_interval(_df, x="LAST", y="NEXT", x_star=3, PI=False)

    mo.output.append(
        mo.md(
            rf"""
    /// details | (b) Calculate a 95% CI for the mean time to the next eruption for a last eruption lasting 3 minutes. Compare this CI with the PI obtained in (a).

    The said CI is calculated to be [{_low:.2f}, {_high:.2f}] which is a lot narrower than the PI in (a).
    ///"""
        )
    )

    [_low, _high] = estimate_interval(_df, x="LAST", y="NEXT", x_star=1, PI=True)

    mo.output.append(
        mo.md(
            rf"""
    /// details | (c) Repeat (a) if the last eruption lasted 1 minute. Do you think this prediction is reliable? Why or why not?

    The PI for a 1 minute last eruption is [{_low:.2f}, {_high:.2f}] which is unreliable because we are extrapolating outside of the data domain.
    """
        )
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
def _(df_ex, estimate_interval, linreg, mo, pl):
    _df = df_ex(7)
    [_low, _high] = estimate_interval(_df, "Year", "Time", 2004, PI=True)


    _y = 60
    _res = _df.select(linreg(pl.col("Year"), pl.col("Time"))).item()
    _x = (_y - _res["β0"]) / _res["β1"]

    mo.md(
        rf"""
    /// details | (a) Calculate a 95% PI for the winning time in 2004. Do you think this prediction is reliable? Why or why not?

    The specified PI is calculated to be [{_low:.2f}, {_high:.2f}]. However, this prediction is unreliable because we are extrapolating outside the data range (latest available year was 1996).
    ///

    /// details | (b) Use the regression equation to find the year in which the winning time would break 1 minute. Given that the Olympics are every four years, during which Olympics would this happen?

    The year {_x:.0f} (by inverse regression).
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex, md, mo):
    mo.md(
        rf"""
    ### Ex 10.13

    The U.S. infant mortality rates (IMR) (per 1000 live births) for both sexes and all races for the years 1981-1990 (coded as years 1-10) were as follows:

    {
            mo.center(
                mo.as_html(
                    df_ex(13)
                    .style.tab_options(
                        table_font_size=12,
                        table_width="60%",
                    )
                    .cols_align("center")
                    .fmt_integer(columns="Year", use_seps=False)
                    .tab_source_note(
                        source_note=md(
                            "Source: _The World Almanac and Book of Facts_ (1994), Mahwah, NJ: Funk & Wagnalls Corporation, p. 956."
                        )
                    )
                )
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
    return


@app.cell(hide_code=True)
def _(mo, np, stats):
    _β1 = 0.2697
    _β10 = 0.3
    _se = 0.01372
    _n = 10
    _t = (_β1 - _β10) / _se
    _pval = stats.t.cdf(_t, _n - 2)

    mo.output.append(
        mo.md(
            rf"""
    /// details | (a) What was the average rate of decrease in the infant mortality rate per year during 1981-1990? Suppose that for the rest of the Western world the rate of decrease in IMR was 0.3 deaths per year per 1000 live births. Was the U.S. rate of decrease significantly less than that for the rest of the Western world? Use $\alpha$ = .05.

    Reading off the MINITAB output, the rate of IMR decrease is {_β1} deaths per 1,000 live births per year. 

    Set up $H_0: \beta_1 \ge 0.3$ and use the MINITAB output $\textrm{{SE}}(\hat{{\beta}}_1)$ = {_se} to get a $t$-statistic of {_t:.3f} with $P$-value {_pval:.3f} < $\alpha$. So yes, the US IMR decrease is less than that for the rest of the Western world.
    /// """
        )
    )

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

    mo.output.append(
        mo.md(
            rf"""
    /// details | (b) Predict the IMR for the year 1995. Calculate a 95% prediction interval. (Note that $S_{{xx}}$ can be obtained from the values of $\textrm{{Stdev}}(\hat{{\beta}}_1)$ and $s$ given in the MINITAB output.)

    $S_{{xx}}$ can be obtained as in the note, or simply $S_{{xx}} = \textrm{{SSR}}/\hat{{\beta}}_1^2$, and both $\textrm{{SSR}}$ and $\hat{{\beta}}_1$ are in the MINITAB output. Anyways, the calculated IMR for the year 1995 is {_y:.2f} ± {_err:.2f} = [{_y - _err:.2f}, {_y + _err:.2f}].
    ///
    """
        )
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
def _(df_ex, md, mo):
    mo.md(
        rf"""
    ### Ex 10.16

    A prime is a positive integer that has no integer factors other than 1 and itself (1 is not regarded as a prime). The number of primes in any given interval of whole numbers is highly irregular. However, the proportion of primes less than or equal to any given number $x$ (denoted by $p(x)$) follows a regular pattern as $x$ increases. The following table gives the number and proportion of primes for $x = 10^n$ for $n = 1, 2, \ldots, 10$. The objective of the present exercise is to discover this pattern.

    {
            mo.center(
                mo.as_html(
                    df_ex(16)
                    .style.tab_options(
                        table_font_size=12,
                        table_width="70%",
                    )
                    .cols_align("center")
                    .fmt_integer(columns="Primes")
                    .tab_source_note(
                        source_note=md(
                            "Source: W. Dunham (1994), _The Mathematical Universe_, New York: Wiley, p. 196."
                        )
                    )
                )
            )
        }
    """
    )
    return


@app.cell(hide_code=True)
def _(alt, df_ex, mo, pl):
    _df = (
        df_ex(16)
        .with_columns(x=10.0 ** pl.col("x").str.extract("\\^(\\d+)").cast(int))
        .with_columns(
            h1=10000 / pl.col("x"),
            h2=1000 / pl.col("x").sqrt(),
            h3=1 / pl.col("x").log10(),
        )
    )

    _chart = (
        alt.Chart(_df)
        .mark_line(point=True)
        .encode(alt.X(alt.repeat("column"), type="quantitative"), alt.Y("Prop"))
        .properties(width=200)
        .repeat(column=["h1", "h2", "h3"])
    )

    # regression through the origin
    _β1 = _df.select(
        (pl.col("Prop") ** 2).sum() / (pl.col("h3") * pl.col("Prop")).sum()
    ).item()

    mo.md(
        r"""
    /// details | (a) Plot the proportion of primes, $p(x)$, against $10,000/x$, $1000/\sqrt{x}$, and $1 / \log_{10}x$. Which relationship appears most linear?

    In the charts below, h1 = $10,000/x$, h2 = $1000/\sqrt{x}$, and h3 = $1 / \log_{10}x$. Apparently the relationship between $1 / \log_{10}x$ and $p(x)$ appears most linear. """
        f"""
    {mo.as_html(_chart)}
    ///
    """
        r"""
    /// details | (b) Estimate the slope of the line $p(x) = \beta_0 + \beta_1 \cdot 1/\log_{10}x$ and show that $\hat{\beta}_1 \approx \log_{10}e = 0.4343$.

    Because $p(x) \to 0$ as $\textrm{h3} \to 0$, it's appropriate to assume $\beta_0 = 0$ and use _regression through the origin_ (Exercise 8) to calculate """
        rf"""

    $$
    \hat{{\beta}}_1 = {_β1:.4f} \approx  \log_{{10}}e = 0.4343.
    $$
    ///
    """
        r"""
    /// details | (c) Explain how the relationship found in (b) roughly translates into the _prime number theorem_: For large $x$, $p(x) \approx 1 / \log_e x$.

    For large $x$,

    $$
    \begin{align*}
    p(x) &= \beta_1 \frac{1}{\log_{10}x} \\
    &\approx \frac{\log_{10}e}{{\log_{10}x}} \\
    &=  \frac{1}{\log_e x}.
    \end{align*}
    $$
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex, md, mo):
    mo.md(
        rf"""
    ### Ex 10.17

    In a memory retention experiment subjects were asked to memorize a list of disconnected items, and then were asked to recall them at various times up to a week later. The proportion $p$ of items recalled at times $t$ (in minutes) is given below.

    {
            mo.center(
                mo.as_html(
                    df_ex(17)
                    .style.tab_options(
                        table_font_size=12,
                        table_width="50%",
                    )
                    .cols_align("center")
                    .cols_label(P="p")
                    .fmt_integer(columns="t")
                    .tab_source_note(
                        source_note=md(
                            "Source: F. Mosteller, R. E. K. Rourke, and G. B. Thomas (1970), _Probability with Statistical Applications_, 2nd ed., Reading, MA: Addison-Wesley. Reprinted in _Small Data Sets_, p. 128."
                        )
                    )
                )
            )
        }
    """
    )
    return


@app.cell(hide_code=True)
def _(alt, df_ex, linreg, mo, np, pl):
    _df = df_ex(17).with_columns(h=pl.col("t").log())

    _res = _df.select(linreg(pl.col("h"), pl.col("P"))).item()
    _β0, _β1 = _res["β0"], _res["β1"]

    _scatter = _df.plot.scatter(
        alt.X("h").title("h = ln t"), alt.Y("P").title("p")
    )
    _line = _scatter.transform_regression("h", "P").mark_line(color="red")

    mo.md(
        r"""
    /// details | (a) Note that $t$ increases almost geometrically throughout. This suggests that a logarithmic transformation of $t$ might linearize the relationship. Plot $p$ vs. $\ln{t}$. Is the relationship approximately linear?

    """
        rf"""{mo.ui.altair_chart(_scatter)}

    Yes. The relationship appears approximately linear.
    ///

    /// details | (b) Fit a trend line to the plot in (a). From the trend line estimate the time for 50% retention.

    The trend line is $\hat p = {_β0:.3f} - {abs(_β1):.3f}h$.

    {mo.ui.altair_chart(_line + _scatter)}

    """
        r"""
    Because $p = \beta_0 + \beta_1\;\ln{t}$, $t=\exp{\frac{p-\beta_0}{\beta_1}}$.
    """
        rf""" For p = 50% retention, $t = {np.exp((0.5 - _β0) / _β1).item():.1f}$ minutes.
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex, html, md, mo):
    mo.md(
        rf"""
    ### Ex 10.18

    The following are the average distances of the planets in the solar system from the sun:

    {
            mo.center(
                mo.as_html(
                    df_ex(18)
                    .style.tab_options(
                        table_font_size=12,
                        table_width="70%",
                    )
                    .cols_align("center")
                    .tab_stub(rowname_col="No")
                    .tab_stubhead(label="Planet No.")
                    .cols_label(Dist=html("Distance<br>(millions of miles)"))
                    .fmt_integer(columns="No")
                    .tab_source_note(
                        source_note=md(
                            "This exercise is based on Example 6, Ch. 3 of F. Mosteller. S. E. Fienherg, and R. E. K. Rourke (1983). _Beginning Stalislics with Data Analysis_, Reading. MA: Addison-Wesley."
                        )
                    )
                )
            )
        }
    """
    )
    return


@app.cell(hide_code=True)
def _(alt, df_ex, linreg, mo, np, pl):
    _df = df_ex(18).with_columns(h=pl.col("Dist").log())

    _scatter = _df.plot.scatter(
        alt.X("No").scale(domain=[0, 11]).title("x=Planet No."),
        alt.Y("h").scale(domain=[3, 9]).title("h=ln(Distance)"),
    )
    _line = _scatter.transform_regression("No", "h").mark_line(color="red")

    _res = _df.select(linreg(pl.col("No"), pl.col("h"))).item()
    _β0, _β1 = _res["β0"], _res["β1"]

    mo.md(
        rf"""
    /// details | (a) How does the distance of a planet from the sun increase with the planet number? Find a transformation of the distance that gives a linear relationship with respect to the planet number.

    The distances seem to increase exponentially with the planet number at a factor of 1.5 - 2. Therefore we take the logarithm of the distances.

    {mo.as_html(_scatter)}

    Yeah, the transformation appears to give a linear relationship.
    ///

    /// details | (b) Fit a least squares straight line after linearizing the relationship.

    By `linreg`, the least squares line is $h = {_β0:.3f} + {_β1:.2f}\;x$.

    {mo.ui.altair_chart(_line + _scatter)}
    ///

    /// details | (c) It is speculated that there is a planet beyond Pluto, called Planet X. Predict its distance from the sun.

    $h^*={_β0:.3f} + {_β1:.2f} \cdot 11 = {_β0 + _β1 * 11:.3f}$. So distance = $\exp{{(h^*)}}$ = {np.exp(_β0 + _β1 * 11).item():.1f} millions of miles.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex, html, mo):
    mo.md(
        rf"""
    ### Ex 10.19

    The following are the speeds of the planets in the solar system as they revolve around the sun:

    {
            mo.center(
                mo.as_html(
                    df_ex(19)
                    .style.tab_options(
                        table_font_size=12,
                    )
                    .cols_align("center")
                    .tab_stub(rowname_col="No")
                    .tab_stubhead(label="Planet No.")
                    .cols_label(
                        Distan=html("Distance<br>(millions of miles)"),
                        Speed=html("Speed<br>(miles per hour)"),
                    )
                    .cols_move_to_start(columns="Planet")
                    .fmt_integer(columns=["No", "Speed"])
                )
            )
        }

    What relationship do you notice between the distances of the planets from the sun given in the previous exercise and planet speeds? Find a transformation of the speed that linearizes its relationship with the planet distance, and fit this relationship.
    """
    )
    return


@app.cell(hide_code=True)
def _(alt, df_ex, mo, pl):
    _df = df_ex(19).with_columns(h=1 / pl.col("Speed") ** 2)

    _scatter = _df.plot.scatter(
        alt.X("Distan", title="distance"),
        alt.Y("h", title="h=speed^(-2)", axis=alt.Axis(format="g")),
    )

    _β1 = _df.select(
        (pl.col("Distan") * pl.col("h")).sum() / (pl.col("Distan") ** 2).sum()
    ).item()

    mo.md(
        r""" 
    Highschool physics tells us that a planet orbiting the sun at distance $R$ and speed $v$ will have centripetal acceleration $g = v^2/R$, which is provided by gravity $g = GM/R^2$. Therefore 

    $$
    R = \frac{GM}{v^2}.
    $$

    So the transformation we are looking for is $h = 1/\textrm{speed}^2$, which is confirmed by the scatter plot that this is a linear relationship.
    """
        rf"""
    {mo.as_html(_scatter)}

    To fit this linear relationship, we should force $\beta_0 = 0$ and use _regression through the origin_. $\hat{{\beta}}_1$ = {_β1:.2e}.
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex, html, md, mo):
    mo.md(
        rf"""
    ### Ex 10.20

    To relate the stopping distance of a car to its speed, ten cars were tested at five different speeds, two cars at each speed. The following data were obtained.

    {
            mo.center(
                mo.as_html(
                    df_ex(20)
                    .style.tab_options(
                        table_font_size=12,
                        table_width="60%",
                    )
                    .cols_align("center")
                    .cols_label(
                        x=html("Speed x<br>(mph)"),
                        y=html("Stop. Dist. y<br>(ft)"),
                    )
                    .fmt_integer(columns="x")
                    .tab_source_note(
                        source_note=md(
                            "This exercise is based on Example 2A. Ch. 12 of F. Mosteller, S. E. Fienberg and R. E. K. Rourke. _op. cit._"
                        )
                    )
                )
            )
        }
    """
    )
    return


@app.cell(hide_code=True)
def _(alt, df_ex, linreg, mo, pl, stats):
    _df = df_ex(20)
    _res = _df.select(linreg(pl.col("x"), pl.col("y"))).item()
    _β0, _β1, _f, _n = _res["β0"], _res["β1"], _res["f"], _res["n"]
    _pval = stats.f.sf(_f, 1, _n - 2)

    _scatter = _df.plot.scatter(
        x=alt.X("x", title="speed").scale(domain=[15, 65]),
        y=alt.Y("y", title="stop distance").scale(domain=[0, 230]),
    ).properties(title="LS fit")
    _line = _scatter.transform_regression("x", "y").mark_line(color="red")

    _df = _df.with_columns(e=pl.col("y") - (_β0 + _β1 * pl.col("x")), y0=0)
    _error = _df.plot.circle(
        x=alt.X("x", title="speed").scale(domain=[15, 65]),
        y=alt.Y("e", title="error").scale(domain=[-40, 50]),
    ).properties(title="residuals")
    _rule = _df.plot.rule(x="x", y="y0", y2="e")

    mo.output.append(
        mo.md(
            rf"""
    /// details | (a) Fit an LS straight line to these data. Plot the residuals against the speed.

    The fitted line is $y = {_β0:.2f} +  {_β1:.2f}x$.

    {mo.as_html((_scatter + _line) | (_error + _rule))}
    ///

    /// details | (b) Comment on the goodness of the fit based on the overall $F$-statistic and the residual plot. Which two assumptions of the linear regression model seem to be violated?

    The $F$-statistic is {_f:.2f} with $P$-value = {_pval:.2e}, showing that the trend clearly has a significant linear component. However, the residual plot reveals that two assumptions seem to be violated: (1) linearity - a systematic, parabolic pattern indicates the regression does not fit the data adequately; and (2) constant variance - the error variance seem to get bigger with $x$.
    ///

    /// details | (c) Based on the residual plot, what transformation of stopping distance should be used to linearize the relationship with respect to speed? A clue to find this transformation is provided by the following engineering argument: In bringing a car to a stop, its kinetic energy is dissipated as its braking energy, and the two are roughly equal. The kinetic energy is proportional to the square of the car's speed, while the braking energy is proportional to the stopping distance, assuming a constant braking force.

    The engineering argument boils down to $\textrm{{distance}} \propto \textrm{{speed}}^2$. Therefore we should take the square root of stopping distance.
    ///
    """
        )
    )

    _df = _df.with_columns(h=pl.col("y").sqrt())
    _res = _df.select(linreg(pl.col("x"), pl.col("h"))).item()
    _β0, _β1, _f, _n = _res["β0"], _res["β1"], _res["f"], _res["n"]
    _pval = stats.f.sf(_f, 1, _n - 2)

    _scatter = _df.plot.scatter(
        x=alt.X("x", title="speed").scale(domain=[15, 65]),
        y=alt.Y("h", title="h=√distance").scale(domain=[2, 18]),
    ).properties(title="LS fit")
    _line = _scatter.transform_regression("x", "h").mark_line(color="red")

    _df = _df.with_columns(e=pl.col("h") - (_β0 + _β1 * pl.col("x")), y0=0)
    _error = _df.plot.circle(
        x=alt.X("x", title="speed").scale(domain=[15, 65]),
        y=alt.Y("e", title="error").scale(domain=[-5, 5]),
    ).properties(title="residuals")
    _rule = _df.plot.rule(x="x", y="y0", y2="e")

    _y = (_β0 + _β1 * 40) ** 2

    mo.output.append(
        mo.md(
            rf"""
    /// details | (d) Make this linearizing transformation and check the goodness of fit. What is the predicted stopping distance according to this model if the car is traveling at 40 mph?

    The fitted line is $h = {_β0:.2f} +  {_β1:.2f}x$.

    {mo.as_html((_scatter + _line) | (_error + _rule))}

    The $F$-statistic is {_f:.2f} with $P$-value = {_pval:.2e}, again showing significant linearity. This time, the residual plot has improved considerably confirming that it is a good fit.

    The stopping distance of a car traveling at 40 mph would be $y = h^2 = ({_β0:.2f} +  {_β1:.2f} \cdot 40)^2$ = {_y:.2f} feet.
    ///
    """
        )
    )
    return


@app.cell(hide_code=True)
def _(df_ex, html, md, mo):
    mo.md(
        rf"""
    ### Ex 10.21

    The direct current output from a windmill ($y$) was measured against the wind velocity ($x$) in miles per hour. The following data were obtained.

    {
            mo.center(
                mo.as_html(
                    df_ex(21)
                    .style.tab_options(
                        table_font_size=12,
                        table_width="40%",
                        container_height="50vh",
                    )
                    .cols_align("center")
                    .cols_label(
                        mph=html("Wind Velocity<br>(mph)"),
                        amps=html("DC Output<br>(amp)"),
                    )
                    .tab_source_note(
                        source_note=md(
                            'Source: G. Joglekar, J. H. Schuenemeyer, and V. LaRicca (1989), "Lack of fit testing when replicates are not available," _The American Statistician_, 43, pp. 135-143. Reprinted in _Small Data Sets_, p. 271.'
                        )
                    )
                )
            )
        }
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | (a) Make a scatter plot of the DC output vs. wind velocity. Describe the relationship. Refer to Figure 10.10. Find a transformation that linearizes the relationship. Fit the LS line.

    ///

    ///details | (b) Check the goodness of fit by making residual plots. Do the assumptions of linear regression seem to be satisfied?

    ///

    /// details | (c) What is the predicted output if the wind velocity is 8 mph?

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex, md, mo):
    mo.md(
        rf"""
    ### Ex 10.22

    This data set illustrates the importance of graphical plotting in regression analysis.

    {
            mo.center(
                mo.as_html(
                    df_ex(22)
                    .style.tab_options(
                        table_font_size=12,
                    )
                    .cols_align("center")
                    .tab_stub(rowname_col="No")
                    .tab_stubhead(label="No.")
                    .fmt_integer(columns=["No", "x1", "x2"])
                    .tab_source_note(
                        source_note=md(
                            'Source: F. J. Anscombe (1973). "Graphs in statistical analysis", _The American Statistician_. 27, pp. 17-21.'
                        )
                    )
                )
            )
        }
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | (a) Make four scatter plots: $y_1$ vs. $x_1$, $y_2$ vs. $x_1$, $y_3$ vs. $x_1$, and $y_4$ vs. $x_2$. Comment on their appearances - in particular, the linearity of the relationship exhibited by the plots.

    ///

    /// details | (b) Fit LS straight lines to the four plots and compute the usual statistics that accompany the LS fits. Note that the numerical results are identical.

    ///

    /// details | (c) Why do you think the LS fits are identical even though the scatter plots are quite different? What does this say about the importance of plotting the data before fitting a model?

    ///

    /// details |  (d) What does this say about $r^2$ or the $t$-statistic for testing the significance of $\hat{\beta}_1$ as measures of the linearity of relationships?

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex, md, mo):
    mo.md(
        rf"""
    ### Ex 10.23

    The approximate gestation time and birthweights are given in the following table for selected mammals. The gestation time $t$ (between fertilization and birth) for a mammal is related to the birthweight $w$ by the relationship $t = a\,b^w$, where $a$ and $b$ are constants. Regression methods can be used to estimate $a$ and $b$ by transforming this relationship into a linear model.

    {
            mo.center(
                mo.as_html(
                    df_ex(23)
                    .style.tab_options(
                        table_font_size=12,
                        table_width="60%",
                    )
                    .cols_align("center")
                    .tab_stub(rowname_col="Mammal")
                    .tab_stubhead(label="Mammal")
                    .cols_label(kg="Birthweight (kg)", days="Gestation (days)")
                    .fmt_integer(columns="days")
                    .tab_source_note(
                        source_note=md(
                            "Source: Adapted from W. Keienburg. D. Heinemann. and S. Schmitz eds. (1990). _Grizmek's Encyclopedia of Mammals_, New York: McGraw-Hill."
                        )
                    )
                )
            )
        }
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | (a) Apply the log transformation to obtain a linear model of the form $y = \beta_0 + \beta_1 w$, where $y = \log{t}$. How are $\beta_0$ and $\beta_1$ related to $a$ and $b$, respectively?

    ///

    /// details |  (b) Plot log(gestation time) vs. weight. Is this relationship approximately linear?

    ///

    /// details | (c) Fit the linear model $y = \beta_0 + \beta_1 w$ to the transformed data.

    ///

    /// details | (d) Using the fitted model in (c), estimate the gestation time of a lion which weighs approximately 1.2 kg at birth.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(df_ex, md, mo):
    mo.md(
        rf"""
    ### Ex 10.24

    Hospitalization cost ($h$) that is reimbursed by insurance is approximately related to the length of stay $l$ in the hospital by the relationship $h = a\,l^b$, where $a$ and $b$ are constants. Regression methods can be used to estimate $a$ and $b$ by transforming this relationship into a linear model by making the log transformation. The reimbursed hospital cost and associated length of stay are given for a sample of 33 elderly people.

    {
            mo.center(
                mo.as_html(
                    df_ex(24)
                    .style.tab_options(
                        table_font_size=12,
                        table_width="60%",
                        container_height="50vh",
                    )
                    .cols_align("center")
                    .fmt_integer(columns=["h", "l"])
                    .tab_source_note(
                        source_note=md(
                            "Data courtesy of Professor Susan Hughes, School of Public Health, University of Illinois. Chicago."
                        )
                    )
                )
            )
        }
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | (a) Apply the log transformation to obtain a linear model of the form $y = \beta_0 + \beta_1 x$, where $x = \log l$ and $y = \log h$. How are $\beta_0$ and $\beta_1$ related to $a$ and $b$, respectively?

    ///

    /// details | (b) Plot $y = \log h$ vs. $x = \log l$. Is this relationship approximately linear?

    ///

    /// details |  (c) Fit the linear model $y = \beta_0 + \beta_1 x$ to the transformed data.

    ///

    /// details |  (d) Use the fitted model from (c) to estimate the average reimbursed cost for a 3-day hospital stay by an elderly person.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 10.5 *Correlation Analysis""")
    return


@app.cell(hide_code=True)
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
def _(df_ex, md, mo):
    mo.md(
        rf"""
    ### Ex 10.28

    The following are the heights and weights of 30 eleven year old girls.

    {
            mo.center(
                mo.as_html(
                    df_ex(28)
                    .style.tab_options(
                        table_font_size=12,
                        table_width="50%",
                        container_height="50vh",
                    )
                    .cols_align("center")
                    .fmt_integer(columns=["Height", "Weight"])
                    .tab_source_note(
                        source_note=md(
                            "Source: The Open University (1983). _MDST242 Srarisics in Society, Unit C3: Is my child normal?_, Milton Keynes: The Open University, Figure 3.12. Reprinted in _Small Data Sets_. p. 75."
                        )
                    )
                )
            )
        }
    """
    )
    return


@app.cell(hide_code=True)
def _(ci, mo, pl, stats):
    # _chart = _df.plot.scatter(
    #     alt.X("Height").scale(domain=[130, 165]),
    #     alt.Y("Weight").scale(domain=[20, 60]),
    # )

    mo.output.append(
        mo.md(
            rf"""
    /// details | (a) Plot weights vs. heights.


    ///
    """
        )
    )


    def corr_test(
        x: pl.Expr, y: pl.Expr, alternative: str = "two-sided"
    ) -> pl.Expr:
        n = x.len()
        r = pl.corr(x, y)
        t = r * (n - 2).sqrt() / (1 - r**2).sqrt()
        s = pl.struct(n.alias("n"), t.alias("t"))

        match alternative:
            case "two-sided":
                res = s.map_batches(
                    lambda s: 2
                    * stats.t.sf(
                        s.struct.field("t").abs(), s.struct.field("n") - 2
                    ).item()
                )
            case "less":
                res = s.map_batches(
                    lambda s: stats.t.cdf(
                        s.struct.field("t"), s.struct.field("n") - 2
                    ).item()
                )
            case "greater":
                res = s.map_batches(
                    lambda s: stats.t.sf(
                        s.struct.field("t"), s.struct.field("n") - 2
                    ).item()
                )
        return res.alias("pval")


    def corr_ci(
        x: pl.Expr, y: pl.Expr, alternative: str = "two-sided", α: float = 0.05
    ) -> pl.Expr:
        r = pl.corr(x, y)
        n = x.len()
        match ci:
            case "two-sided":
                pass
            case "greater":
                pass
            case "less":
                pass


    # _r = _df.select(pl.corr("Height", "Weight")).item()

    mo.output.append(
        mo.md(
            rf"""
    /// details | (b) Calculate the correlation coefficient. Test if it is significantly greater than 0.7.


    ///

    """
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Advanced Exercises""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Ex 10.33

    Show that

    $$
    \sum_{i=1}^n (y_i - \hat{y}_i)(\hat{y}_i - \bar{y}) = 0.
    $$

    (_Hint_: Substitute $\hat{y}_i = \beta_0 + \beta_1 x_i = \bar{y} + \hat{\beta}_1 (x_i - \bar{x})$ and simplify.)
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
