import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Chapter 10 Simple Linear Regression and Correlation""")
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import polars as pl
    from polars import col
    import numpy as np
    import altair as alt
    from scipy import stats
    from great_tables import GT, md, html
    from dataclasses import dataclass

    import inspect


    def get_source(func) -> str:
        """Display a function's source code as markdown"""
        source = inspect.getsource(func)
        return f"""```python
    {source}
    ```"""
    return GT, alt, col, html, md, mo, np, pl, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 10.1 A Probabilistic Model for Simple Linear Regression""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    _Linear regression analysis_ begins by fitting a straight line, $y = \beta_0 + \beta_1 x$, to a set of paired data $\{(x_i, y_i), i = 1, 2, \ldots , n\}$ on two numerical variables $x$ and $y$. The linear regression model
    $$Y_i = \beta_0 + \beta_1 x_i + \epsilon_i\ (i=1,2, \ldots, n)$$
    has these basic assumptions:

    1. The predictor variable $x$ is regarded as nonrandom because it is assumed to be set by the investigator.
    2. The mean of $Y_i$ is a linear function of $x_i$.
    3. The errors $\epsilon_i$ are i.i.d. normal.
    """
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
def _(mo):
    mo.md(
        r"""
    The _least squares(LS) estimates_ $\hat{\beta}_0$ and $\hat{\beta}_1$ minimize $Q = \sum_{i=1}^n [y_i - (\beta_0 + \beta_1 x_i) ]^2$ and are given by
    $$\begin{align*}
    \hat{\beta}_1 &= \frac{S_{xy}}{S_{xx}}, \\
    \hat{\beta}_0 &= \bar{y} - \hat{\beta}_1 \bar{x}
    \end{align*}$$
    where $S_{xy} = \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})$ and $S_{xx} = \sum_{i=1}^n (x_i - \bar{x})^2$. The *fitted values* are given by $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$ and the *residuals* by $e_i = y_i - \hat{y}_i$.

    The total sum of squares (SST), regression sum of squares (SSR) and error sum of squares (SSE) are defined as $\mathrm{SST} = \sum_{i=1}^n (y_i - \bar{y})^2$, $\mathrm{SSR} = \sum_{i=1}^n (\hat{y}_i - \bar{y})^2$, and $\mathrm{SSE} = \sum_{i=1}^n (y_i - \hat{y}_i)^2$. These sums of squares satisfy the identity $\mathrm{SST} = \mathrm{SSR} + \mathrm{SSE}$. A measure of goodness of fit of the least squares line is the *coefficient of determination*,
    $$r^2 = \frac{\mathrm{SSR}}{\mathrm{SST}} = 1 - \frac{\mathrm{SSE}}{\mathrm{SST}}$$
    which represents the proportion of variation in $y$ that is accounted for by regression on $x$. The *correlation coefficient* $r$ equals $\pm\sqrt{r^2}$, where $\mathrm{sign}(r) = \mathrm{sign}(\hat{\beta}_1)$. In fact, $r = \hat{\beta}_1 (s_x / s_y)$, where $s_x$ and $s_y$ are the sample standard deviations of $x$ and $y$, respectively.

    The *probabilistic model* for linear regression assumes that $y_i$ is the observed value of r.v. $Y \thicksim N(\mu_i, \sigma^2)$, where $\mu_i = \beta_0 + \beta_1 x_i$ and the $Y_i$ are independent. An unbiased estimate of $\sigma^2$ is provided by $s^2 = \mathrm{SSE}/(n - 2)$ with $n-2$ d.f.
    """
    )
    return


@app.cell(hide_code=True)
def _(GT, alt, col, mo, pl):
    class Regression:
        @staticmethod
        def linreg(x: pl.Expr, y: pl.Expr) -> pl.Expr:
            """
            Gives various linear regression statistics by directly
            translating textbook formulas to polars expressions.

            Input:
                - x, y: observations

            Output: a pl.struct of
                - n: number of data points
                - xmean: mean of x
                - sxx: sum of squares of x
                - β0/1: intercept/slope of regression line
                - rto: slope of regression through the origin
                - r2: coefficient of determination
                - s2: mean square error estimate of σ^2
                - f: F statistic of H0: β1 = 0. f=t^2
                - se_β0/1: standard error of β0/1
            """
            n = x.len()
            xmean = x.mean()
            ymean = y.mean()
            sxx = ((x - xmean) ** 2).sum()
            syy = ((y - ymean) ** 2).sum()
            sxy = ((x - xmean) * (y - ymean)).sum()
            β1 = sxy / sxx
            β0 = ymean - xmean * β1
            rto = (x * y).sum() / (x**2).sum()
            r2 = sxy**2 / (sxx * syy)
            s2 = (syy - β1**2 * sxx) / (n - 2)
            f = β1**2 * sxx / s2
            se_β0 = (s2 * (x**2).sum() / (n * sxx)).sqrt()
            se_β1 = (s2 / sxx).sqrt()
            return pl.struct(
                n.alias("n"),
                xmean.alias("xmean"),
                sxx.alias("sxx"),
                β0.alias("β0"),
                β1.alias("β1"),
                rto.alias("rto"),
                r2.alias("r2"),
                s2.alias("s2"),
                f.alias("f"),
                se_β0.alias("se_β0"),
                se_β1.alias("se_β1"),
            )

        def _calc(self) -> None:
            stats = self.df.select(
                self.linreg(col(self._x_name), col(self._y_name))
            ).item()
            vars(self).update(stats)
            self.df = self.df.with_columns(
                residual=col(self._y_name)
                - (self.β0 + self.β1 * col(self._x_name)),
                y0=0,
            )

        @staticmethod
        def _parseLabel(s: str):
            return s.split(":") if ":" in s else [s] * 2

        @staticmethod
        def ex(dnum: int) -> pl.DataFrame:
            datafile = f"../SDAEI-Tamhane/ch10/Ex10-{dnum}.json"
            return pl.read_json(datafile).explode(pl.all())

        @staticmethod
        def gt(dnum: int, dark: bool = True) -> GT:
            """return a GT preset in dark mode"""
            gt = Regression.ex(dnum).style.tab_options(
                table_font_size=12,
                # table_width="60%",
                container_height="60vh",
            )
            if dark:
                gt = gt.tab_options(
                    table_font_color="white",
                    table_background_color="#181C1A",
                )

            return gt

        def __init__(
            self, dnum: int, x_label: str = "x", y_label: str = "y"
        ) -> None:
            self.df = self.ex(dnum)

            self._x_name, self._x_title = self._parseLabel(x_label)
            self._y_name, self._y_title = self._parseLabel(y_label)

            self._calc()

        def predict(
            self, *, x: float | None = None, y: float | None = None
        ) -> float:
            if x and not y:
                return self.β0 + self.β1 * x
            elif y and not x:  # inverse regression
                return (y - self.β0) / self.β1
            else:
                raise ValueError("Either x or y should be used.")

        def _chart_scatter(self, x: str, y: str) -> alt.Chart:
            return self.df.plot.scatter(
                x=alt.X(x).title(self._x_title).scale(zero=False, padding=10),
                y=alt.Y(self._y_name)
                .title(self._y_title)
                .scale(zero=False, padding=10),
            )

        def _chart_regression(self) -> alt.Chart:
            scatter = self._chart_scatter()
            line = scatter.transform_regression(
                self._x_name, self._y_name
            ).mark_line(color="red")
            title = f"LS fit: {self._y_title} = {self.β0:.3g} {'-' if self.β1 < 0 else '+'} {abs(self.β1):.3g}×{self._x_title}"
            return (scatter + line).properties(title=title)

        def chart(self, kind: str = "scatter") -> alt.Chart:
            match kind:
                case "scatter":
                    chart = self._chart_scatter()
                case "regression":
                    chart = self._chart_regression()
                case _:
                    raise ValueError("Unkown chart kind.")

            return chart


    mo.show_code()
    return (Regression,)


@app.cell(hide_code=True)
def _(Regression, html, md, mo):
    mo.md(
        rf"""
    ### Ex 10.4

    The time between eruptions of Old Faithful geyser in Yellowstone National Park is random but is related to the duration of the last eruption. The table below shows these times for 21 consecutive eruptions.

    {
            mo.center(
                mo.as_html(
                    Regression.gt(4)
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
def _(Regression, mo, np):
    _r = Regression(dnum=4, x_label="LAST", y_label="NEXT")

    mo.md(
        rf"""
    /// details | (a) Make a scatter plot of NEXT vs. LAST. Does the relationship appear to be approximately linear?

    {mo.as_html(_r.chart("scatter"))}

    Yes, the points appear approximately linear.
    ///

    /// details | (b) Fit a least squares regression line. Use it to predict the time to the next eruption if the last eruption lasted 3 minutes.

    {mo.as_html(_r.chart("regression"))}

    The fit result is that $\beta_0$ = {_r.β0:.3g} and $\beta_1$ = {_r.β1:.3g}. If the last eruption lasted 3 minutes, the time to the next eruption would be in about {_r.β0:.3g} + {_r.β1:.3g} × 3 = {_r.predict(x=3):.3g} minutes.
    ///

    /// details | (c) What proportion of variability in NEXT is accounted for by LAST? Does it suggest that LAST is a good predictor of NEXT?

    $r^2$ = {_r.r2:.3g}, suggesting that `LAST` is a pretty good predictor of `NEXT`.
    ///

    /// details | (d) Calculate the mean square error estimate of $\sigma$.

    $s^2$ = {_r.s2:.3g}, and the mean square error estimate of $\sigma$ is $s$ = {np.sqrt(_r.s2):.3g}.
    ///

                """
    )
    return


@app.cell(hide_code=True)
def _(Regression, md, mo):
    mo.md(
        f"""
    ### Ex 10.5

    The data below show Olympic triple jump winning distances for men in meters for the years 1896 to 1992 (there were no Olympic games in 1916, 1940, and 1944).

    {
            mo.center(
                mo.as_html(
                    Regression.gt(5)
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
def _(Regression, mo, np):
    _r = Regression(dnum=5, x_label="Year", y_label="Distance")

    # _scatter = _df.with_columns(pl.col("Year").cast(int).cast(str)).plot.scatter(
    #     alt.X("Year:T"),
    #     alt.Y("Distance").scale(domain=[13, 19]),
    # )

    mo.md(
        rf"""
    /// details | (a) Make a scatter plot of the length of the jump by year. Does the relationship appear to be approximately linear?

    {mo.as_html(_r.chart("scatter"))}

    Yes, the points appear approximately linear.
    ///

    /// details |  (b) Fit a least squares regression line.

    {mo.as_html(_r.chart("regression"))}
    ///

    /// details | (c) Calculate the mean square error estimate of $\sigma$.

    $s^2$ = {_r.s2:.3g}, and the mean square error estimate of $\sigma$ is $s$ = {np.sqrt(_r.s2):.3g}.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(Regression, md, mo):
    mo.md(
        f"""
    ### Ex 10.6

    The following data give the barometric pressure (in inches of mercury) and the boiling point (in °F) of water in the Alps.

    {
            mo.center(
                mo.as_html(
                    Regression.gt(6)
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
def _(Regression, mo, np):
    _r = Regression(dnum=6, x_label="Pressure", y_label="Temp")

    mo.md(
        rf"""
    /// details | (a) Make a scatter plot of the boiling point by barometric pressure. Does the relationship appear to be approximately linear?

    {mo.as_html(_r.chart("scatter"))}

    Yes, the relationship is approximately linear.
    ///

    /// details | (b) Fit a least squares regression line. What proportion of variation in the boiling point is accounted for by linear regression on the barometric pressure?

    {mo.as_html(_r.chart("regression"))}

    $r^2$ = {_r.r2:.3g}. That is, {_r.r2 * 100:.1f}% percent of variation in the boiling point is accounted for by linear regression on the barometric pressure.
    ///

    /// details | (c) Calculate the mean square error estimate of $\sigma$.

    $s^2$ = {_r.s2:.3g}, and the mean square error estimate of $\sigma$ is $s$ = {np.sqrt(_r.s2):.3g}.
    ///"""
    )
    return


@app.cell(hide_code=True)
def _(Regression, md, mo):
    mo.md(
        rf"""
    ### Ex 10.7

    The following table shows Olympic 100 meter backstroke winning times for women for the years 1924 to 1992 (there were no Olympic games in 1940 and 1944).

    {
            mo.center(
                mo.as_html(
                    Regression.gt(7)
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
def _(Regression, mo, np):
    _r = Regression(dnum=7, x_label="Year", y_label="Time")

    mo.md(
        rf"""
    /// details | (a) Make a scatter plot of the winning times by year. Does the relationship appear to be approximately linear?

    {mo.as_html(_r.chart("scatter"))}

    Yes, the relationship is approximately linear.
    ///

    /// details | (b) Fit a least squares regression line.

    {mo.as_html(_r.chart("regression"))}
    ///

    /// details | (c) Calculate the mean square error estimate of $\sigma$.

    $s^2$ = {_r.s2:.3g}, and the mean square error estimate of $\sigma$ is $s$ = {np.sqrt(_r.s2):.3g}.
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
    $$\beta_1 = \frac{\sum x_i y_i}{\sum x_i^2}.$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To find the minimum of the sum of errors
    $$Q = \sum (\beta_1 x_i - y_i)^2,$$
    we take its derivative with respect to $\beta_1$:
    $$\begin{align*} 
    \frac{\partial Q}{\partial \beta_1} &= \sum 2x_i(\beta_1 x_i - y_i)\\
    &= 2\sum \beta_1 x_i^2 - x_i y_i.
    \end{align*}$$
    Therefore the minimum is taken at
    $$\beta_1 = \frac{\sum x_i^2}{\sum x_i y_i}.$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 10.3 Statistical Inference for Simple Linear Regression""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The estimated standard errors of $\hat{\beta}_0$ and $\hat{\beta}_1$ equal
    $$\mathrm{SE}(\hat{\beta}_0) = s\sqrt{\frac{\sum x_i^2}{n\,S_{xx}}} \quad \text{and}\quad
    \mathrm{SE}(\hat{\beta}_1) = \frac{s}{\sqrt{S_{xx}}}.$$
    These are used to construct confidence intervals and perform hypothesis tests on $\beta_0$ and $\beta_1$. For example, a $100(1-\alpha)\%$ confidence interval on $\beta_1$ is given by
    $$\hat{\beta_1} \pm t_{n-2, \alpha/2}\,\textrm{SE}(\hat{\beta_1}).$$

    A common use of the fitted regression model is to _predict_ $Y^*$ for specified $x = x^*$ or to _estimate_ $\mu^* = \textrm{E}(Y^*)$. In both cases we have
    $$\hat{Y}^* = \hat{\mu}^* = \beta_0 + \beta_1 x^*.$$

    A $100(1-\alpha)\%$ CI for $\mu^*$ is given by
    $$\hat{\mu}^* \pm t_{n-2, \alpha/2} s \sqrt{\frac{1}{n} + \frac{(x^* - \bar{x})^2}{S_{xx}}}.$$

    A $100(1-\alpha)\%$ _prediction interval_ (PI) for $Y^*$ is given by
    $$\hat{Y}^* \pm t_{n-2, \alpha/2} s \sqrt{1 + \frac{1}{n} + \frac{(x^* - \bar{x})^2}{S_{xx}}}.$$

    However, a $100(1-\alpha)\%$ PI for $Y^*$ is wider than a CI for $\mu^*$, because $Y^*$ is an r.v., while $\mu^*$ is a fixed constant.
    """
    )
    return


@app.cell(hide_code=True)
def _(Regression, mo, np, stats):
    from typing import NamedTuple


    class RegressionInference(Regression):
        class slopeTestResult(NamedTuple):
            pval: float
            t: float

        def slopeTest(
            self, k: float = 0, alternative: str = "two-sided"
        ) -> slopeTestResult:
            """
            returns (P-value, t-statistic) of observed data under H0: β1 = k

            input:
                - k: given constant to compare with β1. defaults to 0
                - alternative:
                    'two-sided': β1 ≠ k which is the default
                    'less': β1 < k
                    'greater': β1 > k

            output:
                a slopeTestResult object containing the P-value and the t-statistic.
            """
            t = (self.β1 - k) / self.se_β1
            match alternative:
                case "two-sided":
                    pval = 2 * stats.t.sf(abs(t), self.n - 2)
                case "greater":
                    pval = stats.t.sf(t, self.n - 2)
                case "less":
                    pval = stats.t.cdf(t, self.n - 2)
                case _:
                    raise ValueError("unknown alternative value.")
            return self.slopeTestResult(pval=pval, t=t)

        def estimateInterval(
            self,
            x: float,
            kind: str = "CI",
            α: float = 0.05,
        ) -> list[float]:
            """
            Gives the confidence/prediction interval.

            Input:
                - x: input point for estimation
                - kind: "CI" or "PI"
                - α: significance level, defaults to 0.05

            Output:
                the confidence/prediction interval [low, high].
            """
            y = self.predict(x=x)
            t_crit = stats.t.ppf(1 - α / 2, self.n - 2)  # critical value
            if kind == "CI":
                se = np.sqrt(
                    self.s2 * (1 / self.n + (x - self.xmean) ** 2 / self.sxx)
                )
            elif kind == "PI":
                se = np.sqrt(
                    self.s2 * (1 + 1 / self.n + (x - self.xmean) ** 2 / self.sxx)
                )
            else:
                raise ValueError("Unknown interval kind")

            return [y - t_crit * se, y + t_crit * se]


    mo.show_code()
    return (RegressionInference,)


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
def _(RegressionInference, mo):
    _r = RegressionInference(dnum=5, x_label="Year", y_label="Distance")
    _pval, _t = _r.slopeTest(alternative="greater")
    [_l, _h] = _r.estimateInterval(2004, kind="PI")

    mo.md(
        rf"""
    /// details | (a) Is there a significant increasing linear trend in the triple jump distance? Test at $\alpha = .05$.

    This is a test of $H_0: \beta_1 \le 0$ and the $t$-statistic is {_t:.3g} with a one-sided $P$-value of {_pval:.3g} < $\alpha$. So yes there is a significant increasing trend.
    ///

    /// details | (b) Calculate a 95% PI for the winning jump in 2004. Do you think this prediction is reliable? Why or why not? Would a 95% CI for the winning jump in 2004 have a meaningful interpretation? Explain.

    A 95% PI for the winning jump in 2004 is [{_l:.3g}, {_h:.3g}], but it is not reliable since we are extrapolating. In this case, a CI is not meaningful because there will be at most a single winning jump in 2004. 
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
def _(RegressionInference, mo):
    _r = RegressionInference(dnum=6, x_label="Pressure", y_label="Temp")
    [_l, _h] = _r.estimateInterval(28, kind="CI")

    mo.output.append(
        mo.md(
            rf"""
    /// details | (a) Calculate a 95% CI for the boiling point if the barometric pressure is 28 inches of mercury. Interpret your CI.

    The said CI is calculated to be [{_l:.2f}, {_h:.2f}]. That is to say, there is a 95% chance that this interval includes the boiling point at 28 inches of of mercury on the true regression line.
    ///"""
        )
    )

    [_l, _h] = _r.estimateInterval(31, kind="CI")

    mo.output.append(
        mo.md(
            rf"""
    /// details | (b) Calculate a 95% CI for the boiling point if the barometric pressure is 31 inches of mercury. Compare this with the CI of (a).

    The said CI is calculated to be [{_l:.2f}, {_h:.2f}]. It is much wider than (a) at 28 inches of mercury and should be treated as unreliable because we are extrapolating outside the data domain.
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
def _(RegressionInference, mo):
    _r = RegressionInference(dnum=4, x_label="LAST", y_label="NEXT")
    [_l, _h] = _r.estimateInterval(3, kind="PI")

    mo.output.append(
        mo.md(
            rf"""
    /// details | (a) Calculate a 95% PI for the time to the next eruption if the last eruption lasted 3 minutes.

    The said PI is calculated to be [{_l:.2f}, {_h:.2f}].
    ///"""
        )
    )

    [_l, _h] = _r.estimateInterval(3, kind="CI")

    mo.output.append(
        mo.md(
            rf"""
    /// details | (b) Calculate a 95% CI for the mean time to the next eruption for a last eruption lasting 3 minutes. Compare this CI with the PI obtained in (a).

    The said CI is calculated to be [{_l:.2f}, {_h:.2f}] which is a lot narrower than the PI in (a).
    ///"""
        )
    )

    [_l, _h] = _r.estimateInterval(1, kind="PI")

    mo.output.append(
        mo.md(
            rf"""
    /// details | (c) Repeat (a) if the last eruption lasted 1 minute. Do you think this prediction is reliable? Why or why not?

    The PI for a 1 minute last eruption is [{_l:.2f}, {_h:.2f}] which is unreliable because we are extrapolating outside of the data domain.
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
def _(RegressionInference, mo):
    _r = RegressionInference(dnum=7, x_label="Year", y_label="Time")
    [_l, _h] = _r.estimateInterval(2004, kind="PI")
    _x = _r.predict(y=60)

    mo.md(
        rf"""
    /// details | (a) Calculate a 95% PI for the winning time in 2004. Do you think this prediction is reliable? Why or why not?

    The specified PI is calculated to be [{_l:.1f}, {_h:.1f}]. However, this prediction is unreliable because we are extrapolating outside the data range (latest available year was 1996).
    ///

    /// details | (b) Use the regression equation to find the year in which the winning time would break 1 minute. Given that the Olympics are every four years, during which Olympics would this happen?

    The year {_x:.0f} (by inverse regression).
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(Regression, md, mo):
    mo.md(
        rf"""
    ### Ex 10.13

    The U.S. infant mortality rates (IMR) (per 1000 live births) for both sexes and all races for the years 1981-1990 (coded as years 1-10) were as follows:

    {
            mo.center(
                mo.as_html(
                    Regression.gt(13)
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
    mo.md(r"""Residuals are key to checking the model assumptions such as normality of the $Y_i$, linearity of the regression model, constant variance $\sigma^2$, and independence of the $Y_i$. Residuals are also useful for detecting _outliers_ and _influential observations_. Many of these diagnostic checks are done by plotting residuals in appropriate ways.""")
    return


@app.cell
def _(Regression, alt, mo, pl):
    class RegressionDiagnosis(Regression):
        def transform(
            self,
            x_transform: pl.Expr | None = None,
            x_label: str = "x_transformed",
            y_transform: pl.Expr | None = None,
            y_label: str = "y_transformed",
        ) -> None:
            if x_transform is not None:
                self._x_name, self._x_title = self._parseLabel(x_label)
                self.df = self.df.with_columns(x_transform.alias(self._x_name))
            if y_transform is not None:
                self._y_name, self._y_title = self._parseLabel(y_label)
                self.df = self.df.with_columns(y_transform.alias(self._y_name))
            if not (x_transform is None and y_transform is None):
                self._calc()

        def _chart_residual(self) -> alt.Chart:
            scatter = self.df.plot.circle(
                x=alt.X(self._x_name)
                .title(self._x_title)
                .scale(zero=False, padding=10)
                .axis(grid=False),
                y=alt.Y("residual").scale(zero=False, padding=10),
            ).properties(title="residual plot")
            fill = self.df.plot.rule(
                x=self._x_name,
                y=alt.Y("y0").title(None),
                y2="residual",
                size=alt.value(0.2),
            )
            axis = self.df.plot.rule(y="y0")

            return scatter + fill + axis

        def _chart_normal(self) -> alt.Chart:
            pass

        def chart(self, kind: str = "residual") -> alt.Chart:
            match kind:
                case "residual":
                    chart = self._chart_residual()
                case "normal":
                    chart = self._chart_normal()
                case _:
                    chart = super().chart(kind=kind)

            return chart


    mo.show_code()
    return (RegressionDiagnosis,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Ex 10.15

    Often, the probability of response $p\ (0 \le p \le 1)$ is modeled as a function of a stimulus $x$ by the _logistic function_:

    $$
    p = \frac{\exp{(\beta_0 + \beta_1 x)}}{1+\exp{(\beta_o + \beta_1 x)}}.
    $$

    For example, the stimulus is the dose level of a drug and the response is cured or is not cured. Find the linearizing transformation $h(p)$ so that $h(p) = \beta_0 + \beta_1 x$.
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
def _(Regression, md, mo):
    mo.md(
        rf"""
    ### Ex 10.16

    A prime is a positive integer that has no integer factors other than 1 and itself (1 is not regarded as a prime). The number of primes in any given interval of whole numbers is highly irregular. However, the proportion of primes less than or equal to any given number $x$ (denoted by $p(x)$) follows a regular pattern as $x$ increases. The following table gives the number and proportion of primes for $x = 10^n$ for $n = 1, 2, \ldots, 10$. The objective of the present exercise is to discover this pattern.

    {
            mo.center(
                mo.as_html(
                    Regression.gt(16)
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
def _(Regression, alt, mo, pl):
    _df = (
        Regression.ex(16)
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
    $$\hat{{\beta}}_1 = {_β1:.4g} \approx \log_{{10}}e = 0.4343.$$
    ///
    """
        r"""
    /// details | (c) Explain how the relationship found in (b) roughly translates into the _prime number theorem_: For large $x$, $p(x) \approx 1 / \log_e x$.

    For large $x$,
    $$\begin{align*}
    p(x) &= \beta_1 \frac{1}{\log_{10}x} \\
    &\approx \frac{\log_{10}e}{{\log_{10}x}} \\
    &=  \frac{1}{\log_e x}.
    \end{align*}$$
    ///"""
    )
    return


@app.cell(hide_code=True)
def _(Regression, md, mo):
    mo.md(
        rf"""
    ### Ex 10.17

    In a memory retention experiment subjects were asked to memorize a list of disconnected items, and then were asked to recall them at various times up to a week later. The proportion $p$ of items recalled at times $t$ (in minutes) is given below.

    {
            mo.center(
                mo.as_html(
                    Regression.gt(17)
                    .tab_options(
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
def _(RegressionDiagnosis, col, mo, np):
    _r = RegressionDiagnosis(dnum=17, x_label="t", y_label="P:p")
    _r.transform(x_transform=col("t").log(), x_label="h:ln(t)")

    mo.md(
        r"""
    /// details | (a) Note that $t$ increases almost geometrically throughout. This suggests that a logarithmic transformation of $t$ might linearize the relationship. Plot $p$ vs. $\ln{t}$. Is the relationship approximately linear?

    """
        rf"""{mo.as_html(_r.chart("scatter"))}

    Yes. The relationship appears approximately linear.
    ///

    /// details | (b) Fit a trend line to the plot in (a). From the trend line estimate the time for 50% retention.

    The trend line is $\hat p = {_r.β0:.3g} - {abs(_r.β1):.3g}h$.

    {mo.as_html(_r.chart("regression"))}

    """
        r"""
    Because $p = \beta_0 + \beta_1\;\ln{t}$, $t=\exp{[(p-\beta_0)/\beta_1]}$.
    """
        rf""" For p = 50% retention, $t = {np.exp((0.5 - _r.β0) / _r.β1):.4g}$ minutes.
    """
    )
    return


@app.cell(hide_code=True)
def _(Regression, html, md, mo):
    mo.md(
        rf"""
    ### Ex 10.18

    The following are the average distances of the planets in the solar system from the sun:

    {
            mo.center(
                mo.as_html(
                    Regression.gt(18)
                    .cols_align("center")
                    .tab_stub(rowname_col="No")
                    .tab_stubhead(label="Planet No.")
                    .cols_label(Dist=html("Distance<br>(millions of miles)"))
                    .fmt_integer(columns="No")
                    .tab_source_note(
                        source_note=md(
                            "This exercise is based on Example 6, Ch. 3 of F. Mosteller. S. E. Fienherg, and R. E. K. Rourke (1983). _Beginning Statistics with Data Analysis_, Reading. MA: Addison-Wesley."
                        )
                    )
                )
            )
        }
    """
    )
    return


@app.cell(hide_code=True)
def _(RegressionDiagnosis, col, mo, np):
    _r = RegressionDiagnosis(dnum=18, x_label="No:Planet No.", y_label="Dist")
    _r.transform(y_transform=col("Dist").log(), y_label="h:ln(Distance)")

    mo.md(
        rf"""
    /// details | (a) How does the distance of a planet from the sun increase with the planet number? Find a transformation of the distance that gives a linear relationship with respect to the planet number.

    The distances seem to increase exponentially with the planet number at a factor of 1.5 - 2. Therefore we take the logarithm of the distances.

    {mo.as_html(_r.chart("scatter"))}

    Yeah, the transformation appears to give a linear relationship.
    ///

    /// details | (b) Fit a least squares straight line after linearizing the relationship.

    The least squares line is $h = {_r.β0:.3g} + {_r.β1:.3g}\;x$.

    {mo.as_html(_r.chart("regression"))}
    ///

    /// details | (c) It is speculated that there is a planet beyond Pluto, called Planet X. Predict its distance from the sun.

    $h^*={_r.β0:.3g} + {_r.β1:.3g} \cdot 11 = {_r.predict(x=11):.3f}$. So distance = $\exp{{(h^*)}}$ = {np.exp(_r.predict(x=11)):.0f} millions of miles.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(Regression, html, mo):
    mo.md(
        rf"""
    ### Ex 10.19

    The following are the speeds of the planets in the solar system as they revolve around the sun:

    {
            mo.center(
                mo.as_html(
                    Regression.gt(19)
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
def _(RegressionDiagnosis, col, mo):
    _r = RegressionDiagnosis(dnum=19, x_label="Distan:distance", y_label="Speed:speed")
    _r.transform(y_transform=1/col("Speed")**2, y_label="h:1/speed²")

    mo.md(
        r""" 
    Highschool physics tells us that a planet orbiting the sun at distance $R$ and speed $v$ will have centripetal acceleration $g = v^2/R$, which is provided by gravity $g = GM/R^2$. Therefore 
    $$R = \frac{GM}{v^2}.$$

    So the transformation we are looking for is $h = 1/\textrm{speed}^2$, which is confirmed by the scatter plot that this is a linear relationship.
    """
        rf"""
    {mo.as_html(_r.chart("scatter"))}

    To fit this linear relationship, we should force $\beta_0 = 0$ and use _regression through the origin_. $\hat{{\beta}}_1$ = {_r.rto:.3g}.
    """
    )
    return


@app.cell(hide_code=True)
def _(Regression, html, mo):
    mo.md(
        rf"""
    ### Ex 10.20

    To relate the stopping distance of a car to its speed, ten cars were tested at five different speeds, two cars at each speed. The following data were obtained.

    {
            mo.center(
                mo.as_html(
                    Regression.gt(20)
                    .cols_align("center")
                    .cols_label(
                        x=html("Speed x<br>(mph)"),
                        y=html("Stop. Dist. y<br>(ft)"),
                    )
                    .fmt_integer(columns="x")
                )
            )
        }
    """
    )
    return


@app.cell(hide_code=True)
def _(RegressionDiagnosis, col, mo, stats):
    _r = RegressionDiagnosis(dnum=20, x_label="x:speed", y_label="y:stop distance")
    _pval = stats.f.sf(_r.f, 1, _r.n - 2)

    mo.output.append(
        mo.md(
            rf"""
    /// details | (a) Fit an LS straight line to these data. Plot the residuals against the speed.

    {mo.as_html(_r.chart("regression") | _r.chart("residual"))}
    ///

    /// details | (b) Comment on the goodness of the fit based on the overall $F$-statistic and the residual plot. Which two assumptions of the linear regression model seem to be violated?

    The $F$-statistic is {_r.f:.3g} with $P$-value = {_pval:.3g}, showing that the trend clearly has a significant linear component. However, the residual plot reveals that two assumptions seem to be violated: (1) linearity - a systematic, parabolic pattern indicates the regression does not fit the data adequately; and (2) constant variance - the error variance seem to get bigger with $x$.
    ///

    /// details | (c) Based on the residual plot, what transformation of stopping distance should be used to linearize the relationship with respect to speed? A clue to find this transformation is provided by the following engineering argument: In bringing a car to a stop, its kinetic energy is dissipated as its braking energy, and the two are roughly equal. The kinetic energy is proportional to the square of the car's speed, while the braking energy is proportional to the stopping distance, assuming a constant braking force.

    The engineering argument boils down to $\textrm{{distance}} \propto \textrm{{speed}}^2$. Therefore we should take the square root of stopping distance.
    ///
    """
        )
    )

    _r.transform(y_transform=col("y").sqrt(), y_label="h:√distance")
    _pval = stats.f.sf(_r.f, 1, _r.n - 2)
    _y = _r.predict(x=40) ** 2

    mo.output.append(
        mo.md(
            rf"""
    /// details | (d) Make this linearizing transformation and check the goodness of fit. What is the predicted stopping distance according to this model if the car is traveling at 40 mph?

    {mo.as_html(_r.chart("regression") | _r.chart("residual"))}

    The $F$-statistic is {_r.f:.3g} with $P$-value = {_pval:.3g}, again showing significant linearity. This time, the residual plot has improved considerably confirming that it is a good fit.

    The stopping distance of a car traveling at 40 mph would be $y = h^2 = ({_r.β0:.3g} +  {_r.β1:.3g} \cdot 40)^2$ = {_y:.3g} feet.
    ///
    """
        )
    )
    return


@app.cell(hide_code=True)
def _(Regression, html, md, mo):
    mo.md(
        rf"""
    ### Ex 10.21

    The direct current output from a windmill ($y$) was measured against the wind velocity ($x$) in miles per hour. The following data were obtained.

    {
            mo.center(
                mo.as_html(
                    Regression.gt(21)
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
def _(RegressionDiagnosis, mo):
    _r = RegressionDiagnosis(dnum=21, x_label="mph:wind velocity", y_label="amps:DC output")

    mo.md(
        rf"""
    /// details | (a) Make a scatter plot of the DC output vs. wind velocity. Describe the relationship. Find a transformation that linearizes the relationship. Fit the LS line.

    {mo.as_html(_r.chart("scatter"))}

    The scatter plot shows that the DC output increases with the wind velocity, but the increase gradually peters out. Let's take the transformation $y \to y^3$ and fit the LS line.

    {mo.as_html(_r.chart_ls)}

    ///

    ///details | (b) Check the goodness of fit by making residual plots. Do the assumptions of linear regression seem to be satisfied?

    {mo.as_html(_r.chart_err)}

    The $F$-statistic = {_r.f:.3f} indicating there's a significant linear relationship, and the residual plot appears to be normal. So yes, the assumptions of linear regression are satisfied.
    ///

    /// details | (c) What is the predicted output if the wind velocity is 8 mph?

    According to this model, the DC output at wind velocity = 8 mph would be {_y ** (1 / 3):.3f} amps. 
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(Regression, md, mo):
    mo.md(
        rf"""
    ### Ex 10.22

    This data set illustrates the importance of graphical plotting in regression analysis.

    {
            mo.center(
                mo.as_html(
                    Regression.gt(22)
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
def _(Regression, alt, linreg, mo, pl):
    _df = (
        Regression.ex(22)
        .select(
            "No",
            x1y1=pl.struct(x="x1", y="y1"),
            x1y2=pl.struct(x="x1", y="y2"),
            x1y3=pl.struct(x="x1", y="y3"),
            x2y4=pl.struct(x="x2", y="y4"),
        )
        .unpivot(index="No", variable_name="dataset")
        .select("dataset", pl.col("value").struct.unnest())
    )

    _scatter = _df.plot.circle(
        x=alt.X("x").scale(zero=False, padding=10),
        y=alt.Y("y").scale(zero=False, padding=10),
    ).facet(facet="dataset", columns=2)

    _linreg_by_set = _df.group_by("dataset").agg(
        linreg(pl.col("x"), pl.col("y")).struct.unnest()
    )

    mo.md(
        rf"""
    /// details | (a) Make four scatter plots: $y_1$ vs. $x_1$, $y_2$ vs. $x_1$, $y_3$ vs. $x_1$, and $y_4$ vs. $x_2$. Comment on their appearances - in particular, the linearity of the relationship exhibited by the plots.

    {mo.as_html(_scatter)}

    - These scatter plots show:
        - x1y1: some linearity
        - x1y2: clearly quadratic relationship
        - x1y3: otherwise strong linearity except for one outlier
        - x2y4: linearity questionable, as multiple data points have the same $x$.
    ///

    /// details | (b) Fit LS straight lines to the four plots and compute the usual statistics that accompany the LS fits. Note that the numerical results are identical.

    {
            mo.center(
                mo.as_html(
                    _linreg_by_set.style.tab_options(
                        table_font_size=12,
                    )
                    .tab_header(
                        title="(almost) identical LS fit statistics by dataset"
                    )
                    .cols_hide(columns=["n", "se_est_ci", "se_est_pi"])
                    .cols_align("center")
                    .tab_stub(rowname_col="dataset")
                    .tab_stubhead(label="dataset")
                    .fmt_number(decimals=3)
                )
            )
        }

    ///

    /// details | (c) Why do you think the LS fits are identical even though the scatter plots are quite different? What does this say about the importance of plotting the data before fitting a model?

    Because each LS fit statistic is only a scalar summary of many data points, a great deal of information is by necessity lost about the actual points layout. This highlights the importance of the scatter plot before fitting, which sometimes reveals unique features of the relationship that evade the fit statistics. 
    ///

    /// details |  (d) What does this say about $r^2$ or the $t$-statistic for testing the significance of $\hat{{\beta}}_1$ as measures of the linearity of relationships?

    Again, the same $r^2$ or $t$-statistic may correspond to vastly different actual relationships. These "measures of linearity" may not be as concrete as one thinks and therefore should be taken with a grain of salt. Datasets "x1y2" and "x2y4" are good reminders.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(Regression, md, mo):
    mo.md(
        rf"""
    ### Ex 10.23

    The approximate gestation time and birthweights are given in the following table for selected mammals. The gestation time $t$ (between fertilization and birth) for a mammal is related to the birthweight $w$ by the relationship $t = a\,b^w$, where $a$ and $b$ are constants. Regression methods can be used to estimate $a$ and $b$ by transforming this relationship into a linear model.

    {
            mo.center(
                mo.as_html(
                    Regression.gt(23)
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
def _(alt, col, df_ex, linreg_chart, mo, np):
    _df = df_ex(23).select(col("kg").alias("x"), col("days").log().alias("y"))
    _scatter = _df.plot.scatter(
        x=alt.X("x").scale(zero=False, padding=10).title("weight"),
        y=alt.Y("y").scale(zero=False, padding=10).title("log(gestation time)"),
    )
    _r = linreg_chart(_df, x_title="weight", y_title="log(gestation time)")
    _y = _r.β0 + _r.β1 * 1.2


    mo.md(
        rf"""
    /// details | (a) Apply the log transformation to obtain a linear model of the form $y = \beta_0 + \beta_1 w$, where $y = \log t$. How are $\beta_0$ and $\beta_1$ related to $a$ and $b$, respectively?

    Apply $\log$ to both sides of the equation $t = a\;b^w$ to get
    $$\log t = \log a + \log b \cdot w.$$
    So $\beta_0 = \log a$ and $\beta_1 = \log b$.
    ///

    /// details |  (b) Plot log(gestation time) vs. weight. Is this relationship approximately linear?

    {mo.as_html(_scatter)}

    Yes, the relationship is approximately linear.
    ///

    /// details | (c) Fit the linear model $y = \beta_0 + \beta_1 w$ to the transformed data.

    {mo.as_html(_r.chart_ls | _r.chart_err)}
    ///

    /// details | (d) Using the fitted model in (c), estimate the gestation time of a lion which weighs approximately 1.2 kg at birth.

    $\log t = {_r.β0:.3g} + {_r.β1:.3g} \times 1.2 = {_y:.3g}$. So $t = \exp({_y:.3g}) = {np.exp(_y):.3g}$ days.
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
def _(alt, col, df_ex, linreg_chart, mo, np):
    _df = df_ex(24).select(col("l").log().alias("x"), col("h").log().alias("y"))
    _scatter = _df.plot.scatter(
        x=alt.X("x").scale(zero=False, padding=10).title("log(days)"),
        y=alt.Y("y").scale(zero=False, padding=10).title("log(cost)"),
    )
    _r = linreg_chart(_df, x_title=("log(days)"), y_title="log(cost)")

    _y = _r.β0 + _r.β1 * np.log(3)

    mo.md(
        rf"""
    /// details | (a) Apply the log transformation to obtain a linear model of the form $y = \beta_0 + \beta_1 x$, where $x = \log l$ and $y = \log h$. How are $\beta_0$ and $\beta_1$ related to $a$ and $b$, respectively?

    Apply $\log$ to both sides of $h = a\,l^b$ to get
    $$\log h = \log a + b \log l.$$
    So $\beta_0 = \log a$ and $\beta_1 = b$.
    ///

    /// details | (b) Plot $y = \log h$ vs. $x = \log l$. Is this relationship approximately linear?

    {mo.as_html(_scatter)}

    Yes, the relationship is approximately linear.
    ///

    /// details |  (c) Fit the linear model $y = \beta_0 + \beta_1 x$ to the transformed data.

    {mo.as_html(_r.chart_ls | _r.chart_err)}
    ///

    /// details |  (d) Use the fitted model from (c) to estimate the average reimbursed cost for a 3-day hospital stay by an elderly person.

    $\log h = {_r.β0:.3g} + {_r.β1:.3g} \times \log 3 = {_y:.3g}$. So $h = \exp({_y:.3g}) = {np.exp(_y):.0f}$ dollars.
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
    _Correlation analysis_ assumes that the data $\{(x_i, y_i), i = 1, 2, \ldots , n\}$ form a random sample from a _bivariate normal distribution_ with correlation coefficient $\rho$. An estimate of $\rho$ is the sample correlation coefficient $r$. An exact test of $H_0: \rho = 0$ is a $t$-test with $n-2$ d.f. based on the test statistic $$t = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}}.$$

    This equals $t = \hat{\beta}_1 / \textrm{SE}(\hat{\beta}_1)$ which is used to test $H_0: \beta_1 = 0$ in the related regression model. In other cases only approximate large sample inferences are available. These inferences use the parameterization $$\psi = \frac{1}{2}\log_e \left(\frac{1+\rho}{1-\rho}\right).$$

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

    Show that$$\sum_{i=1}^n (y_i - \hat{y}_i)(\hat{y}_i - \bar{y}) = 0.$$

    (_Hint_: Substitute $\hat{y}_i = \beta_0 + \beta_1 x_i = \bar{y} + \hat{\beta}_1 (x_i - \bar{x})$ and simplify.)
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
