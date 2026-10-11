# **Advertising budget optimizer**

## **Goal**

Determine how advertising spend across three channels (Social Media, YouTube, TV) affects sales and recommend an optimal budget allocation.

## **Technologies**

Language: Python

IDE: PyCharm

## **Table of content**

- [Technologies](#technologies)
- [Dataset Used](#dataset-used)
- [Cleaning data](#Cleaning-data)
- [Exploring with Visuals](#Exploring-with-Visuals)
- [Linear regression](#linear-regression)
- [Predictions](#Predictions)
- [Costs optimizer](#costs-optimizer)
- [Conclusion](#conclusion)

## **Dataset Used**

This dataset was provided as part of a university project and consists of 1017 observations regarding different advertising campaigns,  investments in advertising on each channel (Social Media, YouTube, TV) and the resulting sales, which was calculated after a fixed period of time, so that the ads effect could be captured.

## **Cleaning data**

Let's take a look at the data: 
```python
print(data.shape)
print(data.head(10))
```
 
```python
print(data.info())
```
 
A preliminary review indicates: 
- some  non-numeric artifacts; 
- different case and spaces in column names; 
- wrong data types. 
  
***Header cleaning*** and changing ***data types*** from string to float: 

```python
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('sn','sm')
data = data.rename(columns={'viewsyt': 'views_yt'})

def clean_numeric_column(series):
    cleaned = series.astype(str).str.replace(r'[^\d.\-]', '', regex=True) #getting rid of non-numeric characters
    return pd.to_numeric(cleaned, errors='coerce') # changing data type

columns_to_clean = ['sm_budget', 'tv_budget', 'views_sm', 'views_tv']
for col in columns_to_clean:
    data[col] = clean_numeric_column(data[col])
print(data.head(10))
print(data.dtypes)
```
Results:
 
 
From df.info(), we can see that some features have ***missing values***, let’s look closer and check for ***duplicates***:
```python
print('missing values:\n', data.isnull().sum())
print('duplicated:', data.duplicated().sum())
```
We have 17 duplicates and some missing values:

 
 
We do not know why these values are missing. In a real-world scenario, we would first attempt to recover them by checking with colleagues and management. However, for this project, we bypass this step and drop the rows with missing data directly.
The duplicated rows are completely identical across all fields, so we can safely remove them.

```python
data = data.dropna()
data = data.drop_duplicates()
```



***Handling outliers*** using IQR method:

```python
for col in ['sm_budget', 'yt_budget', 'tv_budget', 'sales']:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 10 * IQR
    lower_bound = 0

    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
    if not outliers.empty:
        print(f'outliers for {col}:\n', outliers)
    else:
        print (f'there are no outliers in {col}')
```

 
During the outlier detection process, no abnormally large values were identified. However, negative values were found in the yt_budget and sales columns. These are not statistical outliers in the usual sense, but still have to be fixed because both metrics cannot be negative. In a production environment, the best practice is to consult the person responsible for data entry to determine the root cause. For the purposes of this project, we assume these entries were caused by a typographical error during manual entry and correct them by taking their absolute values.
```python
data[col] = data[col].abs()
```

## **Exploring with Visuals**

We use histograms together with two reference curves, a normal (red) and a gamma (green) distribution, fitted to the data to see how spend is actually distributed within each channel, not just its range, but where most campaigns are concentrated.
 
***Social Media*** and ***YouTube*** spend are both spread fairly evenly across their full range, without a clear concentration around any single "typical" budget, so neither curve fits particularly well,  both curves assume a central peak that the data doesn't really have.
 
 
***TV*** spend is clearly right-skewed, most campaigns are not expensive, with a long tail of larger ones. The gamma curve represents this kind of skew, fits noticeably better than the normal curve. This means most of the campaigns have lower TV budgets.
Describe the correlation between money spend on each channel and sales using scatterplots:

 
For simple linear regression, let’s consider only the effect of social media ads on sales. We can see a clear relationship between the two.
 
Same goes for YouTube. 

 
But for TV ads spending, linear regression works not that good.

## **Linear regression**

Linear regression estimates how much each channel contributes to sales while holding the other channels constant, producing a coefficient per channel. These are then used to decide where budget should be allocated for the greatest increase in sales.
First we built ***simple linear regression*** for each channel to see how much of sales variation each channel explains on its own.
```python
simple_models = {}
xs = ['sm_budget', 'yt_budget', 'tv_budget']
y = data['sales']
for col in xs:
    xs_const = sm.add_constant(data[col])
    rr1 = sm.OLS(y, xs_const)
    r1 = rr1.fit()
    simple_models[col] = r1
    print(col)
    print(r1.summary())
```
About 62% of the variability of sales is explained by the amount spent on social media ads:

 
 
YouTube about 39%:
 
 
TV explains only 4.7% on its own, TV spend tells us almost nothing about how much sales a campaign will generate.
 
 
The ***multiple linear regression*** shows each channel’s unique contribution, after controlling for the other two. Comparing models highlights how much of each channel’s apparent effect is actually shared with the other channels rather than unique to it.

```python
#multiple linear regression (sklearn)
x_all = data.drop(['sales',"id",'views_sm','views_yt','views_tv'], axis = 1)
y = data['sales'].values.reshape(-1,1)
r2 = LinearRegression()
r2.fit(x_all, y)
print(r2.coef_)
print(r2.intercept_)
print("The linear model is: Y = {:.5} + {:.5}*sm + {:.5}*yt + {:.5}*tv".format(r2.intercept_[0], r2.coef_[0][0], r2.coef_[0][1], r2.coef_[0][2]))
print("R^2: ", r2.score(x_all, y))

#multiple linear regression (statmodels)
x_all2 = np.column_stack((data['sm_budget'], data['yt_budget'], data['tv_budget']))
y = data['sales']
x_all_const = sm.add_constant(x_all2)
rr3 = sm.OLS(y, x_all_const)
r3 = rr3.fit()
print(r3.summary())
```
 
The three channels together explain about 96% of the variation in sales, far more than any single channel alone, confirming that sales depend on the combination of all three, not just one.
 
 
Social Media and YouTube both have 0 p-values, strong evidence that their effect on sales is real rather than chance. 
TV’s coefficient is negative and only marginally statistically significant, because p-value is right at the conventional 0.05 threshold. 
To rule out multicollinearity as an explanation for TV’s small negative coefficient we used Variance Inflation Factors (VIF):

```python
x = data[['sm_budget', 'yt_budget', 'tv_budget']]
x_const = sm.add_constant(x)
vif = pd.DataFrame()
vif['feature'] = x_const.columns
vif['VIF'] = [variance_inflation_factor(x_const.values, i) for i in range(x_const.shape[1])]
print(vif)
```
All VIFs values are close to 1,
 
well below the common threshold of 5, this means correlation between the three channels is not strong enough to destabilize the coefficient estimates in the full model. It doesn't mean there is no correlation at all, though.

Let's look at the correlation matrix:
 
TV and YouTube budgets are moderately correlated with each other (r = 0.35), while social media is essentially uncorrelated with either (r < 0.05). This moderate correlation between TV and YouTube is too weak to cause multicollinearity problems within the full model, but it is enough to explain what happens next. On its own, TV spend is positively and statistically significantly associated with sales (coef = 0.048, p < 0.001), meaning we can be very confident this effect isn't just chance, with almost 1,000 campaigns, even a weak relationship is easy to detect precisely. But that confidence says nothing about how useful the effect is: TV spend alone explains very little of the variation in sales (R² = 0.047). Once social media and YouTube are included in the same model, TV's coefficient flips to small and negative (coef = -0.003), and becomes only marginally significant (p = 0.047). This is not explained by multicollinearity, rather, it suggests that TV's apparent effect in the simple regression was largely borrowed from its moderate correlation with YouTube spend specifically, which has a real, strong effect on sales (campaigns that spend more on TV also tend to spend somewhat more on YouTube). Once YouTube (and Social Media) are explicitly controlled for, almost none of TV's apparent effect remains unique to it.

## **GUI**

Visuals:  view the exploratory charts described above (histograms with fitted distributions, and scatterplots of spend vs. sales) for each advertising channel.
 
Predict sales:  enter a specific budget for each channel and get a predicted sales figure.
  
Advertising costs optimizer: enter a total advertising budget and get a suggested split across the three channels.

## **Predictions**

Let's predict sales using multiple regression model:
```python
def predict():
    inv1 = float(inv_sm.get())
    inv2 = float(inv_yt.get())
    inv3 = float(inv_tv.get())
    sales = (r2.intercept_[0] 
             + r2.coef_[0][0] * inv1 
             + r2.coef_[0][1] * inv2 
             + r2.coef_[0][2] * inv3)
    count_f3.config(text = "Your earnings will be $" + str(round(sales, 2)) + "k.")
```
 
## **Costs optimizer**

Given a total advertising budget, this finds how to split it across the three channels to maximize sales, using linprog and the coefficients from the multiple regression model. So the task becomes "maximize total predicted sales" subject to "total spend ≤ budget" and "spend per channel ≥ 0".

Each channel gets an individual upper bound, built from the average cost per view and the average number of views. This keeps the optimizer from suggesting we pour the entire budget into one channel at a scale far beyond anything in the data, the model doesn't know if its coefficients still hold that far outside what was actually observed.

Because TV's coefficient in the multiple regression is slightly negative, `linprog` automatically assigns it $0.

```python 
CostByView = [round(data["sm_budget"] * 1000 / data["views_sm"], 2),
              round(data["yt_budget"] * 1000 / data["views_yt"], 2),
              round(data["tv_budget"] * 1000 / data["views_tv"], 2)]
AvgCost = [round(np.average(CostByView[0]), 2),
           round(np.average(CostByView[1]), 2),
           round(np.average(CostByView[2]), 2)]
AvgViews = [round(np.average(data['views_sm']), 0),
            round(np.average(data['views_yt']), 0),
            round(np.average(data['views_tv']), 0)]

def find_best():
    invest = float(all_inv.get())*1000 # from $k to $

    coefs = np.array([r2.coef_[0][0], r2.coef_[0][1], r2.coef_[0][2]])
    caps = np.array([AvgViews[i] * AvgCost[i] for i in range(3)])  # realistic spend ceiling per channel

    c = -coefs # linprog minimizes by default, so we negate the coefficients to maximize sales
    A_ub = [[1, 1, 1]]  # total spend across channels
    b_ub = [invest]  # must not exceed the budget
    bounds = [(0, caps[i]) for i in range(3)]  # each channel is capped individually

    result = linprog(c, A_ub = A_ub, b_ub = b_ub, bounds = bounds, method = 'highs')
    profit_sm, profit_yt, profit_tv = result.x
    rest = invest - result.x.sum()

    profit_sm = round(profit_sm / 1000, 1)
    profit_yt = round(profit_yt / 1000, 1)
    profit_tv = round(profit_tv / 1000, 1)
    rest = round(rest / 1000, 1)
    earning = r2.intercept_[0] + r2.coef_[0][0] * profit_sm + r2.coef_[0][1] * profit_yt + r2.coef_[0][2] * profit_tv
    count1_f2.config(text="Your earnings will be $" + str(round(earning, 2))+"k.")
    message = ("Deposits in SM $"+str(profit_sm)
                          +"k, in YT "+str(profit_yt)
                          +"k, in TV $"+str(profit_tv)
                          +"k, and rest $"+str(rest)+"k.")
    if rest > 0:
        message += '\nNote: some budget coudn\'t be allocated'
    count2_f2.config(text = message)
```
 
## **Conclusion**

This project walked through the full process of turning a raw, messy advertising dataset into a usable model and building both simple and multiple linear regression models to understand how Social Media, YouTube and TV spend relate to sales. 

Social Media and YouTube are the channels we can actually rely on based on this data, spend on either one is clearly associated with higher sales, both on their own and after controlling for the other channels. 

TV spend does not show this: although a positive relationship with sales is detectable, it is weak and inconsistent on its own, and once Social Media and YouTube spend are accounted for, TV's apparent contribution disappears almost entirely. This means we cannot recommend TV as a channel to scale up based on this data because any apparent return from TV spend looks like it's riding on the same campaigns that also spend more on YouTube, rather than TV driving sales on its own.

Given a fixed budget, this points to prioritizing Social Media and YouTube allocation over TV, which is exactly what the optimizer tool in this project does automatically. That said, the model is built on spend ranges and assumes a simple linear relationship, it does not capture diminishing returns at very high spend, nor can it tell us whether TV might behave differently under a different strategy (a new creative, a different market) than what's reflected in this dataset. It should be used as a starting point for budget decisions, not a substitute for testing.
The GUI lets you explore these results interactively: view the visuals and regressions, predict sales for any custom budget split, and get the optimal split for a given total budget.

Thank you for reading!
