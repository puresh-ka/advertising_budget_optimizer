import matplotlib
import tkinter as tk
import pandas as pd
import numpy as np
import os
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy.stats as stat
from scipy.optimize import linprog
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

data = pd.read_csv('Advertising_budget.csv')
#data = data.drop(['Unnamed: 0'], axis = 1)

print(data.shape)
print(data.head(10))
print(data.info())

#fixing column names
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('sn','sm')
data = data.rename(columns={'viewsyt': 'views_yt'})

def clean_numeric_column(series):
    cleaned = series.astype(str).str.replace(r'[^\d.\-]', '', regex = True) #getting rid of non-numeric characters
    return pd.to_numeric(cleaned, errors = 'coerce') # changing data type

columns_to_clean = ['sm_budget', 'tv_budget', 'views_sm', 'views_tv']
for col in columns_to_clean:
    data[col] = clean_numeric_column(data[col])
print(data.head(10))
print(data.dtypes)

print('missing values:\n', data.isnull().sum())
print('duplicated:', data.duplicated().sum())

missing_rows = data[data.isnull().any(axis = 1)]
if not missing_rows.empty:
    print('\nrows with missing values: ')
    print(missing_rows)

duplicate_rows = data[data.duplicated(keep = False)].sort_values(by = list(data.columns))
if not duplicate_rows.empty:
    print('\nduplicated rows: ')
    print(duplicate_rows)

#removing duplicates and rows with missing values
data = data.dropna()
data = data.drop_duplicates()
print('after removing:')
print(data.shape)

#checking outliers
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
    data[col] = data[col].abs()

#vif
x = data[['sm_budget', 'yt_budget', 'tv_budget']]
x_const = sm.add_constant(x)
vif = pd.DataFrame()
vif['feature'] = x_const.columns
vif['VIF'] = [variance_inflation_factor(x_const.values, i) for i in range(x_const.shape[1])]
print(vif)

print(data[['sm_budget', 'yt_budget', 'tv_budget']].corr())

#simple linear regression
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



#MAIN WINDOW
root = tk.Tk()
root.title("Menu")
root.geometry('400x300+100+50')
button1 = tk.Button(text = "Visuals")
button1.place(x=125, y=50, width=150)
button2 = tk.Button(text = "Advertising costs optimizer")
button2.place(x=125, y=130, width=150)
button3 = tk.Button(text = "Predict sales")
button3.place(x=125, y=210, width=150)

def AddW1(event):
    f1 = tk.Toplevel()
    f1.title("Visuals")
    f1.geometry('400x400')

    labels = {
        'sm_budget': 'Social media',
        'yt_budget': 'YouTube',
        'tv_budget': 'TV'
    }

    def regr(col):
        x_const = sm.add_constant(data[col])
        predictions = simple_models[col].predict(x_const)
        plt.figure(figsize = (12, 6))
        plt.scatter(
            data[col],
            data['sales'],
            c = 'black',
            alpha = 0.5
        )
        plt.plot(
            data[col],
            predictions,
            c = 'blue',
            linewidth = 2
        )
        plt.xlabel(f"Money spent on {labels[col]} ads ($)")
        plt.ylabel("Sales ($)")
        plt.show()

    def hist(col):
        N = len(data)
        k = round(N ** 0.5)
        histogr, b = np.histogram(data[col], k, density=True)
        plt.hist(data[col], bins=b, density=True)

        xx = np.linspace(np.min(data[col]), np.max(data[col]), 100)
        Mx = np.average(data[col])
        Sx = np.std(data[col])
        y = ((1 / (np.sqrt(2 * np.pi) * Sx)) * np.exp(-0.5 * (1 / Sx * (xx - Mx)) ** 2))

        plt.plot(xx, y, '-r')

        a, dd, p = stat.gamma.fit(data[col])

        yyy = stat.gamma.pdf(xx, a = a, loc = dd, scale = p)
        plt.plot(xx, yyy, '-g')
        plt.title(f'{labels[col]}')
        plt.show()

    tk.Label(f1, text="Social media").place(x=20, y=60)
    tk.Label(f1, text="YouTube").place(x=20, y=100)
    tk.Label(f1, text="TV").place(x=20, y=140)
    tk.Button(f1, text = "R", width = 4, command = lambda: regr('sm_budget')).place(x=250, y=60)
    tk.Button(f1, text = "R", width = 4, command = lambda: regr('yt_budget')).place(x=250, y=100)
    tk.Button(f1, text = "R", width = 4, command = lambda: regr('tv_budget')).place(x=250, y=140)
    tk.Button(f1, text = "H", width = 4, command = lambda: hist('sm_budget')).place(x=300, y=60)
    tk.Button(f1, text = "H", width = 4, command = lambda: hist('yt_budget')).place(x=300, y=100)
    tk.Button(f1, text = "H", width = 4, command = lambda: hist('tv_budget')).place(x=300, y=140)

def AddW2(event):
    f2 = tk.Toplevel()
    f2.title("Advertising costs optimizer")
    f2.geometry('500x250')
    count1_f2 = tk.Label(f2)
    count1_f2.place(x=20, y=90)
    count2_f2 = tk.Label(f2)
    count2_f2.place(x=20, y=130)

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

    tk.Label(f2, text = "Enter your advertising costs:").place(x=10, y=10)
    all_inv = tk.Entry(f2, width=10)
    all_inv.place(x=20, y=50)
    tk.Button(f2, text = "Find", width=20, command=find_best).place(x=20, y=170)

def AddW3(event):
    f3 = tk.Toplevel()
    f3.title("Predict sales")
    f3.geometry('350x600')
    count_f3 = tk.Label(f3)
    count_f3.place(x=20, y=270)

    def predict():
        inv1 = float(inv_sm.get())
        inv2 = float(inv_yt.get())
        inv3 = float(inv_tv.get())
        sales = (r2.intercept_[0]
                 + r2.coef_[0][0] * inv1
                 + r2.coef_[0][1] * inv2
                 + r2.coef_[0][2] * inv3)
        count_f3.config(text = "Your earnings will be $" + str(round(sales, 2)) + "k.")

    tk.Label(f3, text="Spread the cost of advertising:").place(x=10, y=10)
    tk.Label(f3, text="Social media").place(x=20, y=40)
    inv_sm = tk.Entry(f3, width=10)
    inv_sm.place(x=20, y=70)
    tk.Label(f3, text="YouTube").place(x=20, y=110)
    inv_yt = tk.Entry(f3, width=10)
    inv_yt.place(x=20, y=150)
    tk.Label(f3, text="TV").place(x=20, y=190)
    inv_tv = tk.Entry(f3, width=10)
    inv_tv.place(x=20, y=230)
    tk.Button(f3, text="Predict sales", width=20, command=predict).place(x=20, y=310)


button1.bind('<Button-1>', AddW1)   #open "Visuals" window
button2.bind('<Button-1>', AddW2)   #open "Advertising costs optimizer" window
button3.bind('<Button-1>', AddW3)   #open "Predict sales" window
root.mainloop()