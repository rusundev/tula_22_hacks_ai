#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system("cp /content/drive/MyDrive/Colab/Data/'Готовые задачи'/Калининград/participants/train/train.csv ./ ")
get_ipython().system("cp /content/drive/MyDrive/Colab/Data/'Готовые задачи'/Калининград/participants/test/test.csv ./ ")


# In[2]:


#Установка catboost
get_ipython().system('pip install catboost')


# In[3]:


#import необходимых модулей

import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[4]:


#Считывание данных в DataFrame 

train = pd.read_csv('train.csv', sep=';', index_col=None, dtype={'PATIENT_SEX':str, 'MKB_CODE':str, 'ADRES':str, 'VISIT_MONTH_YEAR':str, 'AGE_CATEGORY':str, 'PATIENT_ID_COUNT':int})
test = pd.read_csv('test.csv', sep=';', index_col=None, dtype={'PATIENT_SEX':str, 'MKB_CODE':str, 'ADRES':str, 'VISIT_MONTH_YEAR':str, 'AGE_CATEGORY':str})


# In[5]:


#Отделение меток от данных

X = train[['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY']]
y = train[['PATIENT_ID_COUNT']]


# In[274]:


train['month_year'] = pd.to_datetime(train.loc[:, 'VISIT_MONTH_YEAR'], format='%m.%y', errors='ignore')


# In[275]:


train


# In[8]:


unique_months = X['VISIT_MONTH_YEAR'].unique().astype('str')


# In[9]:


unique_months


# In[10]:


df_uniq_months = pd.to_datetime(unique_months, format='%m.%y', errors='ignore')


# In[11]:


df_uniq_months.sort_values()


# In[12]:


df_uniq_months.sort_values(ascending=True)


# In[13]:


uniq_city = X['ADRES'].unique()


# In[14]:


print(uniq_city)


# In[85]:


# TODO: get population by city name


# In[ ]:


# TODO: get lat, lon by city name


# In[ ]:


# TODO: get weather record by city name


# In[ ]:


# TODO: research periodicity


# In[86]:


# TODO: EDA


# In[127]:


train.head()


# In[135]:


train.groupby('PATIENT_SEX').sum('PATIENT_ID_COUNT')


# In[138]:


train.groupby('AGE_CATEGORY').sum('PATIENT_ID_COUNT').sort_values(by='PATIENT_ID_COUNT', ascending=False)


# In[139]:


train.groupby('ADRES').sum('PATIENT_ID_COUNT').sort_values(by='PATIENT_ID_COUNT', ascending=False)


# In[140]:


train.groupby('VISIT_MONTH_YEAR').sum('PATIENT_ID_COUNT').sort_values(by='PATIENT_ID_COUNT', ascending=False)


# In[16]:


import matplotlib.pyplot as plt


# In[31]:


mkb_count = train.groupby('MKB_CODE').sum('PATIENT_ID_COUNT').reset_index()


# In[32]:


mkb_count.describe()


# In[33]:


mkb_count.index


# In[43]:


mkb_count.sort_values(by=['PATIENT_ID_COUNT'], ascending=False, inplace=True)


# In[44]:


mkb_count.head()


# In[46]:


mkb_count.query('PATIENT_ID_COUNT > 10000')


# In[68]:


mkb_count.query('PATIENT_ID_COUNT > 10000').iloc[0:10,0]


# In[73]:


type(mkb_count.query('PATIENT_ID_COUNT > 10000').iloc[:, 0])


# In[76]:


mkb_top_list = mkb_count.query('PATIENT_ID_COUNT > 10000').iloc[:, 0].tolist()


# In[78]:


mkb_top_list


# J06.9 ОРВИ
# Z25.8 Необходимость иммунизации против одной из других вирусных болезней (коронавирус?)
# Z00.0 Общий медицинский осмотр
# I11.9 

# In[79]:


get_ipython().system('pip install icd10-cm')


# In[80]:


import icd10

code = icd10.find("J20.0")
print(code.description)         # Acute bronchitis due to Mycoplasma pneumoniae
if code.billable:
    print(code, "is billable")  # J20.0 is billable

print(code.chapter)             # X
print(code.block)               # J00-J99
print(code.block_description)   # Diseases of the respiratory system


# In[81]:


import icd10

if icd10.exists("J20.0"):
    print("Exists")


# In[113]:


mkb_top_name_dict = {}


# In[114]:


for mkb in mkb_top_list:
    code = icd10.find(mkb)
    mkb_top_name_dict[mkb] = code.description if code is not None else 'NOT FOUND'


# In[115]:


mkb_top_name_dict


# In[119]:


mkb_top_name_dict['K58.9']


# In[94]:


get_ipython().system('pip install simple-icd-10')


# In[95]:


import simple_icd_10 as icd


# In[96]:


icd.get_description("XII")


# In[103]:


mkb_top_name_simple_dict = {}
cnt_err = 0
for mkb in mkb_top_list:
    try:
        mkb_desc = icd.get_description(mkb)
        
        mkb_top_name_simple_dict[mkb] = mkb_desc
    except ValueError as e:
        mkb_top_name_simple_dict[mkb] = 'NOT FOUND'
        print(mkb)
        cnt_err += 1

print(cnt_err)


# In[101]:


mkb_top_name_simple_dict


# In[123]:


for mkb_code, mkb_name in mkb_top_name_simple_dict.items():
    if mkb_name == 'NOT FOUND':
        print(mkb_code, mkb_name, sep=':')


# In[124]:


mkb_top_name_simple_dict['K58.9'] = mkb_top_name_dict['K58.9']


# In[125]:


for mkb_code, mkb_name in mkb_top_name_simple_dict.items():
    if mkb_name == 'NOT FOUND':
        print(mkb_code, mkb_name, sep=':')


# In[126]:


mkb_top_name_simple_dict


# In[143]:


mkb_count


# In[155]:


mkb_top_plot = mkb_count.query('PATIENT_ID_COUNT > 10000').loc[:, ['MKB_CODE', 'PATIENT_ID_COUNT']].reset_index(drop=True)


# In[158]:


plt.figure(figsize=(16, 8), dpi=80)
plt.bar(mkb_top_plot['MKB_CODE'], mkb_top_plot['PATIENT_ID_COUNT'])
plt.show()


# In[161]:


train.groupby(['ADRES', 'MKB_CODE']).value_counts()


# In[276]:


top_mkb = train.query('ADRES == "Калининград" & MKB_CODE == "J06.9"')
top_mkb


# In[277]:


top_mkb_f = top_mkb.query('PATIENT_SEX == "0"')
top_mkb_f


# In[278]:


top_mkb_m = top_mkb.query('PATIENT_SEX == "1"')
top_mkb_m


# In[297]:


list(range(2018, 2023))


# In[320]:


import datetime as dt
plt.figure(figsize=(12, 6), dpi=80)
plt.plot(top_mkb.groupby('month_year').sum('PATIENT_ID_COUNT'), marker = '+')
plt.plot(top_mkb_f.groupby('month_year').sum('PATIENT_ID_COUNT'), marker = 'v')
plt.plot(top_mkb_m.groupby('month_year').sum('PATIENT_ID_COUNT'), marker = '^')
[plt.axvline([dt.datetime(year, 1, 1)]) for year in range(2018, 2023)]
plt.grid()
plt.show()


# In[321]:


plt.figure(figsize=(12, 6), dpi=80)

top_mkb_by_age = top_mkb.groupby('AGE_CATEGORY').groups.keys()
for age_group in top_mkb_by_age:
    age_group_filtered = top_mkb.query('AGE_CATEGORY == @age_group').sort_values(by='month_year', ascending=True)
    plt.plot(age_group_filtered.loc[:, 'month_year'], age_group_filtered.loc[:, 'PATIENT_ID_COUNT'], marker = 'o')

[plt.axvline([dt.datetime(year, 1, 1)]) for year in range(2018, 2023)]
plt.legend(list(top_mkb_by_age))
plt.grid()
plt.show()


# In[ ]:





# In[351]:


# Moving Average
top_mkb_total = top_mkb.groupby('month_year').sum('PATIENT_ID_COUNT')
#print(top_mkb_total)
moving_average = top_mkb_total.rolling(
    window=12,       # 365-day window
    center=True,      # puts the average at the center of the window
    min_periods=6,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)

ax = top_mkb_total.plot(figsize=(12, 6), style=".", color="0.5", grid=True)

moving_average.plot(
    ax=ax, linewidth=3, title="Top MKB Total Moving Average", legend=False,
);
[plt.axvline([dt.datetime(year, 1, 1)]) for year in range(2018, 2023)]
plt.grid()
plt.show()


# In[364]:


import datetime as dt

# Coranavirus turning trend point at 2020-05-01
corona_turn = dt.datetime(2020, 5, 1)
corona_turn_idx = top_mkb_total[:corona_turn].shape[0]


# In[377]:


top_mkb_total.index[0:corona_turn_idx]


# In[ ]:





# In[395]:


from statsmodels.tsa.deterministic import DeterministicProcess

dp_prior = DeterministicProcess(
    index=top_mkb_total.index[:corona_turn_idx],  # dates from the training data
    constant=True,       # dummy feature for the bias (y_intercept)
    order=1,             # the time dummy (trend)
    drop=True,           # drop terms if necessary to avoid collinearity
)

dp_after = DeterministicProcess(
    index=top_mkb_total.index[corona_turn_idx-1:],  # dates from the training data
    constant=True,       # dummy feature for the bias (y_intercept)
    order=1,             # the time dummy (trend)
    drop=True,           # drop terms if necessary to avoid collinearity
)

# `in_sample` creates features for the dates given in the `index` argument
X_prior = dp_prior.in_sample()
X_after = dp_after.in_sample()

print(X_prior.tail())
print(X_after.head())


# In[752]:


dp_after.out_of_sample(1)


# In[426]:


from sklearn.linear_model import LinearRegression

y_prior = top_mkb_total['PATIENT_ID_COUNT'][:corona_turn_idx]  # the target
y_after = top_mkb_total['PATIENT_ID_COUNT'][corona_turn_idx-1:]  # the target


# The intercept is the same as the `const` feature from
# DeterministicProcess. LinearRegression behaves badly with duplicated
# features, so we need to be sure to exclude it here.
model_prior = LinearRegression(fit_intercept=False)
model_prior.fit(X_prior, y_prior)

model_after = LinearRegression(fit_intercept=False)
model_after.fit(X_after, y_after)

y_pred_prior = pd.Series(model_prior.predict(X_prior), index=X_prior.index)
y_pred_after = pd.Series(model_after.predict(X_after), index=X_after.index)

y_pred_total = pd.concat([y_pred_prior, y_pred_after], axis=0)

print(y_pred_total.reset_index(drop=True).tail())


# In[413]:


X_prior.index


# In[408]:


plt.figure(figsize=(12, 6), dpi=80)

plt.plot(X_prior.index, y_pred_prior, marker = 'o')
plt.plot(X_after.index, y_pred_after, marker = 'o')

plt.plot(top_mkb.groupby('month_year').sum('PATIENT_ID_COUNT'), marker = '+')
plt.plot(top_mkb_f.groupby('month_year').sum('PATIENT_ID_COUNT'), marker = 'v')
plt.plot(top_mkb_m.groupby('month_year').sum('PATIENT_ID_COUNT'), marker = '^')

[plt.axvline([dt.datetime(year, 1, 1)]) for year in range(2018, 2023)]

plt.legend(['fit before corona', 'fit after corona', 'total', 'female', 'male'])

plt.grid()
plt.show()


# In[799]:


import rsnd


# In[810]:


fft_ = rsnd.fft_angle(top_mkb.groupby('month_year').sum('PATIENT_ID_COUNT'), 1)


# In[811]:


fft_


# In[813]:


plt.plot(1/fft_['frequency'], fft_['fft_amplitude'], marker='o')
plt.show()


# In[814]:


1/fft_['frequency']


# In[832]:


1/fft_['frequency'][11]


# In[837]:


fft_['fft_amplitude']


# In[838]:


# @link https://stackoverflow.com/questions/27516849/how-to-convert-list-of-numpy-arrays-into-single-numpy-array
np.concatenate(fft_['fft_amplitude'], axis=0)


# In[842]:


np.stack(fft_['fft_amplitude'], axis=0)
np.vstack(fft_['fft_amplitude'])
np.array(fft_['fft_amplitude'])


# In[848]:


s1 = pd.Series(1/fft_['frequency'])


# In[849]:


s2 = pd.Series(np.concatenate(fft_['fft_amplitude'], axis=0))


# In[855]:


f=pd.concat([s1, s2], axis = 1, names=['Period, month', 'Amplitude'])


# Раз в 3 месяца пик самый большой.

# In[858]:


plt.plot(f[0], f[1], marker='o')
plt.show()


# In[453]:


y_p_t = pd.DataFrame(y_pred_total)


# In[454]:


y_p_t.rename(columns={0: 'PATIENT_ID_COUNT'}, inplace=True)
y_p_t


# In[456]:


top_mkb.groupby('month_year').sum('PATIENT_ID_COUNT') - y_p_t


# In[458]:


plt.figure(figsize=(12, 6), dpi=80)

plt.plot(top_mkb.groupby('month_year').sum('PATIENT_ID_COUNT') - y_p_t, marker = '+')
plt.plot(top_mkb_f.groupby('month_year').sum('PATIENT_ID_COUNT') - y_p_t, marker = 'v')
plt.plot(top_mkb_m.groupby('month_year').sum('PATIENT_ID_COUNT') - y_p_t, marker = '^')

[plt.axvline([dt.datetime(year, 1, 1)]) for year in range(2018, 2023)]

plt.title('DETRENDED')

plt.legend(['total', 'female', 'male'])

plt.grid()
plt.show()


# In[682]:


# @link https://www.kaggle.com/code/ryanholbrook/hybrid-models/tutorial


# In[684]:


y_detrend = top_mkb.groupby('month_year').sum('PATIENT_ID_COUNT') - y_p_t
y_detrend


# In[687]:


from scipy import integrate

#y_detrend.apply(lambda g: integrate.trapz(g.PATIENT_ID_COUNT, x=g.month_year))


# In[694]:


y_detrend[:52].sum()


# In[695]:


y_p_t


# In[719]:


#model_after.predict(X)
#y_pred_after = pd.Series(model_after.predict(X_after), index=X_after.index)
X_after
X_after.index[-1:]
d = {'a': 1, 'b': 2, 'c': 3}
da = {dt.datetime(2022,4,1): 0}
ser = pd.Series(da, name='for pred')
#ser = pd.Series(data=d, index=['a', 'b', 'c'])
ser


# In[704]:


X


# In[721]:


get_ipython().system('pip install xgboost')


# In[722]:


from xgboost import XGBRegressor


# In[725]:


# Pivot wide to long (stack) and convert DataFrame to Series (squeeze)
#y_fit = y_fit.stack().squeeze()    # trend from training set
#y_pred = y_pred.stack().squeeze()  # trend from test set

# Create residuals (the collection of detrended series) from the training set
#y_resid = y_train - y_fit

# Train XGBoost on the residuals
xgb = XGBRegressor()
xgb.fit(X_after, y_detrend[corona_turn_idx:])

# Add the predicted residuals onto the predicted trends
#y_fit_boosted = xgb.predict(X_train) + y_fit
#y_pred_boosted = xgb.predict(X_test) + y_pred


# In[787]:


X_after


# In[788]:


xgb_sh_1 = XGBRegressor()
xgb_sh_1.fit(X_after[:-1], y_detrend.iloc[corona_turn_idx:-1])


# In[792]:


X_after[-1:]


# In[794]:


y_p_t[-1:]


# In[798]:


xgb_sh_1.predict(X_after[-1:])


# In[797]:


y_p_t[-1:]


# In[795]:


xgb_sh_1.predict(X_after[-1:]) + y_p_t[-1:]


# In[737]:


X_test = pd.DataFrame(data = {'PATIENT_ID_COUNT': [0]}, index=[dt.datetime(2022,4,1)])
X_test


# In[753]:


xgb.predict(dp_after.out_of_sample(1))


# In[755]:


# @ 2022-04-01
xgb.predict(dp_after.out_of_sample(1))+model_after.predict(dp_after.out_of_sample(1))


# In[739]:


X_test.index


# In[751]:


X_after


# In[ ]:


# Add the predicted residuals onto the predicted trends
y_fit_boosted = xgb.predict(X_train) + y_fit
y_pred_boosted = xgb.predict(X_test) + y_pred


# In[ ]:


# source: https://www.kaggle.com/code/ryanholbrook/seasonality/tutorial


# In[496]:


from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

simplefilter("ignore")

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# annotations: https://stackoverflow.com/a/49238256/5769929
def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta(np.timedelta64(1, "M"))
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    #ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticks([1, 2, 4, 6])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            #"Monthly (12)",
            #"Biweekly (26)",
            #"Weekly (52)",
            #"Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


#data_dir = Path("../input/ts-course-data")
#tunnel = pd.read_csv(data_dir / "tunnel.csv", parse_dates=["Day"])
#tunnel = tunnel.set_index("Day").to_period("D")


# In[472]:


X_n = top_mkb.groupby('month_year').sum('PATIENT_ID_COUNT').copy()

# days within a week
X_n["day"] = X_n.index.dayofweek  # the x-axis (freq)
X_n["week"] = X_n.index.week  # the seasonal period (period)

# days within a year
X_n["month"] = X_n.index.month
X_n["year"] = X_n.index.year

fig, ax0 = plt.subplots(1, 1, figsize=(11, 6))

#seasonal_plot(X_n, y="PATIENT_ID_COUNT", period="week", freq="day", ax=ax0)
#seasonal_plot(X_n, y="PATIENT_ID_COUNT", period="year", freq="month", ax=ax1);
seasonal_plot(X_n, y="PATIENT_ID_COUNT", period="year", freq="month", ax=ax0);


# In[474]:


X_n.head()


# In[478]:


type(X_n.PATIENT_ID_COUNT.head())


# In[497]:


plot_periodogram(X_n.PATIENT_ID_COUNT);


# In[517]:


pd.DatetimeIndex(X_n.index)


# In[ ]:





# In[518]:


from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

fourier = CalendarFourier(freq="A", order=4)  # 10 sin/cos pairs for "A"nnual seasonality

dp_s = DeterministicProcess(
    #index=X_n.index,
    index=pd.DatetimeIndex(X_n.index),
    constant=True,               # dummy feature for bias (y-intercept)
    order=1,                     # trend (order 1 means linear)
    seasonal=True,               # weekly seasonality (indicators)
    additional_terms=[fourier],  # annual seasonality (fourier)
    drop=True,                   # drop terms to avoid collinearity
)

X_s = dp_s.in_sample()  # create features for dates in tunnel.index


# In[519]:


X_s


# In[ ]:


dp_s.out_of_sample(steps=51,forecast_index=X_n.index)


# In[537]:


X_n.index+51*np.timedelta64(1, 'M')


# In[548]:


pd.date_range(start='2022-04-01', end='2026-07-01', freq='M').shape


# In[553]:


X_n['PATIENT_ID_COUNT'].tail()


# In[534]:


X_n.index[-1]


# In[550]:


y_s = X_n['PATIENT_ID_COUNT']

model = LinearRegression(fit_intercept=False)
_ = model.fit(X_s, y_s)

y_pred = pd.Series(model.predict(X_s), index=y_s.index)
#X_fore = dp_s.out_of_sample(steps=51, forecast_index=X_n.index+51*np.timedelta64(1, 'M'))
#X_fore = dp_s.out_of_sample(steps=51, forecast_index=pd.date_range(start='2022-04-01', end='2026-07-01', freq='M'))

#y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y_s.plot(color='0.25', style='.', title="TOP MKB - Seasonal Forecast")
ax = y_pred.plot(ax=ax, label="Seasonal")
#ax = y_fore.plot(ax=ax, label="Seasonal Forecast", color='C3')
#ax.set_xlim(X_n.index[0], X_n.index[-1])
_ = ax.legend()


# In[ ]:


# Correlations https://www.kaggle.com/code/ryanholbrook/time-series-as-features/tutorial


# In[556]:


from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import periodogram
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf

simplefilter("ignore")

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax,
                     **kwargs)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax


def plot_lags(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    import math
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lags / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            ax = lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig


#data_dir = Path("../input/ts-course-data")
#flu_trends = pd.read_csv(data_dir / "flu-trends.csv")
#flu_trends.set_index(
#    pd.PeriodIndex(flu_trends.Week, freq="W"),
#    inplace=True,
#)
#flu_trends.drop("Week", axis=1, inplace=True)

#ax = flu_trends.FluVisits.plot(title='Flu Trends', **plot_params)
#_ = ax.set(ylabel="Office Visits")


# In[560]:


flu_trends = X_n.copy()
flu_trends.head()


# In[564]:


flu_trends.


# In[566]:


flu_trends.set_index(pd.PeriodIndex(flu_trends.index, freq="M"), inplace=True)


# In[568]:


ax = flu_trends.PATIENT_ID_COUNT.plot(title='Flu Trends', **plot_params)
_ = ax.set(ylabel="Office Visits")


# In[581]:


# Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Monthly

# Set time period
start = datetime(2018, 1, 1)
end = datetime(2022, 4, 30)

# Create Point for Kaliningrad
location = Point(54.710128, 20.5105838, 0)

# Get daily data for 2018
data = Monthly(location, start, end)
data = data.fetch()

# Plot line chart including average, minimum and maximum temperature
data.plot(y=['tavg', 'tmin', 'tmax'], marker='o')
plt.show()


# In[583]:


flu_trends.shape


# In[588]:


data


# In[582]:


data.shape


# In[584]:


# Correlation of flu trends with temperature weather


# In[587]:


flu_trends.PATIENT_ID_COUNT.pct_change().plot()


# In[602]:


flu_trends.PATIENT_ID_COUNT.pct_change()[1:].corr(data.tavg[:-1].pct_change()[1:])


# In[659]:


for sh in range(12):
    cor1 = pd.Series(flu_trends.PATIENT_ID_COUNT.pct_change()[1:]).reset_index(drop=True).shift(-sh).fillna(0)
    cor2 = pd.Series(data.tavg[:-1].pct_change()[1:]).reset_index(drop=True)
    print(cor1.corr(cor2))


# In[ ]:





# In[664]:


for sh in range(12):
    cor1 = pd.Series(flu_trends.PATIENT_ID_COUNT.diff()[1:]).reset_index(drop=True).shift(sh).fillna(0)
    cor2 = pd.Series(data.tavg[:-1].diff()[1:]).reset_index(drop=True)
    print(cor1.corr(cor2))


# In[661]:


for sh in range(12):
    cor = pd.Series(data.tavg).reset_index(drop=True).corr(pd.Series(flu_trends.PATIENT_ID_COUNT).reset_index(drop=True).shift(-sh).fillna(0))
    print(cor)


# In[663]:


pd.Series(data.tavg).reset_index(drop=True).diff().plot(marker='o')


# In[650]:


ft_moving_average = flu_trends.PATIENT_ID_COUNT.rolling(
    window=8,       # 365-day window
    center=True,      # puts the average at the center of the window
    min_periods=4,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)


# In[651]:


ft_moving_average.plot(marker='o')


# In[603]:


data.tavg[:-1].pct_change()[1:]


# In[596]:


flu_trends.PATIENT_ID_COUNT.pct_change()


# In[598]:


pd.Series(list(range(10))).corr(pd.Series(list(range(10))))


# In[601]:


pd.Series(data.tavg[:-1].pct_change())[1:]


# In[599]:


data.tavg[:-1].pct_change().plot()


# In[590]:


plt.scatter(flu_trends.PATIENT_ID_COUNT, data.tavg[:-1])


# In[569]:


_ = plot_lags(flu_trends.PATIENT_ID_COUNT, lags=12, nrows=2)
_ = plot_pacf(flu_trends.PATIENT_ID_COUNT, lags=12)


# In[681]:


_ = plot_lags(flu_trends.PATIENT_ID_COUNT[-36:], lags=12, nrows=2)
_ = plot_pacf(flu_trends.PATIENT_ID_COUNT[-36:], lags=12)


# In[860]:


#ACF
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(flu_trends.PATIENT_ID_COUNT, lags=12)


# In[ ]:





# In[862]:


# https://medium.com/@krzysztofdrelczuk/acf-autocorrelation-function-simple-explanation-with-python-example-492484c32711
# https://github.com/kdrelczuk/medium
from pandas import read_csv
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('monthly-sunspots.txt').drop(['Month'],axis=1)#.head(100)
data_a = data.to_numpy().T[0]
data_a
plt.figure(figsize=(20,10))
plt.plot(data_a)

plt.rc("figure", figsize=(20,10))
plt.figure(figsize=(20,10))
plot_acf(data_a, lags=90)
plt.show()

data = pd.read_csv('AirPassengers.csv').drop(['Month'],axis=1)#.head(100)
data_a = data.to_numpy().T[0]
data_a
plt.figure(figsize=(20,10))
plt.plot(data_a)

plt.rc("figure", figsize=(20,10))
plt.figure(figsize=(20,10))
plot_acf(data_a, lags=50)
plt.show()


# In[668]:


_ = plot_lags(data.tavg, lags=12, nrows=2)
_ = plot_pacf(data.tavg, lags=12)


# In[667]:


data.tavg


# In[570]:


def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)


X = make_lags(flu_trends.PATIENT_ID_COUNT, lags=1)
X = X.fillna(0.0)


# In[572]:


# Create target series and data splits
y = flu_trends.PATIENT_ID_COUNT.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, shuffle=False)

# Fit and predict
model = LinearRegression()  # `fit_intercept=True` since we didn't use DeterministicProcess
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_train), index=y_train.index)
y_fore = pd.Series(model.predict(X_test), index=y_test.index)


# In[573]:


ax = y_train.plot(**plot_params)
ax = y_test.plot(**plot_params)
ax = y_pred.plot(ax=ax)
_ = y_fore.plot(ax=ax, color='C3')


# In[574]:


ax = y_test.plot(**plot_params)
_ = y_fore.plot(ax=ax, color='C3')


# In[576]:


# Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily

# Set time period
start = datetime(2018, 1, 1)
end = datetime(2022, 4, 30)

# Create Point for Kaliningrad
location = Point(54.710128, 20.5105838, 0)

# Get daily data for 2018
data = Daily(location, start, end)
data = data.fetch()

# Plot line chart including average, minimum and maximum temperature
data.plot(y=['tavg', 'tmin', 'tmax'])
plt.show()


# In[578]:


# Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Stations, Monthly

# Set time period
start = datetime(2000, 1, 1)
end = datetime(2018, 12, 31)

# Get Monthly data
data = Monthly('10637', start, end)
data = data.fetch()

# Plot line chart including average, minimum and maximum temperature
data.plot(y=['tavg', 'tmin', 'tmax'])
plt.show()


# In[ ]:


_ = plot_lags(flu_trends.FluVisits, lags=12, nrows=2)
_ = plot_pacf(flu_trends.FluVisits, lags=12)


# In[18]:


train.groupby('MKB_CODE').sum('PATIENT_ID_COUNT').describe()


# In[153]:


train.groupby('VISIT_MONTH_YEAR').sum('PATIENT_ID_COUNT').describe()


# In[172]:


grp_by_month = train
grp_by_month['month'] = pd.to_datetime(grp_by_month['VISIT_MONTH_YEAR'], format='%m.%y', errors='ignore')
grp_by_month = grp_by_month.groupby('month').sum('PATIENT_ID_COUNT')
grp_by_month.describe()


# In[173]:


grp_by_month


# In[174]:


plt.plot(grp_by_month['PATIENT_ID_COUNT'], marker = 'o')
plt.rcParams["figure.figsize"] = (20,30)
plt.show()


# In[99]:


get_ipython().system('pip install geopy')


# In[114]:


get_ipython().system('pip install geopy')
get_ipython().system('pip install geocoder')


# In[104]:


from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="my-my-application")
location = geolocator.geocode("Балтийск Россия")
print(location.address)
#Flatiron Building, 175, 5th Avenue, Flatiron, New York, NYC, New York, ...
print((location.latitude, location.longitude))
#(40.7410861, -73.9896297241625)


# In[130]:


from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="my-my-application")
location = geolocator.geocode("Калининград Россия")
print(location.address)
#Flatiron Building, 175, 5th Avenue, Flatiron, New York, NYC, New York, ...
print((location.latitude, location.longitude))
#(40.7410861, -73.9896297241625)


# In[111]:


def city_state_country(row):
    coord = f"{row['Latitude']}, {row['Longitude']}"
    location = geolocator.reverse(coord, exactly_one=True)
    address = location.raw['address']
    city = address.get('city', '')
    state = address.get('state', '')
    country = address.get('country', '')
    row['city'] = city
    row['state'] = state
    row['country'] = country
    return row


# In[113]:


row = {'Latitude': 54.64, 'Longitude': 19.89}
row = {'Latitude': 55.644466, 'Longitude': 37.395744}
city_state_country(row)


# In[115]:


import geocoder

g = geocoder.osm([51.5074, 0.1278], method='reverse')
g.json['city']


# In[139]:


import geocoder

g = geocoder.osm([54.6437214, 19.8941584], method='reverse')
#str(g).split(',').strip()
print(g.json['town'])
g.json


# In[125]:


get_ipython().system('pip install meteostat')


# In[126]:


# Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily

# Set time period
start = datetime(2018, 1, 1)
end = datetime(2018, 12, 31)

# Create Point for Vancouver, BC
location = Point(49.2497, -123.1193, 70)

# Get daily data for 2018
data = Daily(location, start, end)
data = data.fetch()

# Plot line chart including average, minimum and maximum temperature
data.plot(y=['tavg', 'tmin', 'tmax'])
plt.show()


# In[575]:


# Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily

# Set time period
start = datetime(2018, 1, 1)
end = datetime(2022, 4, 30)

# Create Point for Kaliningrad
location = Point(54.710128, 20.5105838, 0)

# Get daily data for 2018
data = Daily(location, start, end)
data = data.fetch()

# Plot line chart including average, minimum and maximum temperature
data.plot(y=['tavg', 'tmin', 'tmax'])
plt.show()


# In[93]:


import requests
import json

def get_city_opendata(city, country):
    tmp = 'https://public.opendatasoft.com/api/records/1.0/search/?dataset=worldcitiespop&q=%s&sort=population&facet=country&refine.country=%s'
    cmd = tmp % (city, country)
    res = requests.get(cmd)
    dct = json.loads(res.content)
    return dct

#    out = dct['records'][0]['fields']
#    return out

get_city_opendata('Berlin', 'de')

#{'city': 'berlin',
# 'country': 'de',
# 'region': '16',
# 'geopoint': [52.516667, 13.4],
# 'longitude': 13.4,
# 'latitude': 52.516667,
# 'accentcity': 'Berlin',
# 'population': 3398362}

get_city_opendata('San Francisco', 'us')

#{'city': 'san francisco',
# 'country': 'us',
# 'region': 'CA',
# 'geopoint': [37.775, -122.4183333],
# 'longitude': -122.4183333,
# 'latitude': 37.775,
# 'accentcity': 'San Francisco',
# 'population': 732072}


# In[91]:


get_ipython().system('pip install qwikidata')


# In[141]:


import qwikidata
import qwikidata.sparql

def get_city_wikidata(city, country):
    query = """
    SELECT ?city ?cityLabel ?country ?countryLabel ?population
    WHERE
    {
      ?city rdfs:label '%s'@en.
      ?city wdt:P1082 ?population.
      ?city wdt:P17 ?country.
      ?city rdfs:label ?cityLabel.
      ?country rdfs:label ?countryLabel.
      FILTER(LANG(?cityLabel) = "en").
      FILTER(LANG(?countryLabel) = "en").
      FILTER(CONTAINS(?countryLabel, "%s")).
    }
    """ % (city, country)

    res = qwikidata.sparql.return_sparql_query_results(query)
    out = res['results']['bindings'][0]
    return out

get_city_wikidata('Berlin', 'Germany')

#{'city': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q64'},
# 'population': {'datatype': 'http://www.w3.org/2001/XMLSchema#decimal',
#  'type': 'literal',
#  'value': '3613495'},
# 'country': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q183'},
# 'cityLabel': {'xml:lang': 'en', 'type': 'literal', 'value': 'Berlin'},
# 'countryLabel': {'xml:lang': 'en', 'type': 'literal', 'value': 'Germany'}}

get_city_wikidata('San Francisco', 'America')

#{'city': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q62'},
# 'population': {'datatype': 'http://www.w3.org/2001/XMLSchema#decimal',
#  'type': 'literal',
#  'value': '805235'},
# 'country': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q30'},
# 'cityLabel': {'xml:lang': 'en', 'type': 'literal', 'value': 'San Francisco'},
# 'countryLabel': {'xml:lang': 'en',
#  'type': 'literal',
#  'value': 'United States of America'}}



get_city_wikidata('Baltiysk', 'Russia')


# In[142]:


get_city_wikidata('Kaliningrad', 'Russia')


# In[56]:


import matplotlib.pyplot as plt


# In[69]:


plt.plot(df_uniq_months.sort_values(ascending=True), marker='.')
plt.show()


# In[ ]:





# In[6]:


#Разделение на train/test для локального тестирования

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


# In[7]:


#Создание объекта данных Pool, плюсы: возможность указать какие признаки являются категориальными

pool_train = Pool(X_train, y_train, cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY'])
pool_test = Pool(X_test, cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY'])


# In[9]:


#Объявление CatBoostRegressor и обучение

#model = CatBoostRegressor(task_type='GPU')
model = CatBoostRegressor(task_type='CPU')
model.fit(pool_train)


# In[10]:


#Получение ответов модели на тестовой выборке в локальном тестировании 

y_pred = model.predict(pool_test)


# In[13]:


#На локальном тестировании модель выдаёт такой результат

print("Значение метрики R2 на test: ", r2_score(y_test, y_pred))


# In[15]:


#Формируем sample_solution. В обучении используется весь train, ответы получаем на test

pool_train_solution = Pool(X, y, cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY'])
pool_test_solution = Pool(test, cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY'])

#model_solution = CatBoostRegressor(task_type='GPU')
model_solution = CatBoostRegressor()
model_solution.fit(pool_train_solution)


# In[19]:


#Получение ответов

y_pred_solution = model.predict(pool_test_solution)


# In[20]:


#Вот так они выглядят

y_pred_solution.astype(int)


# In[21]:


#Формируем sample_solution для отправки на платформу

test['PATIENT_ID_COUNT'] = y_pred_solution.astype(int)


# In[22]:


#Сохраняем в csv файл
 
test.to_csv('sample_solution.csv', sep=';', index=None)

