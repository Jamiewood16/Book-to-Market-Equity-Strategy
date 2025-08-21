import pandas as pd
import numpy as np
import sqlite3
import statsmodels.api as sm
from scipy import stats  
import matplotlib.pyplot as plt
import pandas_datareader as pdr


#Fama-Macbeth Regression - question b
# Connected to SQLite database and load data
conn = sqlite3.connect("data_assignment.sqlite")
data = pd.read_sql("SELECT * FROM data_all", conn, parse_dates=["date"])

# create log-transformed predictors
data_clean = data.dropna(subset=["ret_excess", "bm", "me", "gp", "inv"]).copy()
data_clean["log_me"] = np.log(data_clean["me"])
data_clean["log_bm"] = np.log(data_clean["bm"])  

# Define Predictors and Target
predictors = ["log_me", "log_bm", "gp", "inv"]  
target = "ret_excess"

# Fama-MacBeth Regression
params_list = []

# Loop  each month and run cross-sectional regression
for date, month_data in data_clean.groupby("date"):
    X = sm.add_constant(month_data[predictors])  
    y = month_data[target]
    
    # Run OLS 
    model = sm.OLS(y, X, missing="drop").fit()
    

    params = model.params.to_dict()
    params["date"] = date
    params_list.append(params)


params_df = pd.DataFrame(params_list).set_index("date")

# Compute Time-Series Averages and Significance 
fm_results = pd.DataFrame({
    "Coefficient": params_df.mean(),
    "Std. Error": params_df.std() / np.sqrt(len(params_df)),  
    "T-stat": params_df.apply(lambda x: x.mean() / (x.std() / np.sqrt(len(x)))),
    "P-value": params_df.apply(lambda x: stats.ttest_1samp(x, 0).pvalue)
})


print("\nFama-MacBeth Regression Results:")
print(fm_results.round(4))

# Plot Coefficient Stability Over Time
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
params_df["log_bm"].rolling(window=24).mean().plot(  # 2-year rolling average
    title="Rolling Average of log_bm Coefficient Over Time"
)
plt.axhline(fm_results.loc["log_bm", "Coefficient"], color="red", linestyle="--")
plt.xlabel("Date")
plt.ylabel("Coefficient")
plt.show()






#question c - quintile portolfio, construct long-short strategy



#  Quintile Portfolios 

data_clean["quintile"] = data_clean.groupby("date")["log_bm"].transform(
    lambda x: pd.qcut(x, q=5, labels=[1, 2, 3, 4, 5])
)

# Calculate equal-weighted returns 
portfolio_returns = data_clean.groupby(["date", "quintile"])["ret_excess"].mean().unstack()

# Long-Short Strategy
portfolio_returns["long_short"] = portfolio_returns[5] - portfolio_returns[1]  

# Results
results = pd.DataFrame({
    "Average Return": portfolio_returns.mean(),
    "T-stat": portfolio_returns.apply(lambda x: stats.ttest_1samp(x, 0).statistic),
    "P-value": portfolio_returns.apply(lambda x: stats.ttest_1samp(x, 0).pvalue)
})

print("=== Quintile Portfolio Results ===")
print(results.round(4))


(1 + portfolio_returns).cumprod().plot(figsize=(10, 5))
plt.title("Cumulative Returns of Quintile Portfolios")
plt.show()







#Question D and E


# Load and prepare factor data with proper column names
try:
    # Download data and rename columns
    ff3 = pdr.get_data_famafrench("F-F_Research_Data_Factors", start="1900")[0]
    ff3 = ff3.rename(columns={
        'Mkt-RF': 'mkt',
        'SMB': 'smb',
        'HML': 'hml'
    })
    ff3.index = ff3.index.to_timestamp()

    ff5 = pdr.get_data_famafrench("F-F_Research_Data_5_Factors_2x3", start="1900")[0]
    ff5 = ff5.rename(columns={
        'Mkt-RF': 'mkt',
        'SMB': 'smb', 
        'HML': 'hml',
        'RMW': 'rmw',
        'CMA': 'cma'
    })
    ff5.index = ff5.index.to_timestamp()

    # Merge with portfolio returns
    factor_data = (
        portfolio_returns[['long_short']]
        .merge(ff3.drop('RF', axis=1).add_suffix('_3f'), 
               left_index=True, right_index=True, how='left')
        .merge(ff5.drop('RF', axis=1).add_suffix('_5f'), 
               left_index=True, right_index=True, how='left')
    )

    
    required_cols = ['mkt_3f', 'smb_3f', 'hml_3f', 'mkt_5f', 'smb_5f', 'hml_5f', 'rmw_5f', 'cma_5f']
    
except Exception as e:
    print(f"Error loading factor data: {e}")
    
    raise

# Run factor model regressions
def run_factor_model(y, X, model_name=""):
    """Run OLS regression with HAC standard errors"""
    X = sm.add_constant(X)
    return sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 6})

try:
    print("\n=== Fama-French 3-Factor Model ===")
    X_ff3 = factor_data[['mkt_3f', 'smb_3f', 'hml_3f']]
    model_ff3 = run_factor_model(factor_data['long_short'], X_ff3)
    print(model_ff3.summary())

    print("\n=== Fama-French 5-Factor Model ===")
    X_ff5 = factor_data[['mkt_5f', 'smb_5f', 'hml_5f', 'rmw_5f', 'cma_5f']]
    model_ff5 = run_factor_model(factor_data['long_short'], X_ff5)
    print(model_ff5.summary())

except Exception as e:
    print(f"\nRegression error: {e}")
    print("\nAvailable columns in factor_data:")
    print(factor_data.columns.tolist())
    raise


plt.figure(figsize=(14, 10))

# Cumulative Returns Plot
plt.subplot(2, 1, 1)
(1 + factor_data['long_short']).cumprod().plot(
    linewidth=2, label='Actual Strategy', color='navy')
(1 + model_ff3.predict(sm.add_constant(X_ff3))).cumprod().plot(
    linewidth=1.5, linestyle='--', label='FF3 Model', color='maroon')
(1 + model_ff5.predict(sm.add_constant(X_ff5))).cumprod().plot(
    linewidth=1.5, linestyle='-.', label='FF5 Model', color='darkgreen')
plt.title('Strategy vs. Factor Model Performance', fontsize=14)
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)

# Rolling Returns Plot
plt.subplot(2, 1, 2)
factor_data['long_short'].rolling(12).mean().plot(
    linewidth=2, label='12-Month Rolling Return', color='purple')
plt.axhline(0, color='black', linestyle='--')
plt.title('Strategy Consistency', fontsize=14)
plt.ylabel('Annualized Return')
plt.xlabel('Date')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()




# Value-Weighted Portfolio Analysis 

# Create value-weighted quintiles using log_bm
data_clean['vw_quintile'] = data_clean.groupby('date')['log_bm'].transform(
    lambda x: pd.qcut(x, q=5, labels=[1, 2, 3, 4, 5])
)

# 2. Calculate value-weighted returns 
vw_portfolios = (
    data_clean.groupby(['date', 'vw_quintile'])
    .apply(lambda x: np.average(x['ret_excess'], weights=x['me']))
    .unstack()
)

# 3. Long-short strategy 
vw_portfolios['long_short_vw'] = vw_portfolios[5] - vw_portfolios[1]

# 4. Performance stats
vw_results = pd.DataFrame({
    'VW Avg Return': vw_portfolios.mean(),
    'VW t-stat': vw_portfolios.apply(lambda x: stats.ttest_1samp(x, 0).statistic),
    'VW p-value': vw_portfolios.apply(lambda x: stats.ttest_1samp(x, 0).pvalue)
})

print("\n=== Value-Weighted Quintile Results ===")
print(vw_results.round(4))

# 5. Factor attribution for value-weighted strategy
vw_factor_data = (
    vw_portfolios[['long_short_vw']]
    .merge(ff3.drop('RF', axis=1).add_suffix('_3f'), left_index=True, right_index=True)
    .merge(ff5.drop('RF', axis=1).add_suffix('_5f'), left_index=True, right_index=True)
)

# FF3 regression
X_vw_ff3 = sm.add_constant(vw_factor_data[['mkt_3f', 'smb_3f', 'hml_3f']])
model_vw_ff3 = sm.OLS(vw_factor_data['long_short_vw'], X_vw_ff3).fit(cov_type='HAC', cov_kwds={'maxlags': 6})

# FF5 regression
X_vw_ff5 = sm.add_constant(vw_factor_data[['mkt_5f', 'smb_5f', 'hml_5f', 'rmw_5f', 'cma_5f']])
model_vw_ff5 = sm.OLS(vw_factor_data['long_short_vw'], X_vw_ff5).fit(cov_type='HAC', cov_kwds={'maxlags': 6})

print("\n=== Value-Weighted FF3 Model ===")
print(model_vw_ff3.summary())
print("\n=== Value-Weighted FF5 Model ===")
print(model_vw_ff5.summary())

# 6. Cumulative returns plot comparison
plt.figure(figsize=(12, 6))
(1 + vw_portfolios['long_short_vw']).cumprod().plot(
    label='Value-Weighted', color='blue', linewidth=2)
(1 + portfolio_returns['long_short']).cumprod().plot(  # From your equal-weighted analysis
    label='Equal-Weighted', color='red', linestyle='--', linewidth=2)
plt.title('Value-Weighted vs Equal-Weighted Strategy Performance')
plt.ylabel('Cumulative Return')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()





# Dependent Double Sorting Analysis


second_char = 'inv'  
main_char = 'log_bm'  

# Create 25 portfolios 
data_clean['first_sort'] = data_clean.groupby('date', observed=False)[second_char].transform(
    lambda x: pd.qcut(x, q=5, labels=[1, 2, 3, 4, 5])
)

# Dependent sort on main characteristic 
data_clean['double_sort'] = data_clean.groupby(['date', 'first_sort'], observed=False)[main_char].transform(
    lambda x: pd.qcut(x, q=5, labels=[1, 2, 3, 4, 5])
)

# 2. Calculate portfolio statistics
double_portfolios = (
    data_clean.groupby(['date', 'first_sort', 'double_sort'], observed=False)
    .agg({
        'ret_excess': 'mean',           
        main_char: 'mean',              
        second_char: 'mean',            
        'me': 'count'                   
    })
    .unstack(['first_sort', 'double_sort'])  
)


avg_returns = double_portfolios['ret_excess'].mean(axis=0).unstack()
avg_main_char = double_portfolios[main_char].mean(axis=0).unstack()
avg_second_char = double_portfolios[second_char].mean(axis=0).unstack()

print("\n=== Average Monthly Returns ===")
print(avg_returns.round(4))
print("\n=== Average", main_char, "===")
print(avg_main_char.round(4))
print("\n=== Average", second_char, "===")
print(avg_second_char.round(4))

# Create enhanced long-short portfolio

top_main = double_portfolios['ret_excess'].xs(5, level='double_sort', axis=1)
bottom_main = double_portfolios['ret_excess'].xs(1, level='double_sort', axis=1)

# Calculate time series of enhanced strategy
enhanced_returns = top_main.mean(axis=1) - bottom_main.mean(axis=1)

# Performance stats
enhanced_results = pd.Series({
    'Avg Return': enhanced_returns.mean(),
    't-stat': stats.ttest_1samp(enhanced_returns, 0).statistic,
    'p-value': stats.ttest_1samp(enhanced_returns, 0).pvalue
})

print("\n=== Enhanced Long-Short Results ===")
print(enhanced_results.to_frame().T.round(4))

# 5. Compare with original univariate sort
comparison = pd.DataFrame({
    'Original': [portfolio_returns['long_short'].mean(), 
                stats.ttest_1samp(portfolio_returns['long_short'], 0).statistic],
    'Enhanced': [enhanced_returns.mean(),
                stats.ttest_1samp(enhanced_returns, 0).statistic]
}, index=['Avg Return', 't-stat'])

print("\n=== Strategy Comparison ===")
print(comparison.round(4))

# plot
plt.figure(figsize=(12, 6))
(1 + portfolio_returns['long_short']).cumprod().plot(label='Original Univariate Sort')
(1 + enhanced_returns).cumprod().plot(label='Enhanced Double Sort')
plt.title('Comparison of Long-Short Strategies')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()