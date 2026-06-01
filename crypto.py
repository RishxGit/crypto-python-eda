import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#---------------GLOBAL SEABORN THEME SETTINGS---------------
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

#---------------DATA INGESTION---------------
df = pd.read_csv('crypto_top1000_dataset.csv')

'''
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
'''

#---------------DATA CLEANING & PREPROCESSING---------------
# Dropping raw unneeded identification metrics
df1 = df.drop(columns=['image', 'last_updated', 'id']).copy()

# Feature Engineering: Creating logical boolean gate for supply analysis
df1['has_max_supply'] = df1['max_supply'].notna()

# Converting text strings to real universal datetime objects
df1['path_date'] = pd.to_datetime(df1["ath_date"], utc=True)
df1['atl_date'] = pd.to_datetime(df1["atl_date"], utc=True)

# Patching column clipping typo to match correct schema boundaries
df1.rename(columns={'supply_utilizatio': 'supply_utilization'}, inplace=True, errors='ignore')


#---------------UNIVARIATE SEGMENTATION FEATURE ENGINEERING---------------
# Categorizing the 1,000 continuous ranks into four separate operational segments
conditions = [
    (df1['market_cap_rank'] <= 250),
    (df1['market_cap_rank'] > 250) & (df1['market_cap_rank'] <= 500),
    (df1['market_cap_rank'] > 500) & (df1['market_cap_rank'] <= 750),
    (df1['market_cap_rank'] > 750)
]
choices = ['Large-Cap', 'Mid-Cap', 'Small-Cap', 'Micro-Cap']
df1['market_tier'] = np.select(conditions, choices, default='Unknown')

# Creating specialized architectural table to handle 1Y data null gaps natively
df_yearly = df1[df1['price_change_percentage_1y'].notna()].copy()


#---------------BIVARIATE VELOCITY FEATURE ENGINEERING---------------
# Calculating the continuous capital transaction velocity profile column
df1['volume_to_marketcap'] = df1['total_volume'] / df1['market_cap']


#---------------MULTIVARIATE TIMEFRAME PERFORMANCE CLASSIFICATION---------------
# Discretizing continuous long-term changes into logical structural state categories
performance = [
    df_yearly['price_change_percentage_1y'] > 20,
    (df_yearly['price_change_percentage_1y'] >= -20) & (df_yearly['price_change_percentage_1y'] <= 20),
    df_yearly['price_change_percentage_1y'] < -20
]
category = ["BULLISH", "FLAT/SIDEWAYS", "BEARISH"]
df_yearly["perf_category"] = np.select(performance, category, default="Unknown")


#---------------------------- AVERAGE PRICE BY MARKET TIER-------------------------
'''
print(df1.groupby('market_tier')['current_price'].mean())
'''


#----------------------- HIGHEST VOLUME TO MARKET CAP RATIO--------------------------
'''
print(df1.sort_values('volume_to_marketcap', ascending=False)[['symbol', 'volume_to_marketcap']].head(10))
'''


#----------------------- PERCENTAGE PROFITABLE OVER 1 YEAR---------------------------
'''
count = df_yearly[df_yearly['price_change_percentage_1y'] > 0]
profitable_count = len(count)
profitable_1yr = (profitable_count / len(df_yearly)) * 100
print(f"Percentage of profitable coins over 1 year: {profitable_1yr:.2f}%")
'''


#-------------------------MARKET TIER BEST YEARLY RETURN----------------------------
'''
print(df_yearly.groupby('market_tier')['price_change_percentage_1y'].mean())
'''


#---------------------------WITHIN 10% OF ALL-TIME HIGH (ATH)-----------------------------
'''
ten_of_alt = df1[df1['ath_change_percentage'] > -10]
print(ten_of_alt[['symbol', 'ath_change_percentage']].head(10))
print(f'Total coins within 10% of ATH: {len(ten_of_alt)}')
'''


#------------------------- MORE THAN 90% CIRCULATING SUPPLY--------------------------------
'''
df_mature_supply = df1[df1['supply_utilization'] > 90]
print(df_mature_supply[['symbol', 'supply_utilization']].head(10))
print(f'Total coins with >90% supply in circulation: {len(df_mature_supply)}')
'''


#-------------------------CORRELATION BETWEEN MARKET CAP AND VOLUME--------------------------
'''
print(df1[['market_cap', 'total_volume']].corr())
'''


#-------------------------UP ACROSS ALL TIMEFRAMES SIMULTANEOUSLY---------------------------
'''
alltime_up = df1[
    (df1['price_change_percentage_1h'] > 0) & 
    (df1['price_change_percentage_24h'] > 0) & 
    (df1['price_change_percentage_7d'] > 0) & 
    (df1['price_change_percentage_30d'] > 0) & 
    (df1['price_change_percentage_1y'] > 0)
]
print(alltime_up[['symbol', 'price_change_percentage_24h', 'price_change_percentage_1y']].head(5))
print(f'Total coins up across all timeframes: {len(alltime_up)}')
'''


#------------------------AVERAGE MARKET CAP BY PERFORMANCE CATEGORY---------------------------
'''
print(df_yearly.groupby('perf_category')['market_cap'].mean())
'''


#------------------------STATISTICAL PRICE OUTLIERS (3-SIGMA SYSTEM)---------------------------
'''
mean_price = df1['current_price'].mean()
std_price = df1['current_price'].std()
outlier_cutoff = mean_price + (3 * std_price)
condition = df1['current_price'] > outlier_cutoff
print(df1[condition][['symbol', 'current_price']])
print(f"Total statistical price outliers: {len(df1[condition])}")
'''


#---------------VISUALIZATION 1 — BIMODAL RETURN DISTRIBUTION HISTOGRAM---------------
'''
sns.histplot(data=df_yearly, x='price_change_percentage_1y', kde=True, bins=np.arange(-100, 501, 10), color='royalblue')
plt.xlim(-100, 500)
plt.title('Distribution of 1-Year Crypto Price Changes')
plt.xlabel('Price Change Percentage (1-Year)')
plt.ylabel('Number of Coins')
plt.show()
'''


#---------------VISUALIZATION 2 — INTER-FEATURE CORRELATION MATRIX HEATMAP---------------
'''
df_num = df1.select_dtypes(include=[np.number])
corr_matrix = df_num.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, linewidths=0.5)
plt.title('Global Feature Correlation Matrix (Collinearity Assessment)')
plt.show()
'''


#---------------VISUALIZATION 3 — AVERAGED RETURNS METRIC BY RANK CLUSTER---------------
'''
sns.barplot(data=df_yearly, x='market_tier', y='price_change_percentage_1y', errorbar=None, palette='Blues_r')
plt.title('Average 1-Year Price Return by Market Capitalization Tier')
plt.xlabel('Market Tier')
plt.ylabel('Average Price Change % (1y)')
plt.show()
'''


#---------------VISUALIZATION 4 — INDEPENDENT SCALE FLUID SCATTER CLOUD---------------
'''
sns.scatterplot(data=df1, x='market_cap', y='total_volume', hue='market_tier', palette='viridis', alpha=0.7)
plt.xscale('log')
plt.yscale('log')
plt.title('Asset Scale Evaluation: Market Capitalization vs. 24h Trading Volume')
plt.xlabel('Market Capitalization (Log Scale - USD)')
plt.ylabel('Total 24h Trading Volume (Log Scale - USD)')
plt.legend(title='Market Tier')
plt.show()
'''


#---------------VISUALIZATION 5 — ESCALATING MULTI-TIMEFRAME VOLATILITY VARIANCE---------------
'''
timeframes = ['price_change_percentage_1h', 'price_change_percentage_24h', 'price_change_percentage_7d', 'price_change_percentage_30d']
sns.boxplot(data=df1[timeframes], palette='Set2')
plt.ylim(-50, 100)
plt.title('Volatility Profile Escalation Across Multiple Aggregation Windows')
plt.xlabel('Trading Timeframe Window')
plt.ylabel('Price Variance Percentage')
plt.xticks(ticks=[0, 1, 2, 3], labels=['1h Window', '24h Window', '7d Window', '30d Window'])
plt.show()
'''


#---------------VISUALIZATION 6 — TOP 20 VALUATION STRUCTURAL OVERVIEW---------------
'''
df_top20 = df1.sort_values('market_cap', ascending=False).head(20)
sns.barplot(data=df_top20, x='market_cap', y='symbol', errorbar=None, color='lightpink')
plt.title('Macro Asset Dominance Profile: Top 20 Cryptocurrencies by Valuation')
plt.xlabel('Market Capitalization (USD - Hundreds of Billions)')
plt.ylabel('Token Symbol Representation')
plt.tight_layout()
plt.show()
'''
