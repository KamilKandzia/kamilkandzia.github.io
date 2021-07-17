---
layout: page
title: Prediction model of sales in alcohol stores by using the Prophet
permalink: /prophet/
parent: Python
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Build a prediction model by the Prophet

This notebook aimed to build a prediction model to analyze the future income of the stores based on 3 files. 

*   The first CSV contains weekly sales volume for alcohol stores divided by types of the alcohol
*   The second has information of distances in km between each pair of stores that are within 5 kilometers of each other
*   The last file contains the number of high education institutions that are
within a 5 km radius from each store in scope.

**Summary in PDF**

<embed src="{{site.url}}/assets/images/prophet_files/alco_stores_prophet.pdf" width="100%" height="600px" 
 type="application/pdf">

**Business case**

What should be taken into account when granting credit?

* Research of the company's revenue/incomes
* Creation of future revenue/incomes model based on financial data
* Analysis of the level of risk to competitors

**Prophet – forecasting procedure**

Why is this a good choice for forecast analysis?

* Flexibility
* The measurements do not need to be regularly spaced
* Facebook uses it for producing reliable forecasts for planning and goal setting


**Prophet**

*Predicted value*: total sales of alcohol per week.

*Additional regressors*: sales of each type of alcohol per week.

Due to the lack of a given country, it is not possible to add additional 
regressors such as days off.

**How to analyse the competitiveness of a given store?**

* Find stores with similar features
* Check how many competitors are in the same group for each store
* Compare the competition in the same group with the median/medium for the market

**Recommendations**

* Long (+4 years) and short (1-2 year) term analysis
* The recommended period to forecast (for store sales) is at least 2 years
* Additional regressors could improve forecasted data
* Clustering is a good approach to finding similar stores

# **Initial stage**


```python
import datetime
import pandas as pd
import numpy as np

from google.colab import drive

drive.mount('/content/gdrive')
```

    Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).
    


```python
!ls "/content/gdrive/My Drive/Prophet"
```

     colab_installed.txt		        sales_of_alcohol_per_store.csv
     details_of_places_close_to_store.csv   store_distances.csv
    'Kamil Kandzia - Prophet'
    


```python
import warnings

warnings.filterwarnings('ignore')
```

# **sales_of_alcohol_per_store**



```python
df_store_sales = pd.read_csv('gdrive/My Drive/Prophet/sales_of_alcohol_per_store.csv', sep=';')
```


```python
df_store_sales.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>week</th>
      <th>store_id</th>
      <th>Vodka</th>
      <th>Tequila</th>
      <th>Whiskey</th>
      <th>Other</th>
      <th>Gin</th>
      <th>Brandy</th>
      <th>Rum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012</td>
      <td>1</td>
      <td>0</td>
      <td>1824.96</td>
      <td>0.00</td>
      <td>3645.81</td>
      <td>5143.30</td>
      <td>0.00</td>
      <td>169.86</td>
      <td>1507.50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012</td>
      <td>1</td>
      <td>1</td>
      <td>279.01</td>
      <td>0.00</td>
      <td>251.54</td>
      <td>28.35</td>
      <td>0.00</td>
      <td>31.28</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012</td>
      <td>1</td>
      <td>2</td>
      <td>441.60</td>
      <td>0.00</td>
      <td>1195.08</td>
      <td>2434.44</td>
      <td>231.12</td>
      <td>162.96</td>
      <td>395.76</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012</td>
      <td>1</td>
      <td>7</td>
      <td>18432.50</td>
      <td>406.68</td>
      <td>24344.53</td>
      <td>19546.09</td>
      <td>3527.14</td>
      <td>2113.41</td>
      <td>7246.16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>1</td>
      <td>8</td>
      <td>3584.56</td>
      <td>0.00</td>
      <td>7175.44</td>
      <td>5970.78</td>
      <td>401.81</td>
      <td>594.37</td>
      <td>1043.21</td>
    </tr>
  </tbody>
</table>
</div>



The first step to analyse the data is to explore it. 


```python
summary = df_store_sales.describe()
summary = summary.transpose()
summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>year</th>
      <td>248355.0</td>
      <td>2014.500469</td>
      <td>1.699403</td>
      <td>2012.0</td>
      <td>2013.00</td>
      <td>2015.00</td>
      <td>2016.000</td>
      <td>2017.00</td>
    </tr>
    <tr>
      <th>week</th>
      <td>248355.0</td>
      <td>26.151489</td>
      <td>14.700727</td>
      <td>1.0</td>
      <td>13.00</td>
      <td>26.00</td>
      <td>39.000</td>
      <td>53.00</td>
    </tr>
    <tr>
      <th>store_id</th>
      <td>248355.0</td>
      <td>660.804695</td>
      <td>451.398204</td>
      <td>0.0</td>
      <td>294.00</td>
      <td>591.00</td>
      <td>993.000</td>
      <td>1881.00</td>
    </tr>
    <tr>
      <th>Vodka</th>
      <td>248355.0</td>
      <td>902.706373</td>
      <td>2243.988801</td>
      <td>0.0</td>
      <td>195.12</td>
      <td>411.90</td>
      <td>841.325</td>
      <td>111469.14</td>
    </tr>
    <tr>
      <th>Tequila</th>
      <td>248355.0</td>
      <td>26.659274</td>
      <td>195.084990</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>21339.60</td>
    </tr>
    <tr>
      <th>Whiskey</th>
      <td>248355.0</td>
      <td>1481.208514</td>
      <td>3371.440033</td>
      <td>0.0</td>
      <td>362.50</td>
      <td>681.16</td>
      <td>1409.120</td>
      <td>212765.03</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>248355.0</td>
      <td>631.865501</td>
      <td>1999.413710</td>
      <td>0.0</td>
      <td>29.34</td>
      <td>185.85</td>
      <td>527.200</td>
      <td>113270.40</td>
    </tr>
    <tr>
      <th>Gin</th>
      <td>248355.0</td>
      <td>153.740689</td>
      <td>470.861189</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>21.90</td>
      <td>137.280</td>
      <td>29961.72</td>
    </tr>
    <tr>
      <th>Brandy</th>
      <td>248355.0</td>
      <td>199.188675</td>
      <td>500.308544</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>29.24</td>
      <td>194.160</td>
      <td>65346.29</td>
    </tr>
    <tr>
      <th>Rum</th>
      <td>248355.0</td>
      <td>434.866006</td>
      <td>1444.561461</td>
      <td>0.0</td>
      <td>42.89</td>
      <td>192.13</td>
      <td>423.295</td>
      <td>255128.24</td>
    </tr>
  </tbody>
</table>
</div>




The data are from 2012-2017. They refer to 1882 shops located in a certain country. But is the data complete? The answer may be given by the mean of store_id value, which is not close to the expected value of 940.5.

There are few missing data imputation techniques (mean/median, most frequent/zero value, etc.), but if the data is not missing at random, any standard calculations give the wrong answer.

Change week and year to YYYY-MM-DD format and calculate Total value per week and store_id




```python
def change_into_date(date_cell: str):
    return datetime.datetime.strptime(date_cell + '-7', '%G-W%V-%u')

df_store_sales['period']= df_store_sales["year"].astype(str)+"-W" + df_store_sales["week"].astype(str)
df_store_sales['period'] = df_store_sales.apply(lambda row: change_into_date(row['period']), axis=1)

df_store_sales.loc[:,'Total'] = (
    df_store_sales[['Vodka', 'Tequila', 'Whiskey', 'Other', 'Gin', 'Brandy', 'Rum']].sum(axis=1))
```

*Additional investigation*

Usability of the data


```python
df_store_sales_by_year_store_id=(
    df_store_sales
    .groupby(['store_id', 'year'], as_index=False, sort=True)['week']
    .count())
```


```python
len(df_store_sales_by_year_store_id.store_id.unique())
```




    1711




```python
store_id_over=(
    df_store_sales_by_year_store_id
    .store_id.loc[df_store_sales_by_year_store_id['week'] > 12]
    .unique())

# get all store_id with sales report over 12 per year
len(store_id_over)
```




    1469



Only 1469 stores have over 12 reports of sales by at least one year


```python
df_grouped_second=df_store_sales.loc[df_store_sales['store_id'].isin(store_id_over)]
df_grouped_second.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>week</th>
      <th>store_id</th>
      <th>Vodka</th>
      <th>Tequila</th>
      <th>Whiskey</th>
      <th>Other</th>
      <th>Gin</th>
      <th>Brandy</th>
      <th>Rum</th>
      <th>period</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012</td>
      <td>1</td>
      <td>0</td>
      <td>1824.96</td>
      <td>0.00</td>
      <td>3645.81</td>
      <td>5143.30</td>
      <td>0.00</td>
      <td>169.86</td>
      <td>1507.50</td>
      <td>2012-01-08</td>
      <td>12291.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012</td>
      <td>1</td>
      <td>1</td>
      <td>279.01</td>
      <td>0.00</td>
      <td>251.54</td>
      <td>28.35</td>
      <td>0.00</td>
      <td>31.28</td>
      <td>0.00</td>
      <td>2012-01-08</td>
      <td>590.18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012</td>
      <td>1</td>
      <td>2</td>
      <td>441.60</td>
      <td>0.00</td>
      <td>1195.08</td>
      <td>2434.44</td>
      <td>231.12</td>
      <td>162.96</td>
      <td>395.76</td>
      <td>2012-01-08</td>
      <td>4860.96</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012</td>
      <td>1</td>
      <td>7</td>
      <td>18432.50</td>
      <td>406.68</td>
      <td>24344.53</td>
      <td>19546.09</td>
      <td>3527.14</td>
      <td>2113.41</td>
      <td>7246.16</td>
      <td>2012-01-08</td>
      <td>75616.51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>1</td>
      <td>8</td>
      <td>3584.56</td>
      <td>0.00</td>
      <td>7175.44</td>
      <td>5970.78</td>
      <td>401.81</td>
      <td>594.37</td>
      <td>1043.21</td>
      <td>2012-01-08</td>
      <td>18770.17</td>
    </tr>
  </tbody>
</table>
</div>



# **store_distances**


```python
df_store_distances = pd.read_csv('gdrive/My Drive/Prophet/store_distances.csv', sep=',')
```


```python
df_store_distances.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id_1</th>
      <th>store_id_2</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>80</td>
      <td>4.472470</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>87</td>
      <td>1.621428</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>140</td>
      <td>2.226306</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>174</td>
      <td>2.904311</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>348</td>
      <td>3.788029</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df_store_distances.store_id_1.unique())
```




    1384




```python
len(df_store_distances.store_id_2.unique())
```




    1392



There is a difference between the two columns and depended on values (distance). Let's concatenate df with another (with swapped columns store_id).


```python
results_df=pd.concat([df_store_distances,df_store_distances.rename(columns={'store_id_1': 'store_id_2', 
                                                                            'store_id_2': 'store_id_1'})], 
                     ignore_index=True)
results_df=results_df.drop_duplicates()
```

Count no of stores within 5km distance


```python
df_store_distances_grouped=results_df.groupby(['store_id_1'], as_index=False)['store_id_2'].count()
df_store_distances_grouped.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id_1</th>
      <th>store_id_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>63</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



Calculate  25%, 50%, 75%, and mean of the distance between specific store_id_1


```python
df_store_percentage=(
    results_df
    .groupby(['store_id_1'])
    .describe()
    .stack(level=0)[['25%', '50%', '75%', 'mean']])
```

Delete unnecessary rows and columns


```python
df_store_percentage=df_store_percentage.iloc[::2, :]
df_store_percentage=df_store_percentage.reset_index()

df_store_merged=pd.merge(df_store_percentage, 
                         df_store_distances_grouped, 
                         left_index=True, 
                         right_index=True)

df_store_merged.drop(columns=['level_1', 'store_id_1_y'])
df_store_merged = df_store_merged.rename(columns={'store_id_1_x': 'store_id', 
                                                  'store_id_2': 'total_no_stores_in_5_km'})
```

Fills the zeros when there are no concurrency stores within 5 km.
We need to be aware that some data might be missing (instead of zero values of the total stores within 5 km).


```python
df_store_merged=(
    df_store_merged
    .set_index('store_id')
    .reindex(index = np.arange(0,1882), fill_value=0)
    .reset_index())
```


```python
df_store_merged
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>level_1</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>mean</th>
      <th>store_id_1_y</th>
      <th>total_no_stores_in_5_km</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>distance</td>
      <td>1.502608</td>
      <td>2.219791</td>
      <td>3.260144</td>
      <td>2.420927</td>
      <td>0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>distance</td>
      <td>0.084720</td>
      <td>0.084720</td>
      <td>0.084720</td>
      <td>0.084720</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>distance</td>
      <td>1.616549</td>
      <td>2.441432</td>
      <td>3.637397</td>
      <td>2.663611</td>
      <td>2</td>
      <td>63</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>distance</td>
      <td>0.250866</td>
      <td>0.495019</td>
      <td>0.661759</td>
      <td>0.417607</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>distance</td>
      <td>0.002576</td>
      <td>0.002576</td>
      <td>0.002576</td>
      <td>0.002576</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1877</th>
      <td>1877</td>
      <td>distance</td>
      <td>1.794602</td>
      <td>2.619740</td>
      <td>3.507170</td>
      <td>2.687761</td>
      <td>1877</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1878</th>
      <td>1878</td>
      <td>distance</td>
      <td>4.701323</td>
      <td>4.754282</td>
      <td>4.809905</td>
      <td>4.312814</td>
      <td>1878</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1879</th>
      <td>1879</td>
      <td>distance</td>
      <td>3.556206</td>
      <td>4.762775</td>
      <td>4.800873</td>
      <td>3.594304</td>
      <td>1879</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1880</th>
      <td>1880</td>
      <td>distance</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1880</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1881</th>
      <td>1881</td>
      <td>distance</td>
      <td>3.797875</td>
      <td>3.849405</td>
      <td>4.614612</td>
      <td>3.776452</td>
      <td>1881</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>1882 rows × 8 columns</p>
</div>



# **details_of_places_close_to_store**


```python
df_store_gdata = pd.read_csv('gdrive/My Drive/Prophet/details_of_places_close_to_store.csv', sep=',')
```


```python
df_store_gdata
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>university or college</th>
      <th>foodstores or supermarkets or gorceries</th>
      <th>restaurant</th>
      <th>churches</th>
      <th>gym</th>
      <th>stadium</th>
      <th>store_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1856</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>15</td>
      <td>3</td>
      <td>0</td>
      <td>1857</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>1858</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1859</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1860</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1877</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1851</td>
    </tr>
    <tr>
      <th>1878</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1852</td>
    </tr>
    <tr>
      <th>1879</th>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1853</td>
    </tr>
    <tr>
      <th>1880</th>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1854</td>
    </tr>
    <tr>
      <th>1881</th>
      <td>0</td>
      <td>2</td>
      <td>19</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>1855</td>
    </tr>
  </tbody>
</table>
<p>1882 rows × 7 columns</p>
</div>




```python
# from sklearn.preprocessing import normalize
# df_norm=df_store_gdata.loc[:, df_store_gdata.columns != 'store_id']

# data_scaled = normalize(df_store_gdata.loc[:, df_store_gdata.columns != 'store_id'])
# data_scaled=df_store_gdata
# data_scaled = pd.DataFrame(data_scaled, columns=df_norm.columns)
# data_scaled = pd.concat([data_scaled, pd.DataFrame(df_store_gdata['store_id'])], axis=1)
```

Data normalization was not applied because we are looking for stores with similar parameters. Clustering to the same category of stores, which have one restaurant and one church, as well as 20 restaurants and 20 churches, is not a good approach. We will have stores with different incomes and the concurrency level within 5 km.


```python
df_store_gdata.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>university or college</th>
      <th>foodstores or supermarkets or gorceries</th>
      <th>restaurant</th>
      <th>churches</th>
      <th>gym</th>
      <th>stadium</th>
      <th>store_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1877</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1851</td>
    </tr>
    <tr>
      <th>1878</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1852</td>
    </tr>
    <tr>
      <th>1879</th>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1853</td>
    </tr>
    <tr>
      <th>1880</th>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1854</td>
    </tr>
    <tr>
      <th>1881</th>
      <td>0</td>
      <td>2</td>
      <td>19</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>1855</td>
    </tr>
  </tbody>
</table>
</div>




```python
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
```


```python
Z = linkage(df_store_gdata.loc[:, df_store_gdata.columns != 'store_id'], 'ward')

# c, coph_dists = cophenet(Z, pdist(df_store_gdata))
# c

max_d = 5
clusters = fcluster(Z, max_d, criterion='distance')

def add_clusters_to_frame(frame, clusters):
    frame = pd.DataFrame(data=frame)
    frame_labelled = pd.concat([frame, pd.DataFrame(clusters)], axis=1)
    return(frame_labelled)

df_store_gdata = add_clusters_to_frame(df_store_gdata, clusters)
df_store_gdata.columns = ['university or college', 'foodstores or supermarkets or gorceries', 'restaurant', 'churches', 'gym', 'stadium', 'store_id','cluster']
```


```python
c, coph_dists = cophenet(Z, pdist(df_store_gdata))
print(c)
```

    0.019288638247969937
    


```python
# import scipy.cluster.hierarchy as shc
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 7))  
# plt.title("Dendrograms") 
# dend=shc.dendrogram(Z, labels=list(df_store_gdata.loc[:,'store_id']))
```

Show similar stores (by the cluster). e.g. store_id=0


```python
df_clustered = df_store_gdata.loc[df_store_gdata['store_id'] == 0]
df_clustered_by_chosen_store= df_store_gdata.loc[df_store_gdata['cluster'] == int(df_clustered.cluster)]
df_clustered_by_chosen_store.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>university or college</th>
      <th>foodstores or supermarkets or gorceries</th>
      <th>restaurant</th>
      <th>churches</th>
      <th>gym</th>
      <th>stadium</th>
      <th>store_id</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26</th>
      <td>5</td>
      <td>7</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>3</td>
      <td>0</td>
      <td>135</td>
    </tr>
    <tr>
      <th>65</th>
      <td>5</td>
      <td>7</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>3</td>
      <td>39</td>
      <td>135</td>
    </tr>
    <tr>
      <th>66</th>
      <td>5</td>
      <td>7</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>3</td>
      <td>40</td>
      <td>135</td>
    </tr>
    <tr>
      <th>210</th>
      <td>4</td>
      <td>6</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>3</td>
      <td>184</td>
      <td>135</td>
    </tr>
    <tr>
      <th>301</th>
      <td>4</td>
      <td>6</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>3</td>
      <td>275</td>
      <td>135</td>
    </tr>
  </tbody>
</table>
</div>



# **Prophet model**


Possibility to analyze seasonality, as well as additional regression factors such as free Sundays from shopping, bank holidays or paydays among people who work or live near the store


```python
import fbprophet
```

Forecasting Growth (by default, Prophet uses a linear model for its forecast)


```python
df_store_selected=df_store_sales.loc[df_store_sales['store_id'] == 0]
df_prophet=df_store_selected[['period', 'Total']]

df_prophet = df_prophet.rename(columns={'period': 'ds', 'Total': 'y'})

m = fbprophet.Prophet()
m.fit(df_prophet)
future = m.make_future_dataframe(44, freq='W') #MS 52
forecast = m.predict(future)
fig = m.plot(forecast, ylabel='Euro', xlabel='date')
```

    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    


![png]({{site.url}}/assets/images/prophet_files/output_52_1.png)


There is a decreasing trend in total alcohol sales for the store. To implement the solution, a model with logistic regression was used, where the lower value of saturation is 0 and the upper one is the maximum historical value.


```python
del m, future
```


```python
df_store_selected= df_store_sales.loc[df_store_sales['store_id'] == 0]
df_prophet=df_store_selected[['period', 'Total']]

df_prophet = df_prophet.rename(columns={'period': 'ds', 
                                        'Total': 'y'})
df_prophet['cap'] = df_prophet.y.max()
df_prophet['floor'] = 0

m = fbprophet.Prophet(growth='logistic', weekly_seasonality=True)
m.fit(df_prophet)

future = m.make_future_dataframe(44, freq='W') #MS
future['cap'] = df_prophet.y.max()
future['floor'] = 0

forecast = m.predict(future)
fig = m.plot(forecast, ylabel='Euro', xlabel='date')
```

    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    


![png]({{site.url}}/assets/images/prophet_files/output_55_1.png)


The previous model was using weekly_seasonality=True. Now check the next model by disabling it.


```python
del m, future
```


```python
df_store_selected= df_store_sales.loc[df_store_sales['store_id'] == 0]
df_prophet=df_store_selected[['period', 'Total']]

df_prophet = df_prophet.rename(columns={'period': 'ds', 
                                        'Total': 'y'})
df_prophet['cap'] = df_prophet.y.max()
df_prophet['floor'] = 0

m = fbprophet.Prophet(growth='logistic')
m.fit(df_prophet)

future = m.make_future_dataframe(44, freq='W') #MS
future['cap'] = df_prophet.y.max()
future['floor'] = 0

forecast = m.predict(future)
fig = m.plot(forecast, ylabel='Euro', xlabel='date')
```

    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    


![png]({{site.url}}/assets/images/prophet_files/output_58_1.png)


For standardization: The future dataframe should have original, not-standardized values, and the forecasted values should also not be standardized. For predictions, Prophet will apply the same standardization offset and scale used in fitting.

https://github.com/facebook/prophet/issues/484

https://github.com/facebook/prophet/issues/1392#issuecomment-602892389

Box-Cox transformation despite the information above


```python
from scipy import stats
```


```python
del m, future
```


```python
df_store_selected= df_store_sales.loc[df_store_sales['store_id'] == 0]
df_prophet=df_store_selected[['period', 'Total']]

df_prophet = df_prophet.rename(columns={'period': 'ds', 
                                        'Total': 'y'})
df_prophet.y=stats.boxcox(df_prophet.y)[0]
df_prophet['cap'] = df_prophet.y.max()
df_prophet['floor'] = 0

m = fbprophet.Prophet(growth='logistic')
m.fit(df_prophet)

future = m.make_future_dataframe(44, freq='W') #MS
future['cap'] = df_prophet.y.max()
future['floor'] = 0

forecast = m.predict(future)
fig = m.plot(forecast, ylabel='Euro', xlabel='date')
```

    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    


![png]({{site.url}}/assets/images/prophet_files/output_63_1.png)


Lambda


```python
df_store_selected= df_store_sales.loc[df_store_sales['store_id'] == 0]
df_prophet=df_store_selected[['period', 'Total']]

df_prophet['floor'] = 0
df_prophet = df_prophet.rename(columns={'period': 'ds', 
                                        'Total': 'y'})

stats.boxcox(df_prophet.y)[1]
```




    0.8974534720944913



Using the Box-Cox transformation, the data is more within the upper and thelower yhat range. However, that in making the reverse transformation, the yhat ranges will be wider than in the previous model. 

https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#additional-regressors

"The extra regressor must be known for both the history and for future dates. It thus must either be something that has known future values (such as nfl_sunday), or something that has separately been forecasted elsewhere. Prophet will also raise an error if the regressor is constant throughout the history, since there is nothing to fit from it.

Extra regressors are put in the linear component of the model, so the underlying model is that the time series depends on the extra regressor as either an additive or multiplicative factor (see the next section for multiplicativity)."

When using additional regressors, it is possible to predict the data (y as search value = Total) based on each type of alcohol.

In the final version, it was decided to select several types of alcohols to check the fit of the model.

# **Final version**

**Prophet model**

**By selecting the chosen store, make sure that the available data will allow sufficient analysis of sales over a certain period** (see *store_sales_per_category* section -> Usability of the data).



**Choose id of the store**


```python
store_id_chosen = 0
```


```python
from fbprophet.diagnostics import performance_metrics, cross_validation
```

**Model based on all date points**


```python
def prophet_predict(store_id_chosen: int, 
                    df_store_sales: pd.DataFrame, 
                    multiplier: int, 
                    growth_fbprophet: str, 
                    *column_types):

  df_store_sales= df_store_sales.loc[df_store_sales['store_id'] == store_id_chosen]
  df_prophet=df_store_sales[np.asarray(column_types)]

  m_full = fbprophet.Prophet(growth=growth_fbprophet)
  df_forecast_regressors = pd.DataFrame([], columns=['ds'])

  #predict additional regressors 
  for element in column_types[2:]:

    m=fbprophet.Prophet(growth=growth_fbprophet)
    df_inside = df_prophet.rename(columns={'period': 'ds', 
                                           element: 'y'})
    df_inside['floor'] = 0
    df_inside['cap'] = df_inside.y.max()
    m.fit(df_inside[['ds', 'y', 'floor', 'cap']])

    future = m.make_future_dataframe(44, freq='W') #MS  52
    future['cap'] = df_inside.y.max()
    future['floor'] = 0
    forecast = m.predict(future)

    #merge values into df
    df_forecast_regressors = df_forecast_regressors.merge(forecast[['ds', 'yhat']], 
                                                          on='ds', 
                                                          how='outer')
                                                          
    df_forecast_regressors = df_forecast_regressors.rename(columns={'yhat': element})
    #add regressor layer into full model
    m_full.add_regressor(element)

  df_prophet = df_prophet.rename(columns={'period': 'ds', 
                                          'Total': 'y'})
  df_prophet['cap'] = multiplier*df_prophet.y.max()
  df_prophet['floor'] = 0

  m_full.fit(df_prophet)

  future_full = m_full.make_future_dataframe(44, freq='W') #MS  52
  future_full['cap'] = multiplier*df_prophet.y.max()
  future_full['floor'] = 0

  array_types=np.asarray(column_types[2:])
  array_types=np.append(array_types, 'ds')
  future_full = future_full.merge(df_forecast_regressors[array_types], 
                                  on='ds', 
                                  how='left')

  forecast_full = m_full.predict(future_full)
  fig = m_full.plot(forecast_full, 
                    ylabel='Euro', 
                    xlabel='date')

  #cutoff should be adjusted manually!
  #more information about cross validation https://facebook.github.io/prophet/docs/diagnostics.html
  df_cv = cross_validation(m_full, 
                           initial='1800 days', 
                           period='180 days', 
                           horizon = '270 days')
  df_p = performance_metrics(df_cv)

  return forecast_full, fig, df_p, df_cv
```

In the next section of the code, please specify the chosen parameters:

**1** - multiplier for logistic forecast (use e.g. 2 to change twice the cap value in the model, but use only 1 for linear approach).

**'linear'/'logistic'** - choose the model specification

**'period', 'Total'**, 'Gin'- the first two elements are required, additional regressors as Gin, Whiskey etc. are optionally



```python
forecast_second_approach, fig_second_approach, df_p_metrics, df_cv = (
    prophet_predict(store_id_chosen,
                    df_store_sales, 
                    1, 
                    'linear',
                    'period', 
                    'Total', 
                    'Gin'))
```

    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Making 1 forecasts with cutoffs between 2017-02-01 00:00:00 and 2017-02-01 00:00:00
    


    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    
    


![png]({{site.url}}/assets/images/prophet_files/output_76_3.png)


Logistic version


```python
forecast_second_approach, fig_second_approach, df_p_metrics, df_cv = (
    prophet_predict(store_id_chosen, 
                    df_store_sales,
                    1, 
                    'logistic', 
                    'period', 
                    'Total', 
                    'Gin'))
```

    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Making 1 forecasts with cutoffs between 2017-02-01 00:00:00 and 2017-02-01 00:00:00
    


    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    
    


![png]({{site.url}}/assets/images/prophet_files/output_78_3.png)


Logistic with multiplication for cap values


```python
forecast_second_approach, fig_second_approach, df_p_metrics, df_cv = (
    prophet_predict(store_id_chosen, 
                    df_store_sales, 
                    2, 
                    'logistic', 
                    'period', 
                    'Total', 
                    'Gin'))
```

    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Making 1 forecasts with cutoffs between 2017-02-01 00:00:00 and 2017-02-01 00:00:00
    


    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    
    


![png]({{site.url}}/assets/images/prophet_files/output_80_3.png)


Different start date of observations


```python
def prophet_predict_date_range(store_id_chosen: int, 
                               df_store_sales: pd.DataFrame, 
                               date_start_fitting: str, 
                               multiplier: int, 
                               growth_fbprophet: str, 
                               *column_types):

  df_store_sales= df_store_sales.loc[df_store_sales['store_id'] == store_id_chosen]
  df_prophet=df_store_sales[np.asarray(column_types)]
  df_prophet=df_prophet[df_prophet['period']>=date_start_fitting]

  m_full = fbprophet.Prophet(growth=growth_fbprophet)
  df_forecast_regressors = pd.DataFrame([], columns=['ds'])

  #predict additional regressors 
  for element in column_types[2:]:

    m=fbprophet.Prophet(growth=growth_fbprophet)
    df_inside = df_prophet.rename(columns={'period': 'ds', element: 'y'})
    df_inside['floor'] = 0
    df_inside['cap'] = df_inside.y.max()

    m.fit(df_inside[['ds', 'y', 'floor', 'cap']])

    future = m.make_future_dataframe(44, freq='W') #MS  52
    future['cap'] = df_inside.y.max()
    future['floor'] = 0

    forecast = m.predict(future)
    df_forecast_regressors = df_forecast_regressors.merge(forecast[['ds', 'yhat']], 
                                                          on='ds', 
                                                          how='outer')
    df_forecast_regressors = df_forecast_regressors.rename(columns={'yhat': element})
    
    m_full.add_regressor(element)

  df_prophet['floor'] = 0
  df_prophet = df_prophet.rename(columns={'period': 'ds', 
                                          'Total': 'y'})
  df_prophet['cap'] = multiplier*df_prophet.y.max()

  m_full.fit(df_prophet)

  future_full = m_full.make_future_dataframe(44, freq='W') #MS  52
  future_full['cap'] = multiplier*df_prophet.y.max()
  future_full['floor'] = 0

  array_types=np.asarray(column_types[2:])
  array_types=np.append(array_types, 'ds')
  future_full = future_full.merge(df_forecast_regressors[array_types], 
                                  on='ds', 
                                  how='left')

  forecast_full = m_full.predict(future_full)
  fig = m_full.plot(forecast_full, ylabel='Euro', xlabel='date')

  #cutoff should be adjucted manually
  df_cv = cross_validation(m_full, 
                           initial='730 days', 
                           period='180 days', 
                           horizon = '270 days')
  df_p = performance_metrics(df_cv)

  return forecast_full, fig, df_p, df_cv
```

**'2015-01-01'** enter start date for forecast


```python
forecast_second_approach, fig_second_approach, df_p_metrics, df_cv = (
    prophet_predict_date_range(
        store_id_chosen, 
        df_store_sales, 
        '2015-01-01', 
        2, 
        'logistic', 
        'period', 
        'Total', 
        'Gin'))
```

    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Making 1 forecasts with cutoffs between 2017-02-01 00:00:00 and 2017-02-01 00:00:00
    


    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    
    


![png]({{site.url}}/assets/images/prophet_files/output_84_3.png)



```python
#from fbprophet.plot import plot_cross_validation_metric
fig = fbprophet.plot.plot_cross_validation_metric(df_cv, metric='mape')
```


![png]({{site.url}}/assets/images/prophet_files/output_85_0.png)



```python
df_p_metrics.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>horizon</th>
      <th>mse</th>
      <th>rmse</th>
      <th>mae</th>
      <th>mape</th>
      <th>mdape</th>
      <th>coverage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18 days</td>
      <td>2.032174e+07</td>
      <td>4507.963676</td>
      <td>4455.489368</td>
      <td>0.598439</td>
      <td>0.585844</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25 days</td>
      <td>1.342665e+07</td>
      <td>3664.239998</td>
      <td>3544.375524</td>
      <td>0.429956</td>
      <td>0.478417</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32 days</td>
      <td>1.656929e+07</td>
      <td>4070.539116</td>
      <td>3849.406295</td>
      <td>0.506169</td>
      <td>0.478417</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39 days</td>
      <td>1.865996e+07</td>
      <td>4319.718001</td>
      <td>4115.942535</td>
      <td>0.486830</td>
      <td>0.420400</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>46 days</td>
      <td>2.337920e+07</td>
      <td>4835.204350</td>
      <td>4800.941722</td>
      <td>0.591528</td>
      <td>0.539703</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_cv.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>yhat</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>y</th>
      <th>cutoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-02-05</td>
      <td>12195.885741</td>
      <td>8493.102411</td>
      <td>15952.736176</td>
      <td>7045.34</td>
      <td>2017-02-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-02-12</td>
      <td>12705.813049</td>
      <td>9041.965477</td>
      <td>16419.567468</td>
      <td>8012.02</td>
      <td>2017-02-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-02-19</td>
      <td>10884.179313</td>
      <td>7017.838101</td>
      <td>14276.419510</td>
      <td>7362.05</td>
      <td>2017-02-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-02-26</td>
      <td>13131.454210</td>
      <td>9539.785105</td>
      <td>16624.701770</td>
      <td>10714.25</td>
      <td>2017-02-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-03-05</td>
      <td>12495.325361</td>
      <td>8618.546173</td>
      <td>16072.765590</td>
      <td>6886.44</td>
      <td>2017-02-01</td>
    </tr>
  </tbody>
</table>
</div>



Clustering was done previously. In this subsection we only choose stores with same cluster label.


```python
df_clustered = df_store_gdata.loc[df_store_gdata['store_id'] == store_id_chosen]
df_clustered_by_chosen_store= df_store_gdata.loc[df_store_gdata['cluster'] == int(df_clustered.cluster)]
df_clustered_by_chosen_store.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>university or college</th>
      <th>foodstores or supermarkets or gorceries</th>
      <th>restaurant</th>
      <th>churches</th>
      <th>gym</th>
      <th>stadium</th>
      <th>store_id</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26</th>
      <td>5</td>
      <td>7</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>3</td>
      <td>0</td>
      <td>135</td>
    </tr>
    <tr>
      <th>65</th>
      <td>5</td>
      <td>7</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>3</td>
      <td>39</td>
      <td>135</td>
    </tr>
    <tr>
      <th>66</th>
      <td>5</td>
      <td>7</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>3</td>
      <td>40</td>
      <td>135</td>
    </tr>
    <tr>
      <th>210</th>
      <td>4</td>
      <td>6</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>3</td>
      <td>184</td>
      <td>135</td>
    </tr>
    <tr>
      <th>301</th>
      <td>4</td>
      <td>6</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>3</td>
      <td>275</td>
      <td>135</td>
    </tr>
  </tbody>
</table>
</div>



Build a classifier using this "labelled" data:


```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
```


```python
np.random.seed(42)
train, test = train_test_split(df_store_gdata, test_size=0.20)

col_list_to_classify = ['university or college', 'foodstores or supermarkets or gorceries', 'restaurant', 'churches', 'gym', 'stadium']

X_train = train[col_list_to_classify]
y_train = train[['cluster']]

X_test = test[col_list_to_classify]
y_test = test[['cluster']]

knn = KNeighborsClassifier()
knn.fit(X_train, y_train) 
res = knn.predict(X_test)
acc = accuracy_score(res.transpose(), y_test.values)
acc
```




    0.896551724137931



By checking which cluster of stores belong to, it is needed to use knn.predict on the chosen model. Instead of using, again, clustering while adding one record (our client), the prediction using KNN Classifier gave us about 90% accuracy and is a quite good solution instead of creating new clusterization.


```python
df_another_stores_by_cluster=df_store_merged.loc[df_store_merged['store_id'].isin(df_clustered_by_chosen_store.store_id)]
```


```python
df_another_stores_by_cluster_by_the_store=df_another_stores_by_cluster.loc[df_store_merged['store_id']==store_id_chosen]
```


```python
d_concurrency = {'store_id': store_id_chosen, 
                 'Number of store in same cluster': df_another_stores_by_cluster.store_id.count(), 
                 'Median of the competitor stores (within 5km radius from each store) in the same cluster': np.median(df_another_stores_by_cluster.total_no_stores_in_5_km), 
                 'Number of competitor stores (within 5 km radius from each store) of chosen store': df_another_stores_by_cluster_by_the_store.total_no_stores_in_5_km}

df_concurrency = pd.DataFrame(data=d_concurrency)
df_concurrency
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>Number of store in same cluster</th>
      <th>Median of the competitor stores (within 5km radius from each store) in the same cluster</th>
      <th>Number of competitor stores (within 5 km radius from each store) of chosen store</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>33</td>
      <td>41.0</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>



*Interpretation*

If the level of competition is lower than the median/average, this means that the store potentially has less risk of competition.


```python
# pip freeze --local > "/content/gdrive/My Drive/Prophet/colab_installed.txt"
```


```python
import sys
```


```python
print(sys.version)
```

    3.7.10 (default, Feb 20 2021, 21:17:23) 
    [GCC 7.5.0]
    