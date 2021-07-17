---
layout: page
title: Predicting profitable customer segments
permalink: /customer_segments/
parent: Python
---

# Navigation Structure
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

# Predicting profitable customer segments

**Context**

Marketing is a key component of every modern business. Companies continuously re-invest large cuts of their profits for marketing purposes, trying to target groups of customers who have the potential to bring back the highest Return On Investment for the company. The cost of marketing can be very high though, meaning that the decision about which customer group to target is of great financial importance.

This dataset was made available by an online retail company that has collected historical data about such groups of customers, tracked the profitability of each group after the respective marketing campaign, and retrospectively assessed whether investing in marketing spend for that group was a good choice.

**Content**

To enable machine learning experimentation, this dataset has been structured as follows:

Each row is a comparison between two groups of potential customers:

* 1) Column names starting with "g1" represent characteristics of the first customer group (these were known before the campaign was run) 

* 2) Column names starting with "g2" represent characteristics of the second customer group (these were known before the campaign was run)

* 3) Column names starting with "c_" are features representing some comparison of the two groups (also known before the campaign was run)

The last column, named "target" is categorical, with three categories:

*   0 - none of the two groups were profitable
*   1 - group1 turned out to be more profitable
*   2 - group2 turned out to be more profitable

# Proposed steps for solving the tasks
5 different approaches were figured out and created for the task: 
 * **First approach**: try to predict return based on only the first group. 

Basing on the only first group, I have created the model using the DecisionTreeClassifier for **prediction of the campaign's success rate to be launched against this group**. Additionally, minmaxscaling and removing of columns g1_XX has been applied in cases where the correlation between them is very high (>0.9). This was used for the reduction of dimensionality. 
* **Second approach**: try to predict if the campaign should be launched (based on both groups). 

Basing on both groups, I have created the model using the DecisionTreeClassifier and GradientBoostingClassifier to **predict if the campaign should be launched against one of the possible cases (one of the groups, both of them, or simply none)**. Additionally, minmaxscaling, and removing columns g1_XX have been used for instances where the correlation between them is very high (>0.9) for reduction of dimensionality. 
* **Third approach**: try to predict which group should we target.

Basing on both groups, I have created the model using the GradientBoostingClassifier to **predict if the campaign should be launched for the group (one of them, or none of them)**. Additionally, minmaxscaling and removing columns g1_XX have been used if the correlation between them is very high (>0.9) for reduction of dimensionality in a similar way as two cases before.

* **Fourth approach**: try to predict if the campaign should be stopped based on the gX_21 column.

Basing on the third approach, an additional step has been added. In this case, **another model that tries to catch if the campaign should be stopped** is used. The assumption is that the gX_21 features have not been generated in the last stage of the campaign.

* **Fifth approach** use additional parameters c_XX to predict the campaign target group.

This approach is the connection of the third one and c params. In this case, the c_28 feature is not used (prior is it unknown when the campaign is launched). The created model was made by using the GradientBoostingClassifier to **predict if the campaign should be launched for the group (one of them, or none of them)**

**tl;dr**

According to obtained results, it was decided to use the third approach with GradientBoostingClassifier. For this classifier, the accuracy was over 0.54 for the test set. The exported model was used to build an application in python (using the flask library). By using Postman, we can perform queries to select the group for which we want to run marketing activities. 



```python
from google.colab import drive
drive.mount('/content/gdrive')
```


```python
import pandas as pd
import numpy as np
```

# Dataset investigation

Each row is a comparison between two customer groups. 
 
Column names starting with “g1_”: 
* contain information about the first customer group 
*  variables g1_1 until g1_20 were known before the campaign was run 
*  variable g1_21 was recorded after the campaign was run 
 
Column names starting with “g2_”: 
*  contain information about the second customer group 
*  variables g2_1 until g2_20 were known before the campaign was run 
*  variable g2_21 was recorded after the campaign was run 
 
Column names starting with “c_”: 
*  contain features representing some comparison of the two groups 
*  variables c_1 until c_27 were known before the campaign was run 
*  variable c_28 was recorded after the campaign was run 
 
Target – is categorical. This is what the categories mean: 
*  0: none of the two groups were profitable 
*  1: group 1 was the most profitable 
*  2: group 2 was the most profitable 

A caveat: if one of the groups was even slightly better (concerning ROI) then the target we set may affect the result. Based on the dataset, we do not know the return on investment for each group, so we do not know how the results of the two groups differed.


```python
campaign = pd.read_csv('gdrive/My Drive/customer_segments_predicting/customerGroups.csv', sep = ',')
campaign.head()
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
      <th>g1_1</th>
      <th>g1_2</th>
      <th>g1_3</th>
      <th>g1_4</th>
      <th>g1_5</th>
      <th>g1_6</th>
      <th>g1_7</th>
      <th>g1_8</th>
      <th>g1_9</th>
      <th>g1_10</th>
      <th>g1_11</th>
      <th>g1_12</th>
      <th>g1_13</th>
      <th>g1_14</th>
      <th>g1_15</th>
      <th>g1_16</th>
      <th>g1_17</th>
      <th>g1_18</th>
      <th>g1_19</th>
      <th>g1_20</th>
      <th>g1_21</th>
      <th>g2_1</th>
      <th>g2_2</th>
      <th>g2_3</th>
      <th>g2_4</th>
      <th>g2_5</th>
      <th>g2_6</th>
      <th>g2_7</th>
      <th>g2_8</th>
      <th>g2_9</th>
      <th>g2_10</th>
      <th>g2_11</th>
      <th>g2_12</th>
      <th>g2_13</th>
      <th>g2_14</th>
      <th>g2_15</th>
      <th>g2_16</th>
      <th>g2_17</th>
      <th>g2_18</th>
      <th>g2_19</th>
      <th>g2_20</th>
      <th>g2_21</th>
      <th>c_1</th>
      <th>c_2</th>
      <th>c_3</th>
      <th>c_4</th>
      <th>c_5</th>
      <th>c_6</th>
      <th>c_7</th>
      <th>c_8</th>
      <th>c_9</th>
      <th>c_10</th>
      <th>c_11</th>
      <th>c_12</th>
      <th>c_13</th>
      <th>c_14</th>
      <th>c_15</th>
      <th>c_16</th>
      <th>c_17</th>
      <th>c_18</th>
      <th>c_19</th>
      <th>c_20</th>
      <th>c_21</th>
      <th>c_22</th>
      <th>c_23</th>
      <th>c_24</th>
      <th>c_25</th>
      <th>c_26</th>
      <th>c_27</th>
      <th>c_28</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.50</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>-2</td>
      <td>-2</td>
      <td>2.505032</td>
      <td>2.551406</td>
      <td>6.240000</td>
      <td>3.608000</td>
      <td>0.744000</td>
      <td>1.216000</td>
      <td>0.003078</td>
      <td>0.003028</td>
      <td>0.578205</td>
      <td>1.83</td>
      <td>6</td>
      <td>0</td>
      <td>6</td>
      <td>7</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>-1</td>
      <td>3</td>
      <td>2.888736</td>
      <td>2.616855</td>
      <td>5.552000</td>
      <td>0.728000</td>
      <td>0.160000</td>
      <td>0.002994</td>
      <td>0.002953</td>
      <td>0.586149</td>
      <td>3.50</td>
      <td>1.97</td>
      <td>-1</td>
      <td>7</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3.223605</td>
      <td>1</td>
      <td>-3</td>
      <td>-2</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>-6</td>
      <td>-5</td>
      <td>-0.383704</td>
      <td>-0.065449</td>
      <td>0.584000</td>
      <td>0.488000</td>
      <td>0</td>
      <td>-3.232000</td>
      <td>-1.944000</td>
      <td>-0.007944</td>
      <td>1.76</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.20</td>
      <td>24</td>
      <td>22</td>
      <td>46</td>
      <td>10</td>
      <td>24</td>
      <td>28</td>
      <td>18</td>
      <td>22</td>
      <td>-4</td>
      <td>-4</td>
      <td>-8</td>
      <td>3.718983</td>
      <td>3.882271</td>
      <td>7.423435</td>
      <td>5.048030</td>
      <td>0.836178</td>
      <td>1.975244</td>
      <td>0.784882</td>
      <td>0.019448</td>
      <td>0.680013</td>
      <td>2.80</td>
      <td>34</td>
      <td>14</td>
      <td>48</td>
      <td>10</td>
      <td>25</td>
      <td>16</td>
      <td>16</td>
      <td>24</td>
      <td>9</td>
      <td>-8</td>
      <td>1</td>
      <td>4.065822</td>
      <td>4.042015</td>
      <td>6.369385</td>
      <td>1.511704</td>
      <td>1.783791</td>
      <td>0.784882</td>
      <td>0.033373</td>
      <td>0.498949</td>
      <td>3.25</td>
      <td>1.85</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.541039</td>
      <td>10</td>
      <td>-12</td>
      <td>-2</td>
      <td>0</td>
      <td>12</td>
      <td>2</td>
      <td>-3</td>
      <td>4</td>
      <td>-13</td>
      <td>-9</td>
      <td>-0.346839</td>
      <td>-0.159744</td>
      <td>-0.947614</td>
      <td>0.463540</td>
      <td>0</td>
      <td>-5.342174</td>
      <td>-1.321355</td>
      <td>0.181064</td>
      <td>1.85</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.00</td>
      <td>7</td>
      <td>4</td>
      <td>11</td>
      <td>18</td>
      <td>8</td>
      <td>11</td>
      <td>2</td>
      <td>10</td>
      <td>-3</td>
      <td>-8</td>
      <td>-11</td>
      <td>2.244550</td>
      <td>2.458087</td>
      <td>11.091399</td>
      <td>5.853005</td>
      <td>0.730046</td>
      <td>2.022004</td>
      <td>0.043937</td>
      <td>0.014264</td>
      <td>0.527707</td>
      <td>1.30</td>
      <td>11</td>
      <td>18</td>
      <td>29</td>
      <td>2</td>
      <td>13</td>
      <td>3</td>
      <td>16</td>
      <td>1</td>
      <td>10</td>
      <td>15</td>
      <td>25</td>
      <td>4.918483</td>
      <td>4.050389</td>
      <td>10.029408</td>
      <td>2.489174</td>
      <td>0.204741</td>
      <td>0.022247</td>
      <td>0.042004</td>
      <td>0.567984</td>
      <td>5.00</td>
      <td>1.70</td>
      <td>-5</td>
      <td>10</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2.049024</td>
      <td>-11</td>
      <td>-7</td>
      <td>-18</td>
      <td>7</td>
      <td>-5</td>
      <td>-1</td>
      <td>-3</td>
      <td>-18</td>
      <td>-18</td>
      <td>-36</td>
      <td>-2.673934</td>
      <td>-1.592303</td>
      <td>0.525305</td>
      <td>-0.467169</td>
      <td>0</td>
      <td>-6.566521</td>
      <td>-4.176403</td>
      <td>-0.040277</td>
      <td>2.05</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.91</td>
      <td>8</td>
      <td>5</td>
      <td>13</td>
      <td>14</td>
      <td>6</td>
      <td>7</td>
      <td>6</td>
      <td>9</td>
      <td>-1</td>
      <td>-3</td>
      <td>-4</td>
      <td>2.580190</td>
      <td>2.683092</td>
      <td>9.864426</td>
      <td>2.582357</td>
      <td>0.656638</td>
      <td>1.407549</td>
      <td>0.041563</td>
      <td>0.021386</td>
      <td>0.261785</td>
      <td>4.50</td>
      <td>5</td>
      <td>3</td>
      <td>8</td>
      <td>17</td>
      <td>5</td>
      <td>9</td>
      <td>7</td>
      <td>16</td>
      <td>-4</td>
      <td>-9</td>
      <td>-13</td>
      <td>1.964163</td>
      <td>2.278147</td>
      <td>3.369489</td>
      <td>0.665585</td>
      <td>2.163561</td>
      <td>0.043937</td>
      <td>0.010358</td>
      <td>0.273886</td>
      <td>3.60</td>
      <td>1.98</td>
      <td>-1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2.284503</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>-10</td>
      <td>0</td>
      <td>-3</td>
      <td>4</td>
      <td>8</td>
      <td>1</td>
      <td>9</td>
      <td>0.616027</td>
      <td>0.404945</td>
      <td>-1.506923</td>
      <td>0.741964</td>
      <td>0</td>
      <td>-2.438120</td>
      <td>-0.787132</td>
      <td>-0.012101</td>
      <td>1.82</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.50</td>
      <td>23</td>
      <td>16</td>
      <td>39</td>
      <td>14</td>
      <td>33</td>
      <td>25</td>
      <td>18</td>
      <td>27</td>
      <td>8</td>
      <td>-9</td>
      <td>-1</td>
      <td>3.470617</td>
      <td>3.055989</td>
      <td>11.672962</td>
      <td>4.554560</td>
      <td>1.895740</td>
      <td>1.237122</td>
      <td>0.941241</td>
      <td>0.000062</td>
      <td>0.390180</td>
      <td>3.00</td>
      <td>29</td>
      <td>23</td>
      <td>52</td>
      <td>8</td>
      <td>31</td>
      <td>22</td>
      <td>21</td>
      <td>23</td>
      <td>9</td>
      <td>-2</td>
      <td>7</td>
      <td>4.527831</td>
      <td>4.215284</td>
      <td>4.494986</td>
      <td>1.419174</td>
      <td>1.144728</td>
      <td>0.364776</td>
      <td>0.008148</td>
      <td>0.347568</td>
      <td>3.40</td>
      <td>1.80</td>
      <td>-3</td>
      <td>2</td>
      <td>-1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2.648418</td>
      <td>0</td>
      <td>-13</td>
      <td>-13</td>
      <td>10</td>
      <td>4</td>
      <td>-4</td>
      <td>-4</td>
      <td>10</td>
      <td>-18</td>
      <td>-8</td>
      <td>-1.057214</td>
      <td>-1.159294</td>
      <td>0.751012</td>
      <td>-0.182052</td>
      <td>0</td>
      <td>-1.259728</td>
      <td>0.059574</td>
      <td>0.042613</td>
      <td>1.99</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
campaign.describe()
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
      <th>g1_1</th>
      <th>g1_2</th>
      <th>g1_3</th>
      <th>g1_4</th>
      <th>g1_5</th>
      <th>g1_6</th>
      <th>g1_7</th>
      <th>g1_8</th>
      <th>g1_9</th>
      <th>g1_10</th>
      <th>g1_11</th>
      <th>g1_12</th>
      <th>g1_13</th>
      <th>g1_14</th>
      <th>g1_15</th>
      <th>g1_16</th>
      <th>g1_17</th>
      <th>g1_18</th>
      <th>g1_19</th>
      <th>g1_20</th>
      <th>g1_21</th>
      <th>g2_1</th>
      <th>g2_2</th>
      <th>g2_3</th>
      <th>g2_4</th>
      <th>g2_5</th>
      <th>g2_6</th>
      <th>g2_7</th>
      <th>g2_8</th>
      <th>g2_9</th>
      <th>g2_10</th>
      <th>g2_11</th>
      <th>g2_12</th>
      <th>g2_13</th>
      <th>g2_14</th>
      <th>g2_15</th>
      <th>g2_16</th>
      <th>g2_17</th>
      <th>g2_18</th>
      <th>g2_19</th>
      <th>g2_20</th>
      <th>g2_21</th>
      <th>c_1</th>
      <th>c_2</th>
      <th>c_3</th>
      <th>c_4</th>
      <th>c_5</th>
      <th>c_6</th>
      <th>c_7</th>
      <th>c_8</th>
      <th>c_9</th>
      <th>c_10</th>
      <th>c_11</th>
      <th>c_12</th>
      <th>c_13</th>
      <th>c_14</th>
      <th>c_15</th>
      <th>c_16</th>
      <th>c_17</th>
      <th>c_18</th>
      <th>c_19</th>
      <th>c_20</th>
      <th>c_21</th>
      <th>c_22</th>
      <th>c_23</th>
      <th>c_24</th>
      <th>c_25</th>
      <th>c_26</th>
      <th>c_27</th>
      <th>c_28</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.00000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.00000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
      <td>6620.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.708779</td>
      <td>14.424018</td>
      <td>10.485650</td>
      <td>24.909668</td>
      <td>10.988066</td>
      <td>13.412085</td>
      <td>10.161027</td>
      <td>10.745468</td>
      <td>14.169033</td>
      <td>3.251057</td>
      <td>-3.423565</td>
      <td>-0.172508</td>
      <td>3.154143</td>
      <td>3.043544</td>
      <td>10.268756</td>
      <td>4.736862</td>
      <td>1.121263</td>
      <td>1.159848</td>
      <td>0.205070</td>
      <td>0.058852</td>
      <td>0.449405</td>
      <td>4.809875</td>
      <td>15.113897</td>
      <td>10.018580</td>
      <td>25.132477</td>
      <td>11.025076</td>
      <td>14.040634</td>
      <td>10.633837</td>
      <td>10.253323</td>
      <td>13.500453</td>
      <td>3.406798</td>
      <td>-3.247130</td>
      <td>0.159668</td>
      <td>3.183453</td>
      <td>3.050268</td>
      <td>4.840590</td>
      <td>1.151021</td>
      <td>1.125410</td>
      <td>0.205567</td>
      <td>0.058207</td>
      <td>0.448996</td>
      <td>3.899359</td>
      <td>1.88984</td>
      <td>1.563595</td>
      <td>1.558761</td>
      <td>3.122356</td>
      <td>0.183686</td>
      <td>0.200906</td>
      <td>0.183686</td>
      <td>0.200906</td>
      <td>1.945808</td>
      <td>4.405438</td>
      <td>-4.628248</td>
      <td>-0.222810</td>
      <td>-0.088369</td>
      <td>-0.092296</td>
      <td>0.111631</td>
      <td>0.128399</td>
      <td>6.498187</td>
      <td>-6.830363</td>
      <td>-0.332175</td>
      <td>-0.029311</td>
      <td>-0.006724</td>
      <td>-0.004147</td>
      <td>0.008827</td>
      <td>0.00000</td>
      <td>-0.228426</td>
      <td>-0.103728</td>
      <td>0.000408</td>
      <td>1.917134</td>
      <td>1.031722</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.857725</td>
      <td>10.700787</td>
      <td>8.384203</td>
      <td>18.174948</td>
      <td>5.635985</td>
      <td>10.090030</td>
      <td>7.495039</td>
      <td>7.964247</td>
      <td>9.866734</td>
      <td>8.481210</td>
      <td>8.580752</td>
      <td>15.036306</td>
      <td>0.931224</td>
      <td>0.825628</td>
      <td>3.760524</td>
      <td>2.127352</td>
      <td>0.580622</td>
      <td>0.566745</td>
      <td>0.273416</td>
      <td>0.151767</td>
      <td>0.139392</td>
      <td>3.937164</td>
      <td>10.836923</td>
      <td>8.251602</td>
      <td>18.190664</td>
      <td>5.666965</td>
      <td>10.205415</td>
      <td>7.563664</td>
      <td>7.831935</td>
      <td>9.722428</td>
      <td>8.750434</td>
      <td>8.313375</td>
      <td>15.025919</td>
      <td>0.928835</td>
      <td>0.823931</td>
      <td>2.150843</td>
      <td>0.588387</td>
      <td>0.552912</td>
      <td>0.273798</td>
      <td>0.151470</td>
      <td>0.139194</td>
      <td>1.093160</td>
      <td>0.22708</td>
      <td>4.063520</td>
      <td>4.057417</td>
      <td>3.939467</td>
      <td>0.387257</td>
      <td>0.400708</td>
      <td>0.387257</td>
      <td>0.400708</td>
      <td>1.217214</td>
      <td>8.497254</td>
      <td>9.093944</td>
      <td>14.470732</td>
      <td>7.466654</td>
      <td>6.378463</td>
      <td>6.343190</td>
      <td>7.420718</td>
      <td>12.175872</td>
      <td>12.973601</td>
      <td>21.498095</td>
      <td>1.220752</td>
      <td>1.068199</td>
      <td>0.663238</td>
      <td>0.683422</td>
      <td>0.32287</td>
      <td>3.390902</td>
      <td>1.944419</td>
      <td>0.092761</td>
      <td>0.302175</td>
      <td>0.731042</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-27.000000</td>
      <td>-38.000000</td>
      <td>-65.000000</td>
      <td>0.000000</td>
      <td>0.172875</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.950000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-27.000000</td>
      <td>-36.000000</td>
      <td>-63.000000</td>
      <td>0.000000</td>
      <td>0.216094</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.500000</td>
      <td>0.00000</td>
      <td>-10.000000</td>
      <td>-10.000000</td>
      <td>-6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-28.000000</td>
      <td>-47.000000</td>
      <td>-63.000000</td>
      <td>-33.000000</td>
      <td>-29.000000</td>
      <td>-28.000000</td>
      <td>-40.000000</td>
      <td>-39.000000</td>
      <td>-74.000000</td>
      <td>-101.000000</td>
      <td>-4.684111</td>
      <td>-4.319826</td>
      <td>-2.512919</td>
      <td>-3.118836</td>
      <td>-2.00000</td>
      <td>-15.202740</td>
      <td>-9.181722</td>
      <td>-0.750000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.667000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>10.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>-2.000000</td>
      <td>-8.000000</td>
      <td>-8.250000</td>
      <td>2.499106</td>
      <td>2.493665</td>
      <td>8.512643</td>
      <td>3.389698</td>
      <td>0.735749</td>
      <td>0.780218</td>
      <td>0.011054</td>
      <td>0.001827</td>
      <td>0.348935</td>
      <td>2.500000</td>
      <td>6.000000</td>
      <td>3.000000</td>
      <td>11.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>-2.000000</td>
      <td>-8.000000</td>
      <td>-8.000000</td>
      <td>2.500000</td>
      <td>2.500000</td>
      <td>3.470612</td>
      <td>0.760000</td>
      <td>0.758589</td>
      <td>0.011054</td>
      <td>0.001827</td>
      <td>0.347931</td>
      <td>3.250000</td>
      <td>1.74000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.901358</td>
      <td>0.000000</td>
      <td>-9.000000</td>
      <td>-7.000000</td>
      <td>-4.000000</td>
      <td>-3.000000</td>
      <td>-3.000000</td>
      <td>-3.000000</td>
      <td>-1.000000</td>
      <td>-13.000000</td>
      <td>-10.000000</td>
      <td>-0.716407</td>
      <td>-0.531224</td>
      <td>-0.440407</td>
      <td>-0.406811</td>
      <td>0.00000</td>
      <td>-2.222226</td>
      <td>-1.293471</td>
      <td>-0.054331</td>
      <td>1.710000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.150000</td>
      <td>13.000000</td>
      <td>9.000000</td>
      <td>22.000000</td>
      <td>11.000000</td>
      <td>12.000000</td>
      <td>9.000000</td>
      <td>9.000000</td>
      <td>13.000000</td>
      <td>1.000000</td>
      <td>-2.000000</td>
      <td>-1.000000</td>
      <td>2.905237</td>
      <td>2.764749</td>
      <td>10.539520</td>
      <td>4.675946</td>
      <td>1.079138</td>
      <td>1.156497</td>
      <td>0.065102</td>
      <td>0.006409</td>
      <td>0.482790</td>
      <td>3.500000</td>
      <td>14.000000</td>
      <td>8.000000</td>
      <td>22.000000</td>
      <td>11.000000</td>
      <td>13.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>12.000000</td>
      <td>1.000000</td>
      <td>-2.000000</td>
      <td>-1.000000</td>
      <td>2.931093</td>
      <td>2.769496</td>
      <td>4.789416</td>
      <td>1.110897</td>
      <td>1.126207</td>
      <td>0.067168</td>
      <td>0.006314</td>
      <td>0.483239</td>
      <td>3.500000</td>
      <td>1.90000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.012806</td>
      <td>3.000000</td>
      <td>-3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>-4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.001070</td>
      <td>0.00000</td>
      <td>-0.119378</td>
      <td>-0.012487</td>
      <td>0.000000</td>
      <td>1.850000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.800000</td>
      <td>21.000000</td>
      <td>15.000000</td>
      <td>36.000000</td>
      <td>16.000000</td>
      <td>20.000000</td>
      <td>15.000000</td>
      <td>16.000000</td>
      <td>21.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>3.756311</td>
      <td>3.491114</td>
      <td>12.500879</td>
      <td>6.072358</td>
      <td>1.472574</td>
      <td>1.533258</td>
      <td>0.314664</td>
      <td>0.029305</td>
      <td>0.554830</td>
      <td>5.500000</td>
      <td>22.000000</td>
      <td>15.000000</td>
      <td>36.000000</td>
      <td>16.000000</td>
      <td>20.000000</td>
      <td>16.000000</td>
      <td>15.000000</td>
      <td>20.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>3.779267</td>
      <td>3.497749</td>
      <td>6.221507</td>
      <td>1.500404</td>
      <td>1.482003</td>
      <td>0.314664</td>
      <td>0.028546</td>
      <td>0.553774</td>
      <td>4.000000</td>
      <td>2.04000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.982930</td>
      <td>9.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>13.000000</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>0.654627</td>
      <td>0.490504</td>
      <td>0.410915</td>
      <td>0.450104</td>
      <td>0.00000</td>
      <td>1.809334</td>
      <td>1.035235</td>
      <td>0.054825</td>
      <td>2.020000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>23.000000</td>
      <td>52.000000</td>
      <td>47.000000</td>
      <td>94.000000</td>
      <td>20.000000</td>
      <td>61.000000</td>
      <td>43.000000</td>
      <td>48.000000</td>
      <td>52.000000</td>
      <td>48.000000</td>
      <td>31.000000</td>
      <td>76.000000</td>
      <td>5.000000</td>
      <td>4.994496</td>
      <td>20.502260</td>
      <td>12.520989</td>
      <td>3.721637</td>
      <td>3.185834</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>41.000000</td>
      <td>55.000000</td>
      <td>47.000000</td>
      <td>97.000000</td>
      <td>20.000000</td>
      <td>61.000000</td>
      <td>43.000000</td>
      <td>45.000000</td>
      <td>50.000000</td>
      <td>47.000000</td>
      <td>31.000000</td>
      <td>78.000000</td>
      <td>5.000000</td>
      <td>4.995596</td>
      <td>13.428196</td>
      <td>3.721637</td>
      <td>3.197138</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>19.000000</td>
      <td>2.91000</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>18.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>5.334529</td>
      <td>41.000000</td>
      <td>33.000000</td>
      <td>74.000000</td>
      <td>38.000000</td>
      <td>23.000000</td>
      <td>34.000000</td>
      <td>36.000000</td>
      <td>75.000000</td>
      <td>42.000000</td>
      <td>108.000000</td>
      <td>4.821136</td>
      <td>4.396281</td>
      <td>2.987136</td>
      <td>2.830550</td>
      <td>2.00000</td>
      <td>12.562698</td>
      <td>8.209578</td>
      <td>0.666667</td>
      <td>4.330000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>



Based on the describe function results, it may be thought that the values in column g2_21 should be scaled as it is in the column g1_21.


```python
# Check the number of nans in df
sum(campaign.isna().sum())
```




    0




```python
campaign[['target']].value_counts()
```




    target
    1         3076
    2         1877
    0         1667
    dtype: int64



What percentage of campaigns led to group 1 is the most profitable? What about group 2? And neither of the groups? 

Based on the results, more profits generate group 1. In 46.46% the campaign target group 1 gives us good ROI.


```python
campaign[['target']].value_counts()[1]/campaign[['target']].value_counts().sum()*100
```




    target
    1         46.465257
    dtype: float64



# First approach: try to predict return based on only the first group

Basing on the only first group, I have created the model using the DecisionTreeClassifier for **prediction of the campaign's success rate to be launched against this group**. Additionally, minmaxscaling and removing of columns g1_XX has been applied in cases where the correlation between them is very high (>0.9). This was used for the reduction of dimensionality. 


```python
# Get one hot encoding of columns target to use categorical label in prediction
# The prediction is to get the successful campaign of the first group
one_hot = pd.get_dummies(campaign['target'])

# Join the encoded df
campaign = campaign.join(one_hot)
```


```python
# Filtering out columns other than group 1, target and encoded target
df_first_group = campaign.filter(regex = 'g1_|target|^[0-9]')
```


```python
df_first_group
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
      <th>g1_1</th>
      <th>g1_2</th>
      <th>g1_3</th>
      <th>g1_4</th>
      <th>g1_5</th>
      <th>g1_6</th>
      <th>g1_7</th>
      <th>g1_8</th>
      <th>g1_9</th>
      <th>g1_10</th>
      <th>g1_11</th>
      <th>g1_12</th>
      <th>g1_13</th>
      <th>g1_14</th>
      <th>g1_15</th>
      <th>g1_16</th>
      <th>g1_17</th>
      <th>g1_18</th>
      <th>g1_19</th>
      <th>g1_20</th>
      <th>g1_21</th>
      <th>target</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.50</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>-2</td>
      <td>-2</td>
      <td>2.505032</td>
      <td>2.551406</td>
      <td>6.240000</td>
      <td>3.608000</td>
      <td>0.744000</td>
      <td>1.216000</td>
      <td>0.003078</td>
      <td>0.003028</td>
      <td>0.578205</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.20</td>
      <td>24</td>
      <td>22</td>
      <td>46</td>
      <td>10</td>
      <td>24</td>
      <td>28</td>
      <td>18</td>
      <td>22</td>
      <td>-4</td>
      <td>-4</td>
      <td>-8</td>
      <td>3.718983</td>
      <td>3.882271</td>
      <td>7.423435</td>
      <td>5.048030</td>
      <td>0.836178</td>
      <td>1.975244</td>
      <td>0.784882</td>
      <td>0.019448</td>
      <td>0.680013</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.00</td>
      <td>7</td>
      <td>4</td>
      <td>11</td>
      <td>18</td>
      <td>8</td>
      <td>11</td>
      <td>2</td>
      <td>10</td>
      <td>-3</td>
      <td>-8</td>
      <td>-11</td>
      <td>2.244550</td>
      <td>2.458087</td>
      <td>11.091399</td>
      <td>5.853005</td>
      <td>0.730046</td>
      <td>2.022004</td>
      <td>0.043937</td>
      <td>0.014264</td>
      <td>0.527707</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.91</td>
      <td>8</td>
      <td>5</td>
      <td>13</td>
      <td>14</td>
      <td>6</td>
      <td>7</td>
      <td>6</td>
      <td>9</td>
      <td>-1</td>
      <td>-3</td>
      <td>-4</td>
      <td>2.580190</td>
      <td>2.683092</td>
      <td>9.864426</td>
      <td>2.582357</td>
      <td>0.656638</td>
      <td>1.407549</td>
      <td>0.041563</td>
      <td>0.021386</td>
      <td>0.261785</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.50</td>
      <td>23</td>
      <td>16</td>
      <td>39</td>
      <td>14</td>
      <td>33</td>
      <td>25</td>
      <td>18</td>
      <td>27</td>
      <td>8</td>
      <td>-9</td>
      <td>-1</td>
      <td>3.470617</td>
      <td>3.055989</td>
      <td>11.672962</td>
      <td>4.554560</td>
      <td>1.895740</td>
      <td>1.237122</td>
      <td>0.941241</td>
      <td>0.000062</td>
      <td>0.390180</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>6615</th>
      <td>1.30</td>
      <td>3</td>
      <td>6</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>9</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>7</td>
      <td>3.225437</td>
      <td>2.745887</td>
      <td>8.128000</td>
      <td>3.584000</td>
      <td>1.904000</td>
      <td>0.728000</td>
      <td>0.002832</td>
      <td>0.003089</td>
      <td>0.440945</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6616</th>
      <td>1.85</td>
      <td>19</td>
      <td>12</td>
      <td>31</td>
      <td>6</td>
      <td>15</td>
      <td>9</td>
      <td>15</td>
      <td>14</td>
      <td>6</td>
      <td>1</td>
      <td>7</td>
      <td>3.802635</td>
      <td>3.643989</td>
      <td>12.445367</td>
      <td>3.772041</td>
      <td>1.275865</td>
      <td>1.215939</td>
      <td>0.090605</td>
      <td>0.064248</td>
      <td>0.303088</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6617</th>
      <td>2.50</td>
      <td>5</td>
      <td>8</td>
      <td>13</td>
      <td>19</td>
      <td>3</td>
      <td>8</td>
      <td>12</td>
      <td>10</td>
      <td>-5</td>
      <td>2</td>
      <td>-3</td>
      <td>2.639729</td>
      <td>2.639532</td>
      <td>12.735732</td>
      <td>6.930399</td>
      <td>1.018920</td>
      <td>1.422971</td>
      <td>0.056135</td>
      <td>0.007597</td>
      <td>0.544170</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6618</th>
      <td>1.80</td>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>10</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>2.733488</td>
      <td>2.573255</td>
      <td>11.668416</td>
      <td>4.200704</td>
      <td>0.721472</td>
      <td>0.597312</td>
      <td>0.007508</td>
      <td>0.004967</td>
      <td>0.360006</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6619</th>
      <td>1.95</td>
      <td>35</td>
      <td>37</td>
      <td>72</td>
      <td>1</td>
      <td>26</td>
      <td>15</td>
      <td>31</td>
      <td>16</td>
      <td>11</td>
      <td>15</td>
      <td>26</td>
      <td>5.000000</td>
      <td>4.931130</td>
      <td>12.919765</td>
      <td>4.544010</td>
      <td>1.411588</td>
      <td>0.304650</td>
      <td>0.006134</td>
      <td>0.923116</td>
      <td>0.351710</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>6620 rows × 25 columns</p>
</div>




```python
from sklearn.model_selection import train_test_split, cross_val_score

# Exclude g1_21 as the unknown when the campaign is launched
# X is the first group features
X = df_first_group.filter(regex = '^(?!g1_21)g1_') 

# y is the successful campaign of the first group
y = df_first_group.loc[:, df_first_group.columns == 1]

# Split the X and our target variable into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```


```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth = 5, random_state = 0)
clf.fit(X_train, y_train)
cross_val_score(clf, X, y, cv=10)
```




    array([0.66767372, 0.63746224, 0.66616314, 0.63444109, 0.66163142,
           0.65558912, 0.63293051, 0.65558912, 0.64048338, 0.65256798])




```python
from sklearn.metrics import multilabel_confusion_matrix

clf_predictions = clf.predict(X_test)
multilabel_confusion_matrix(y_test, clf_predictions)
```




    array([[[315, 289],
            [169, 551]],
    
           [[551, 169],
            [289, 315]]])



Of all 1324 test cases, classifier finds 551 launch campaigns that could be categorized as successful and 315 cases that campaign shouldn't be created. That's 65.4% accuracy of our classifier.

Let's show the correlation between the encoded target for the first group.


```python
import seaborn as sns
%matplotlib inline

# Calculate the correlation matrix
correlation_train_set = X_train.corr()

# Plot the heatmap to visualise the correlation of the features
sns.heatmap(correlation_train_set, 
            xticklabels = correlation_train_set.columns,
            yticklabels = correlation_train_set.columns)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fa484318ad0>




![png]({{site.url}}/assets/images/Predicting_Profitable_Customer_Segments_files/Predicting_Profitable_Customer_Segments_25_1.png)    
    



```python
# Using preprocessing minmaxscaler to change the range of the feautres
from sklearn import preprocessing

def scale_values(X: pd.DataFrame) -> pd.DataFrame:
  """
  Scale features by minmaxscaler

  """
  x = X.values # Returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  X_scaled = pd.DataFrame(x_scaled, columns = X.columns)

  return X_scaled

X_train = scale_values(X_train)
```

Use the correlation between features to get only one column (instead of using all independent variables).


```python
def drop_correlated_features(X_train: pd.DataFrame):
  corr_matrix = X_train.corr().abs()

  # Select upper_part trof correlation matrix
  upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

  # Find features with correlation greater than 0.90
  features_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.90)]

  # Drop features 
  X_train.drop(features_to_drop, axis = 1, inplace=True)

  return X_train, features_to_drop

X_train_scaled, features_to_drop = drop_correlated_features(X_train)
```


```python
# Save dropped features for further API purposes
import json
with open('/content/gdrive/My Drive/customer_segments_predicting/features_to_drop.json', 'w') as output_json:
    json.dump(features_to_drop, output_json)
```


```python
# Show selected features
X_train_scaled
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
      <th>g1_1</th>
      <th>g1_2</th>
      <th>g1_3</th>
      <th>g1_5</th>
      <th>g1_7</th>
      <th>g1_9</th>
      <th>g1_10</th>
      <th>g1_11</th>
      <th>g1_12</th>
      <th>g1_13</th>
      <th>g1_15</th>
      <th>g1_16</th>
      <th>g1_17</th>
      <th>g1_18</th>
      <th>g1_19</th>
      <th>g1_20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.052392</td>
      <td>0.096154</td>
      <td>0.085106</td>
      <td>0.315789</td>
      <td>0.023256</td>
      <td>0.134615</td>
      <td>0.373333</td>
      <td>0.478261</td>
      <td>0.432624</td>
      <td>0.518286</td>
      <td>0.378995</td>
      <td>0.302635</td>
      <td>0.110128</td>
      <td>0.357061</td>
      <td>0.010509</td>
      <td>0.006738</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.006834</td>
      <td>0.615385</td>
      <td>0.489362</td>
      <td>0.105263</td>
      <td>0.209302</td>
      <td>0.307692</td>
      <td>0.613333</td>
      <td>0.695652</td>
      <td>0.666667</td>
      <td>0.987124</td>
      <td>0.875567</td>
      <td>0.790637</td>
      <td>0.456739</td>
      <td>0.247622</td>
      <td>0.169272</td>
      <td>0.522046</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.066059</td>
      <td>0.076923</td>
      <td>0.021277</td>
      <td>0.789474</td>
      <td>0.139535</td>
      <td>0.096154</td>
      <td>0.360000</td>
      <td>0.521739</td>
      <td>0.446809</td>
      <td>0.495890</td>
      <td>0.396688</td>
      <td>0.355862</td>
      <td>0.306257</td>
      <td>0.424941</td>
      <td>0.008189</td>
      <td>0.005462</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.015945</td>
      <td>0.134615</td>
      <td>0.212766</td>
      <td>0.052632</td>
      <td>0.046512</td>
      <td>0.038462</td>
      <td>0.386667</td>
      <td>0.608696</td>
      <td>0.503546</td>
      <td>0.688240</td>
      <td>0.589805</td>
      <td>0.485894</td>
      <td>0.351539</td>
      <td>0.178269</td>
      <td>0.008386</td>
      <td>0.010620</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.093394</td>
      <td>0.057692</td>
      <td>0.085106</td>
      <td>0.736842</td>
      <td>0.186047</td>
      <td>0.230769</td>
      <td>0.293333</td>
      <td>0.449275</td>
      <td>0.375887</td>
      <td>0.410594</td>
      <td>0.341043</td>
      <td>0.192402</td>
      <td>0.188375</td>
      <td>0.866021</td>
      <td>0.014462</td>
      <td>0.007263</td>
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
      <th>5291</th>
      <td>0.061503</td>
      <td>0.076923</td>
      <td>0.063830</td>
      <td>1.000000</td>
      <td>0.302326</td>
      <td>0.307692</td>
      <td>0.253333</td>
      <td>0.434783</td>
      <td>0.347518</td>
      <td>0.353289</td>
      <td>0.597160</td>
      <td>0.578858</td>
      <td>0.253232</td>
      <td>0.483775</td>
      <td>0.088282</td>
      <td>0.001225</td>
    </tr>
    <tr>
      <th>5292</th>
      <td>0.030524</td>
      <td>0.076923</td>
      <td>0.148936</td>
      <td>0.578947</td>
      <td>0.139535</td>
      <td>0.096154</td>
      <td>0.306667</td>
      <td>0.579710</td>
      <td>0.446809</td>
      <td>0.545162</td>
      <td>0.479694</td>
      <td>0.446075</td>
      <td>0.303471</td>
      <td>0.408376</td>
      <td>0.014462</td>
      <td>0.006409</td>
    </tr>
    <tr>
      <th>5293</th>
      <td>0.107062</td>
      <td>0.192308</td>
      <td>0.127660</td>
      <td>1.000000</td>
      <td>0.674419</td>
      <td>0.634615</td>
      <td>0.120000</td>
      <td>0.231884</td>
      <td>0.177305</td>
      <td>0.208641</td>
      <td>0.510886</td>
      <td>0.218138</td>
      <td>0.085544</td>
      <td>0.797435</td>
      <td>0.535261</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>5294</th>
      <td>0.084282</td>
      <td>0.076923</td>
      <td>0.212766</td>
      <td>0.684211</td>
      <td>0.209302</td>
      <td>0.038462</td>
      <td>0.306667</td>
      <td>0.565217</td>
      <td>0.439716</td>
      <td>0.551292</td>
      <td>0.379544</td>
      <td>0.219643</td>
      <td>0.210401</td>
      <td>0.241557</td>
      <td>0.025318</td>
      <td>0.014264</td>
    </tr>
    <tr>
      <th>5295</th>
      <td>0.027790</td>
      <td>0.250000</td>
      <td>0.191489</td>
      <td>0.263158</td>
      <td>0.093023</td>
      <td>0.153846</td>
      <td>0.480000</td>
      <td>0.565217</td>
      <td>0.531915</td>
      <td>0.728111</td>
      <td>0.660469</td>
      <td>0.537986</td>
      <td>0.374451</td>
      <td>0.386376</td>
      <td>0.053598</td>
      <td>0.030501</td>
    </tr>
  </tbody>
</table>
<p>5296 rows × 16 columns</p>
</div>




```python
# Get only selected features based on correlation values
list_of_columns = list(X_train_scaled.columns)

X_selected_features = X[X.columns.intersection(list_of_columns)]
```


```python
X_selected_features
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
      <th>g1_1</th>
      <th>g1_2</th>
      <th>g1_3</th>
      <th>g1_5</th>
      <th>g1_7</th>
      <th>g1_9</th>
      <th>g1_10</th>
      <th>g1_11</th>
      <th>g1_12</th>
      <th>g1_13</th>
      <th>g1_15</th>
      <th>g1_16</th>
      <th>g1_17</th>
      <th>g1_18</th>
      <th>g1_19</th>
      <th>g1_20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.50</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>-2</td>
      <td>-2</td>
      <td>2.505032</td>
      <td>6.240000</td>
      <td>3.608000</td>
      <td>0.744000</td>
      <td>1.216000</td>
      <td>0.003078</td>
      <td>0.003028</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.20</td>
      <td>24</td>
      <td>22</td>
      <td>10</td>
      <td>28</td>
      <td>22</td>
      <td>-4</td>
      <td>-4</td>
      <td>-8</td>
      <td>3.718983</td>
      <td>7.423435</td>
      <td>5.048030</td>
      <td>0.836178</td>
      <td>1.975244</td>
      <td>0.784882</td>
      <td>0.019448</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.00</td>
      <td>7</td>
      <td>4</td>
      <td>18</td>
      <td>11</td>
      <td>10</td>
      <td>-3</td>
      <td>-8</td>
      <td>-11</td>
      <td>2.244550</td>
      <td>11.091399</td>
      <td>5.853005</td>
      <td>0.730046</td>
      <td>2.022004</td>
      <td>0.043937</td>
      <td>0.014264</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.91</td>
      <td>8</td>
      <td>5</td>
      <td>14</td>
      <td>7</td>
      <td>9</td>
      <td>-1</td>
      <td>-3</td>
      <td>-4</td>
      <td>2.580190</td>
      <td>9.864426</td>
      <td>2.582357</td>
      <td>0.656638</td>
      <td>1.407549</td>
      <td>0.041563</td>
      <td>0.021386</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.50</td>
      <td>23</td>
      <td>16</td>
      <td>14</td>
      <td>25</td>
      <td>27</td>
      <td>8</td>
      <td>-9</td>
      <td>-1</td>
      <td>3.470617</td>
      <td>11.672962</td>
      <td>4.554560</td>
      <td>1.895740</td>
      <td>1.237122</td>
      <td>0.941241</td>
      <td>0.000062</td>
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
      <th>6615</th>
      <td>1.30</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>7</td>
      <td>3.225437</td>
      <td>8.128000</td>
      <td>3.584000</td>
      <td>1.904000</td>
      <td>0.728000</td>
      <td>0.002832</td>
      <td>0.003089</td>
    </tr>
    <tr>
      <th>6616</th>
      <td>1.85</td>
      <td>19</td>
      <td>12</td>
      <td>6</td>
      <td>9</td>
      <td>14</td>
      <td>6</td>
      <td>1</td>
      <td>7</td>
      <td>3.802635</td>
      <td>12.445367</td>
      <td>3.772041</td>
      <td>1.275865</td>
      <td>1.215939</td>
      <td>0.090605</td>
      <td>0.064248</td>
    </tr>
    <tr>
      <th>6617</th>
      <td>2.50</td>
      <td>5</td>
      <td>8</td>
      <td>19</td>
      <td>8</td>
      <td>10</td>
      <td>-5</td>
      <td>2</td>
      <td>-3</td>
      <td>2.639729</td>
      <td>12.735732</td>
      <td>6.930399</td>
      <td>1.018920</td>
      <td>1.422971</td>
      <td>0.056135</td>
      <td>0.007597</td>
    </tr>
    <tr>
      <th>6618</th>
      <td>1.80</td>
      <td>5</td>
      <td>4</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>2.733488</td>
      <td>11.668416</td>
      <td>4.200704</td>
      <td>0.721472</td>
      <td>0.597312</td>
      <td>0.007508</td>
      <td>0.004967</td>
    </tr>
    <tr>
      <th>6619</th>
      <td>1.95</td>
      <td>35</td>
      <td>37</td>
      <td>1</td>
      <td>15</td>
      <td>16</td>
      <td>11</td>
      <td>15</td>
      <td>26</td>
      <td>5.000000</td>
      <td>12.919765</td>
      <td>4.544010</td>
      <td>1.411588</td>
      <td>0.304650</td>
      <td>0.006134</td>
      <td>0.923116</td>
    </tr>
  </tbody>
</table>
<p>6620 rows × 16 columns</p>
</div>




```python
# Resample using only selected features
X_train, X_test, y_train, y_test = train_test_split(X_selected_features, y, test_size = 0.2, 
                                                    random_state = 42)

clf = DecisionTreeClassifier(max_depth = 5, random_state = 0)
clf.fit(X_train, y_train)
dtree_predictions = clf.predict(X_test)
multilabel_confusion_matrix(y_test, dtree_predictions)
```




    array([[[313, 291],
            [169, 551]],
    
           [[551, 169],
            [291, 313]]])



Conclusions:

Removing correlated columns gave us much more close results compared to the the original dataframe. This shows than in some cases simplicity of the model allows avoidance of overfitting.

# Second approach: try to predict if the campaign should be launched (based on both groups)

Basing on both groups, I have created the model using the GradientBoostingClassifier to **predict if the campaign should be launched for the group (one of them, or none of them)**. Additionally, minmaxscaling and removing columns g1_XX have been used if the correlation between them is very high (>0.9) for reduction of dimensionality in a similar way as two cases before.


```python
# Get first and second group in different df
df_first_group = campaign.filter(regex = '^(?!g1_21)g1_')
df_second_group = campaign.filter(regex = '^(?!g2_21)g2_');
df_second_group.columns = df_first_group.columns

# Append second group to the first one
df_both_groups = df_first_group.append(df_second_group, ignore_index=True)
```


```python
"""
Get dummy variable as the column of the first group success.
y is the successful campaign of the both group (our target in this approach)
Append second group target (campaign[2]) to the first one (campaign[1])
"""
y = campaign[1] 
y = y.append(campaign[2], ignore_index=True)
y.columns = 'y'
df_both_groups['y'] = y
```


```python
df_both_groups
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
      <th>g1_1</th>
      <th>g1_2</th>
      <th>g1_3</th>
      <th>g1_4</th>
      <th>g1_5</th>
      <th>g1_6</th>
      <th>g1_7</th>
      <th>g1_8</th>
      <th>g1_9</th>
      <th>g1_10</th>
      <th>g1_11</th>
      <th>g1_12</th>
      <th>g1_13</th>
      <th>g1_14</th>
      <th>g1_15</th>
      <th>g1_16</th>
      <th>g1_17</th>
      <th>g1_18</th>
      <th>g1_19</th>
      <th>g1_20</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.50</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>-2</td>
      <td>-2</td>
      <td>2.505032</td>
      <td>2.551406</td>
      <td>6.240000</td>
      <td>3.608000</td>
      <td>0.744000</td>
      <td>1.216000</td>
      <td>0.003078</td>
      <td>0.003028</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.20</td>
      <td>24</td>
      <td>22</td>
      <td>46</td>
      <td>10</td>
      <td>24</td>
      <td>28</td>
      <td>18</td>
      <td>22</td>
      <td>-4</td>
      <td>-4</td>
      <td>-8</td>
      <td>3.718983</td>
      <td>3.882271</td>
      <td>7.423435</td>
      <td>5.048030</td>
      <td>0.836178</td>
      <td>1.975244</td>
      <td>0.784882</td>
      <td>0.019448</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.00</td>
      <td>7</td>
      <td>4</td>
      <td>11</td>
      <td>18</td>
      <td>8</td>
      <td>11</td>
      <td>2</td>
      <td>10</td>
      <td>-3</td>
      <td>-8</td>
      <td>-11</td>
      <td>2.244550</td>
      <td>2.458087</td>
      <td>11.091399</td>
      <td>5.853005</td>
      <td>0.730046</td>
      <td>2.022004</td>
      <td>0.043937</td>
      <td>0.014264</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.91</td>
      <td>8</td>
      <td>5</td>
      <td>13</td>
      <td>14</td>
      <td>6</td>
      <td>7</td>
      <td>6</td>
      <td>9</td>
      <td>-1</td>
      <td>-3</td>
      <td>-4</td>
      <td>2.580190</td>
      <td>2.683092</td>
      <td>9.864426</td>
      <td>2.582357</td>
      <td>0.656638</td>
      <td>1.407549</td>
      <td>0.041563</td>
      <td>0.021386</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.50</td>
      <td>23</td>
      <td>16</td>
      <td>39</td>
      <td>14</td>
      <td>33</td>
      <td>25</td>
      <td>18</td>
      <td>27</td>
      <td>8</td>
      <td>-9</td>
      <td>-1</td>
      <td>3.470617</td>
      <td>3.055989</td>
      <td>11.672962</td>
      <td>4.554560</td>
      <td>1.895740</td>
      <td>1.237122</td>
      <td>0.941241</td>
      <td>0.000062</td>
      <td>0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>13235</th>
      <td>12.00</td>
      <td>6</td>
      <td>3</td>
      <td>9</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>3.118825</td>
      <td>2.701205</td>
      <td>2.912000</td>
      <td>1.016000</td>
      <td>0.128000</td>
      <td>0.002832</td>
      <td>0.002953</td>
      <td>0.555725</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13236</th>
      <td>4.50</td>
      <td>9</td>
      <td>8</td>
      <td>17</td>
      <td>16</td>
      <td>13</td>
      <td>16</td>
      <td>5</td>
      <td>12</td>
      <td>-3</td>
      <td>-7</td>
      <td>-10</td>
      <td>2.352446</td>
      <td>2.515035</td>
      <td>4.177159</td>
      <td>0.761815</td>
      <td>1.741461</td>
      <td>0.197213</td>
      <td>0.002953</td>
      <td>0.306024</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13237</th>
      <td>2.88</td>
      <td>6</td>
      <td>13</td>
      <td>19</td>
      <td>7</td>
      <td>7</td>
      <td>12</td>
      <td>12</td>
      <td>9</td>
      <td>-5</td>
      <td>3</td>
      <td>-2</td>
      <td>3.009765</td>
      <td>2.715531</td>
      <td>5.770674</td>
      <td>1.372201</td>
      <td>1.070904</td>
      <td>0.051467</td>
      <td>0.021068</td>
      <td>0.524975</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13238</th>
      <td>5.25</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>15</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>-1</td>
      <td>-2</td>
      <td>-3</td>
      <td>2.419853</td>
      <td>2.488684</td>
      <td>2.804288</td>
      <td>0.760320</td>
      <td>1.288640</td>
      <td>0.008076</td>
      <td>0.004472</td>
      <td>0.288826</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13239</th>
      <td>4.20</td>
      <td>28</td>
      <td>25</td>
      <td>53</td>
      <td>6</td>
      <td>27</td>
      <td>19</td>
      <td>27</td>
      <td>23</td>
      <td>8</td>
      <td>4</td>
      <td>12</td>
      <td>4.650996</td>
      <td>4.454084</td>
      <td>5.036473</td>
      <td>2.008953</td>
      <td>1.752584</td>
      <td>0.199638</td>
      <td>0.124930</td>
      <td>0.309187</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>13240 rows × 21 columns</p>
</div>




```python
# Sample train and test chunks
X = df_both_groups.filter(regex = '^g1_')
y = df_both_groups.loc[:, df_both_groups.columns == 'y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```


```python
# Minmaxscale on the train dataset
X_train = scale_values(X_train)
```


```python
X_train, features_to_drop = drop_correlated_features(X_train)
```


```python
X_train
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
      <th>g1_1</th>
      <th>g1_2</th>
      <th>g1_3</th>
      <th>g1_5</th>
      <th>g1_7</th>
      <th>g1_9</th>
      <th>g1_10</th>
      <th>g1_11</th>
      <th>g1_12</th>
      <th>g1_13</th>
      <th>g1_15</th>
      <th>g1_16</th>
      <th>g1_17</th>
      <th>g1_18</th>
      <th>g1_19</th>
      <th>g1_20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.051186</td>
      <td>0.200000</td>
      <td>0.212766</td>
      <td>0.578947</td>
      <td>0.232558</td>
      <td>0.192308</td>
      <td>0.346667</td>
      <td>0.536232</td>
      <td>0.440559</td>
      <td>0.603098</td>
      <td>0.321472</td>
      <td>0.073915</td>
      <td>0.233640</td>
      <td>0.038948</td>
      <td>0.005946</td>
      <td>0.494133</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.051186</td>
      <td>0.363636</td>
      <td>0.361702</td>
      <td>0.631579</td>
      <td>0.302326</td>
      <td>0.442308</td>
      <td>0.413333</td>
      <td>0.507246</td>
      <td>0.461538</td>
      <td>0.702444</td>
      <td>0.282190</td>
      <td>0.094862</td>
      <td>0.411311</td>
      <td>0.197906</td>
      <td>0.000187</td>
      <td>0.509689</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.057428</td>
      <td>0.527273</td>
      <td>0.382979</td>
      <td>0.368421</td>
      <td>0.418605</td>
      <td>0.461538</td>
      <td>0.426667</td>
      <td>0.420290</td>
      <td>0.426573</td>
      <td>0.762853</td>
      <td>0.209630</td>
      <td>0.082662</td>
      <td>0.343944</td>
      <td>0.082729</td>
      <td>0.013774</td>
      <td>0.417014</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.023720</td>
      <td>0.290909</td>
      <td>0.042553</td>
      <td>0.578947</td>
      <td>0.279070</td>
      <td>0.423077</td>
      <td>0.360000</td>
      <td>0.376812</td>
      <td>0.370629</td>
      <td>0.497270</td>
      <td>0.460963</td>
      <td>0.407735</td>
      <td>0.393238</td>
      <td>0.529657</td>
      <td>0.128022</td>
      <td>0.018316</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.035456</td>
      <td>0.181818</td>
      <td>0.085106</td>
      <td>0.473684</td>
      <td>0.069767</td>
      <td>0.134615</td>
      <td>0.413333</td>
      <td>0.536232</td>
      <td>0.475524</td>
      <td>0.631968</td>
      <td>0.480408</td>
      <td>0.420553</td>
      <td>0.398432</td>
      <td>0.402851</td>
      <td>0.014114</td>
      <td>0.008148</td>
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
      <th>10587</th>
      <td>0.047940</td>
      <td>0.127273</td>
      <td>0.021277</td>
      <td>0.736842</td>
      <td>0.139535</td>
      <td>0.096154</td>
      <td>0.400000</td>
      <td>0.521739</td>
      <td>0.461538</td>
      <td>0.544786</td>
      <td>0.254946</td>
      <td>0.117313</td>
      <td>0.284131</td>
      <td>0.003487</td>
      <td>0.005407</td>
      <td>0.516484</td>
    </tr>
    <tr>
      <th>10588</th>
      <td>0.019226</td>
      <td>0.072727</td>
      <td>0.148936</td>
      <td>0.578947</td>
      <td>0.139535</td>
      <td>0.096154</td>
      <td>0.306667</td>
      <td>0.579710</td>
      <td>0.440559</td>
      <td>0.545162</td>
      <td>0.479694</td>
      <td>0.446075</td>
      <td>0.303471</td>
      <td>0.408376</td>
      <td>0.014462</td>
      <td>0.006409</td>
    </tr>
    <tr>
      <th>10589</th>
      <td>0.048689</td>
      <td>0.072727</td>
      <td>0.212766</td>
      <td>0.684211</td>
      <td>0.209302</td>
      <td>0.038462</td>
      <td>0.306667</td>
      <td>0.565217</td>
      <td>0.433566</td>
      <td>0.551292</td>
      <td>0.379544</td>
      <td>0.219643</td>
      <td>0.210401</td>
      <td>0.241557</td>
      <td>0.025318</td>
      <td>0.014264</td>
    </tr>
    <tr>
      <th>10590</th>
      <td>0.017728</td>
      <td>0.236364</td>
      <td>0.191489</td>
      <td>0.263158</td>
      <td>0.093023</td>
      <td>0.153846</td>
      <td>0.480000</td>
      <td>0.565217</td>
      <td>0.524476</td>
      <td>0.728111</td>
      <td>0.660469</td>
      <td>0.537986</td>
      <td>0.374451</td>
      <td>0.386376</td>
      <td>0.053598</td>
      <td>0.030501</td>
    </tr>
    <tr>
      <th>10591</th>
      <td>0.300874</td>
      <td>0.236364</td>
      <td>0.212766</td>
      <td>0.894737</td>
      <td>0.488372</td>
      <td>0.403846</td>
      <td>0.280000</td>
      <td>0.434783</td>
      <td>0.356643</td>
      <td>0.470302</td>
      <td>0.348022</td>
      <td>0.083209</td>
      <td>0.386636</td>
      <td>0.152257</td>
      <td>0.000747</td>
      <td>0.572742</td>
    </tr>
  </tbody>
</table>
<p>10592 rows × 16 columns</p>
</div>




```python
# Create classifier for the second approach
clf = DecisionTreeClassifier(max_depth = 6, random_state = 0)
clf.fit(X_train, y_train)
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=6, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=0, splitter='best')



After fitting, let's rescale using the test dataset with the train set and then dropping the train set


```python
X_scaled = scale_values(X)
```


```python
# Get only rows with testset
X_test = X_scaled.iloc[X_test.index]
```


```python
# Drop features 
X_test.drop(features_to_drop, axis=1, inplace=True)

clf_predictions = clf.predict(X_test)
multilabel_confusion_matrix(y_test, clf_predictions)
```

    /usr/local/lib/python3.7/dist-packages/pandas/core/frame.py:4174: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      errors=errors,
    




    array([[[ 469,  521],
            [ 279, 1379]],
    
           [[1379,  279],
            [ 521,  469]]])



Accuracy on the combined first and the second group is equal to 69.78%. For 469 cases, the campaign was rightly not released. In contrast, for 1379 occurrences, it was rightly released. For 521 examples, the promotion was not launched even though it would have been profitable, and for 279 cases, the marketing campaign was started even though it should not have been.


```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Try another classifier: GradientBoosting with the GridSearchCV to find best params
gbc = GradientBoostingClassifier()
params = {'max_depth':[2, 3, 5, 7],
         'max_leaf_nodes': [3, 4, 5, 6, 7, 8],
         'random_state': [0]}
gs_gbc = GridSearchCV(gbc,
                      param_grid=params,
                      scoring = 'accuracy',
                      cv = 5)
gs_gbc.fit(X_train, y_train.values.ravel())
gs_gbc.score(X_train, y_train)
```




    0.7128965256797583




```python
gs_gbc.best_params_
```




    {'max_depth': 3, 'max_leaf_nodes': 4, 'random_state': 0}




```python
gbc = GradientBoostingClassifier(max_depth = 3, max_features = 'auto', 
                                 max_leaf_nodes = 4, random_state = 0)
gbc.fit(X_train, y_train)

X_scaled = scale_values(X)
X_test = X_scaled.iloc[X_test.index]

# Drop features 
X_test.drop(features_to_drop, axis = 1, inplace = True)

dtree_predictions = gbc.predict(X_test)
multilabel_confusion_matrix(y_test, dtree_predictions)
```

    /usr/local/lib/python3.7/dist-packages/sklearn/ensemble/_gb.py:1454: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    /usr/local/lib/python3.7/dist-packages/pandas/core/frame.py:4174: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      errors=errors,
    




    array([[[ 443,  547],
            [ 217, 1441]],
    
           [[1441,  217],
            [ 547,  443]]])



Conclusions:

Accuracy agains combined groups is equal to 71.14%. For 443 cases, the campaign shoulnd't be released. In contrast, for 1441 occurrences, it was released correctly. For 547 examples, the promotion was not launched even though it would have been profitable, and for 217 cases, the marketing campaign was started even though it should not have been.

# Third approach: try to predict which group should we target

Basing on both groups, I have created the model using the GradientBoostingClassifier to **predict if the campaign should be launched for the group (one of them, or none of them)**. Additionally, minmaxscaling and removing columns g1_XX have been used if the correlation between them is very high (>0.9) for reduction of dimensionality in a similar way as two cases before.



```python
# X is the first and second group features
df_groups = campaign.filter(regex = '^(?!g1_21)g1_|g2_')
X = df_groups.filter(regex = '^(?!g2_21)')

# y is the successful variable of the first and second group
y = campaign[[1,2]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```


```python
# For the training set get proper groups

def train_set_append_one_group_into_another(X_train: pd.DataFrame):
  """
  To the first group append second group, to store in the same columns as in the first group.
  This is the part for the training set.

  """

  g1_Xtrain = X_train.filter(regex = '^g1_')
  g2_Xtrain = X_train.filter(regex = '^g2_')
  g2_Xtrain.columns = g1_Xtrain.columns

  # To the first group append second group, to store in the same columns as in the first group
  g12_Xtrain = g1_Xtrain.append(g2_Xtrain, ignore_index = True)
  g12_Xtrain_minmax = scale_values(g12_Xtrain)

  # Drop features 
  g12_Xtrain_minmax.drop(features_to_drop, axis = 1, inplace = True)
  g12_Xtrain_minmax

  return g12_Xtrain_minmax, g12_Xtrain

g12_Xtrain_minmax, g12_Xtrain = train_set_append_one_group_into_another(X_train)
```


```python
# Save dropped features for further API purposes
g12_Xtrain.to_json(r'/content/gdrive/My Drive/customer_segments_predicting/g12_Xtrain.json')
```


```python
# To the first group append second group target, to store in the same columns as in the first group
y = y_train[1].append(y_train[2], ignore_index=True)
```


```python
gbc = GradientBoostingClassifier(max_depth = 3, max_features = 'auto', 
                                 max_leaf_nodes = 4, random_state = 0)
gbc.fit(g12_Xtrain_minmax, y)
```




    GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                               learning_rate=0.1, loss='deviance', max_depth=3,
                               max_features='auto', max_leaf_nodes=4,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=100,
                               n_iter_no_change=None, presort='deprecated',
                               random_state=0, subsample=1.0, tol=0.0001,
                               validation_fraction=0.1, verbose=0,
                               warm_start=False)




```python
from sklearn.externals import joblib
# Export model to pickle
joblib.dump(clf, '/content/gdrive/My Drive/customer_segments_predicting/model_third.pkl', compress = 1)
```

    /usr/local/lib/python3.7/dist-packages/sklearn/externals/joblib/__init__.py:15: FutureWarning: sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. Please import this functionality directly from joblib, which can be installed with: pip install joblib. If this warning is raised when loading pickled models, you may need to re-serialize those models with scikit-learn 0.21+.
      warnings.warn(msg, category=FutureWarning)
    




    ['/content/gdrive/My Drive/customer_segments_predicting/model_third.pkl']




```python
def test_set_append_one_group_into_another(X_test: pd.DataFrame, g12_Xtrain: pd.DataFrame):
  """
  To the first group append second group, to store in the same columns as in the first group.
  This is the part for the test set.

  """
  g1_Xtest = X_test.filter(regex = '^g1_')
  g2_Xtest = X_test.filter(regex = '^g2_')
  g2_Xtest.columns = g1_Xtest.columns
  g12_Xtest = g1_Xtest.append(g2_Xtest, ignore_index = True)

  indexes = g12_Xtest.index.copy()

  # Add training set to make proper minmaxscaling
  g12_Xtest = g12_Xtest.append(g12_Xtrain, ignore_index = True)
  g12_Xtest_minmax = scale_values(g12_Xtest)

  # Drop rows from the training set
  g12_Xtest_minmax = g12_Xtest_minmax.iloc[indexes] 

  # Drop features 
  g12_Xtest_minmax.drop(features_to_drop, axis = 1, inplace = True)
  g12_Xtest_minmax

  return g12_Xtest_minmax

g12_Xtest_minmax =  test_set_append_one_group_into_another(X_test, g12_Xtrain)
```


```python
# To the first group append second group target, to store in the same columns as in the first group
y = y_test[1].append(y_test[2], ignore_index = True)

gbc_predictions = gbc.predict(g12_Xtest_minmax)
multilabel_confusion_matrix(y, gbc_predictions)
```




    array([[[ 459,  537],
            [ 230, 1422]],
    
           [[1422,  230],
            [ 537,  459]]])




```python
prediction_probability = gbc.predict_proba(g12_Xtest_minmax)
prediction_probability = pd.DataFrame(prediction_probability)
```

Which group to target?


```python
# Get probability from the first group and the second for the same campaign 
prediction_probability.iloc[0::int(y.size/2), :]
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.664091</td>
      <td>0.335909</td>
    </tr>
    <tr>
      <th>1324</th>
      <td>0.649775</td>
      <td>0.350225</td>
    </tr>
  </tbody>
</table>
</div>




```python
def dividing_test_set_into_two_groups(prediction_probability: pd.DataFrame, y: pd.DataFrame):
  """
  Dividing the set into two group. As is in the original frame.

  """
  # Divide the appended test set into group 1 and 2

  group_1_prediction_result = prediction_probability.iloc[:int(y.size/2), :]
  group_2_prediction_result = prediction_probability.iloc[int(y.size/2):, :].reset_index()

  group_2_prediction_result.drop('index', axis = 1, inplace = True)
  group_2_prediction_result.columns=['g2_0', 'g2_1']
  
  joined_results = group_1_prediction_result.join(group_2_prediction_result)

  return joined_results

joined_results = dividing_test_set_into_two_groups(prediction_probability, y)
```


```python
def prediction_probability_process(prediction_probability: pd.DataFrame,
                                   threshold: float) -> pd.DataFrame:
    """
    Get the target by the comparison of the probability for the first group and 
    the second. Additionally, the threshold has been used if the probability 
    is low. In this case, none of the target groups should be used for marketing purposes.
    """

    mask = [(prediction_probability[1] >= prediction_probability['g2_1']) & 
            (prediction_probability[1] > threshold),
            (prediction_probability[1] < prediction_probability['g2_1']) & 
            (prediction_probability['g2_1'] > threshold)]
    prediction_probability['label'] = np.select(mask, [1, 2], default=0)

    return prediction_probability

joined_results = prediction_probability_process(prediction_probability = joined_results, 
                                                   threshold = 0.45)
```


```python
# Get the original label for target
y_target = campaign[['target']]

y_train, y_test = train_test_split(y_target, test_size = 0.2, random_state = 42)
multilabel_confusion_matrix(y_test, joined_results['label'], labels=[0, 1, 2])
```




    array([[[721, 275],
            [190, 138]],
    
           [[446, 274],
            [179, 425]],
    
           [[858,  74],
            [254, 138]]])




```python
# Copy the results from this approach, due to usage it in the forth approach
third_approach_y_test = joined_results['label'].copy()
```


```python
from sklearn.metrics import accuracy_score, precision_score
accuracy_score(y_test, joined_results['label'])
```




    0.5294561933534743



Conclusions:

14 campaigns were not launched in comparison to the original dataset. It has turned out that 13 of these that not launching them was a wrong idea as they gave measurable benefits for both of the groups for each case.

When accuracy is higher than 50%, it means that the created model is more effective than launching campaigns for two groups at the same time. The assumption behind this is that the groups are equal, and thy share same actions costs. Otherwise, it is impossible to compare costs of the campaigns for a given case for both groups.

As it is not known exact moment when the gX_21 index is determined, it is hard to find out of its usefulness in prediction. If it was an indicator calculated shortly after the start of a campaign, it could be used to calculate the correlation with ROI values and used as a stopping marker for campaign run in case of poor results.

# Fourth approach: try to predict if the campaign should be stopped based on the gX_21 column

Basing on the third approach, an additional step has been added. In this case, **another model that tries to catch if the campaign should be stopped** is used. The assumption is that the gX_21 features have not been generated in the last stage of the campaign.


```python
def fourth_approach_train_preprocess(campaign: pd.DataFrame):
    """
    Due to the different scale in the g1_21 and g2_21 columns, the columns 
    should be scale separated. 

    """
    # X is the first and second group feature (unknown when the campaign start)
    X = campaign.filter(regex = '^g1_21|g2_21')

    # y is the successful variable of the first and second group
    y = campaign[[1, 2]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Scale one column, then the second
    g1_Xtrain = X_train.filter(regex = '^g1_')
    g1_Xtrain_minmax = scale_values(g1_Xtrain)
    g1_Xtrain_minmax = pd.DataFrame(g1_Xtrain_minmax)

    g2_Xtrain = X_train.filter(regex = '^g2_')
    g2_Xtrain_minmax = scale_values(g2_Xtrain)
    g2_Xtrain_minmax = pd.DataFrame(g2_Xtrain_minmax).rename(columns={"g2_21": "g1_21"})

    # Append the second group feature into the first one
    g12_Xtrain_minmax = g1_Xtrain_minmax.append(g2_Xtrain_minmax, ignore_index=True)

    y = y_train[1].append(y_train[2], ignore_index=True)

    return X, X_train, X_test, y_train, y_test, g12_Xtrain_minmax, y
```


```python
def fourth_approach_test_preprocess(X_test: pd.DataFrame,
                                    X: pd.DataFrame):
  """
  Due to the different scale in the g1_21 and g2_21 columns, the columns 
  should be scale separated. 

  """

  indexes = X_test.index
  #g12_Xtest_minmax = scale_values(X_test)
  g12_Xtest_minmax = scale_values(X)

  # Drop rows from the training set
  g12_Xtest_minmax = g12_Xtest_minmax.iloc[indexes]

  # Append g2_21 into g1_21
  g12_Xtest = g12_Xtest_minmax['g1_21'].append(g12_Xtest_minmax['g2_21'], ignore_index = True)
  g12_Xtest = pd.DataFrame(g12_Xtest, columns = ['g1_21'])

  y = y_test[1].append(y_test[2], ignore_index=True)

  return g12_Xtest, y
```

## GradientBoostingClassifier


```python
# Due to the different scale in the g1_21 and g2_21 columns, the columns should be scale separated. 
X, X_train, X_test, y_train, y_test, g12_Xtrain_minmax, y = fourth_approach_train_preprocess(campaign)
```


```python
g12_Xtrain_minmax
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
      <th>g1_21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.487666</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.551474</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.547860</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.503118</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.344538</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>10587</th>
      <td>0.054545</td>
    </tr>
    <tr>
      <th>10588</th>
      <td>0.048485</td>
    </tr>
    <tr>
      <th>10589</th>
      <td>0.042424</td>
    </tr>
    <tr>
      <th>10590</th>
      <td>0.054545</td>
    </tr>
    <tr>
      <th>10591</th>
      <td>0.060606</td>
    </tr>
  </tbody>
</table>
<p>10592 rows × 1 columns</p>
</div>




```python
gbc = GradientBoostingClassifier(random_state = 0)
gbc.fit(g12_Xtrain_minmax, y)
gbc.score(g12_Xtrain_minmax, y)
```




    0.6549282477341389




```python
g12_Xtest, y = fourth_approach_test_preprocess(X_test, X)

# If the label is zero, then the campaign should be stopped.
gbc_predictions = gbc.predict(g12_Xtest)
multilabel_confusion_matrix(y, gbc_predictions)
```




    array([[[  90,  906],
            [  91, 1561]],
    
           [[1561,   91],
            [ 906,   90]]])




```python
def prediction_probability_fourth_approach_processing(prediction_probability: pd.DataFrame) -> pd.DataFrame:
    """
    Get the target by the comparison of the probability for the first group and 
    the second. Additionally, the threshold has been used if the probability 
    is low. In this case, none of the target groups should be used for marketing purposes.
    """
    mask = [(prediction_probability[1] >= prediction_probability['g2_0']), # Don't stop campaign
         (prediction_probability[1] < prediction_probability['g2_0'])] 

    prediction_probability['label'] = np.select(mask, [1, 0], default=0)

    return prediction_probability
```


```python
prediction_probability = gbc.predict_proba(g12_Xtest)
prediction_probability = pd.DataFrame(prediction_probability)

joined_g_21 = dividing_test_set_into_two_groups(prediction_probability, y)
joined_g_21 = prediction_probability_fourth_approach_processing(joined_g_21)
```


```python
prediction_probability
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.538459</td>
      <td>0.461541</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.504122</td>
      <td>0.495878</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.524196</td>
      <td>0.475804</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.807927</td>
      <td>0.192073</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.541904</td>
      <td>0.458096</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2643</th>
      <td>0.671757</td>
      <td>0.328243</td>
    </tr>
    <tr>
      <th>2644</th>
      <td>0.661518</td>
      <td>0.338482</td>
    </tr>
    <tr>
      <th>2645</th>
      <td>0.701733</td>
      <td>0.298267</td>
    </tr>
    <tr>
      <th>2646</th>
      <td>0.701733</td>
      <td>0.298267</td>
    </tr>
    <tr>
      <th>2647</th>
      <td>0.701733</td>
      <td>0.298267</td>
    </tr>
  </tbody>
</table>
<p>2648 rows × 2 columns</p>
</div>




```python
# y is the target variable
y = campaign['target']

y_train, y_test = train_test_split(y, test_size = 0.2, random_state = 42)

# Copy to results to use it in further comparison 
y_target = y_test.copy()
y_target_original = y_test.copy()

multilabel_confusion_matrix(y_test, joined_g_21['label'], labels=[0, 1])
```




    array([[[  5, 991],
            [  1, 327]],
    
           [[716,   4],
            [602,   2]]])




```python
def processing_y_target(y_target: pd.DataFrame, 
                        third_y_test: pd.DataFrame, 
                        joined_g_21: pd.DataFrame) -> pd.DataFrame:
  """
  Get the results from the third and forth approach, to stop campaigns that 
  might be unsuccessful. If the campaign is predicted as group 1 or 2, then
  the check is made to the classifier of the gX_21 params. 

  """

  y_target_results = pd.DataFrame(y_target.reset_index().drop('index', axis=1))

  y_target_results['third'] = third_y_test
  y_target_results['label'] = joined_g_21['label']

  mask = [((y_target_results['third'] == 1) | (y_target_results['third'] == 2) ) & 
          (y_target_results['label'] == 0), # Stop campaign
          (y_target_results['third'] == 1),
          (y_target_results['third'] == 2)] 
  y_target_results['label'] = np.select(mask, [0, 1, 2], default=0)

  return y_target_results
```


```python
joined_g_21 = processing_y_target(y_target=y_target, 
                                  third_y_test=third_approach_y_test, 
                                  joined_g_21=joined_g_21)
```


```python
multilabel_confusion_matrix(y_target_original, joined_g_21['label'], labels=[0, 1, 2])
```




    array([[[  4, 992],
            [  1, 327]],
    
           [[718,   2],
            [602,   2]],
    
           [[932,   0],
            [391,   1]]])




```python
accuracy_score(y_target_original, joined_g_21['label'])
```




    0.24924471299093656




```python
precision_score(y_target_original, joined_g_21['label'], average='weighted')
```




    0.5855862149527359



## LogisticRegression

https://i.pinimg.com/originals/f6/dc/ab/f6dcabbfe346ea36f4a71e60542657bc.jpg


```python
# Trying to get better results by the LogisticRegression
from sklearn.linear_model import LogisticRegression

X, X_train, X_test, y_train, y_test, g12_Xtrain_minmax, y = fourth_approach_train_preprocess(campaign)

logreg = LogisticRegression()
params = {'class_weight': [{ 0:0.5, 1:0.8 }, { 0:0.3, 1:0.8 }, { 0:1.0, 1:0.8 }, 
                           { 0:1.2, 1:0.8 }, { 0:1.5, 1:0.8 }, { 0:1.8, 1:0.8 }, 
                           { 0:3, 1:0.8 }],
          'random_state': [0]}

# By using gridsearchcv, try to find best params       
gs_logreg = GridSearchCV(logreg,
                      param_grid = params,
                      scoring = 'precision',
                      cv = 5)
gs_logreg.fit(g12_Xtrain_minmax, y)
gs_logreg.score(g12_Xtrain_minmax, y)
```

    0.4730785931393834




```python
gs_logreg.best_params_
```




    {'class_weight': {0: 0.5, 1: 0.8}, 'random_state': 0}




```python
logreg = LogisticRegression(class_weight = {0: 0.5, 1: 0.8}, random_state = 0)
logreg.fit(g12_Xtrain_minmax, y)

g12_Xtest, y = fourth_approach_test_preprocess(X_test, X)

logreg_predictions = logreg.predict(g12_Xtest)
multilabel_confusion_matrix(y, logreg_predictions)
```




    array([[[ 535,  461],
            [ 637, 1015]],
    
           [[1015,  637],
            [ 461,  535]]])




```python
prediction_probability = logreg.predict_proba(g12_Xtest)
prediction_probability = pd.DataFrame(prediction_probability)

joined_g_21 = dividing_test_set_into_two_groups(prediction_probability, y)
joined_g_21 = prediction_probability_fourth_approach_processing(joined_g_21)

joined_g_21 = processing_y_target(y_target=y_target, 
                                  third_y_test=third_approach_y_test, 
                                  joined_g_21=joined_g_21)
```


```python
multilabel_confusion_matrix(y_target_original, joined_g_21['label'], labels=[0, 1, 2])
```




    array([[[238, 758],
            [ 62, 266]],
    
           [[640,  80],
            [441, 163]],
    
           [[913,  19],
            [354,  38]]])




```python
accuracy_score(y_target_original, joined_g_21['label'])
```




    0.3527190332326284




```python
precision_score(y_target_original, joined_g_21['label'], average='weighted')
```




    0.5677407263654222



## KNN


```python
# Trying to get better results by the KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

X, X_train, X_test, y_train, y_test, g12_Xtrain_minmax, y = fourth_approach_train_preprocess(campaign)

knn = KNeighborsClassifier(20)
knn.fit(g12_Xtrain_minmax, y)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=20, p=2,
                         weights='uniform')




```python
g12_Xtest, y = fourth_approach_test_preprocess(X_test, X)

knn_predictions = knn.predict(g12_Xtest)
multilabel_confusion_matrix(y, knn_predictions)
```




    array([[[ 219,  777],
            [ 231, 1421]],
    
           [[1421,  231],
            [ 777,  219]]])




```python
prediction_probability = knn.predict_proba(g12_Xtest)
prediction_probability = pd.DataFrame(prediction_probability)

joined_g_21 = dividing_test_set_into_two_groups(prediction_probability, y)
joined_g_21 = prediction_probability_fourth_approach_processing(joined_g_21)

joined_g_21 = processing_y_target(y_target=y_target, 
                                  third_y_test=third_approach_y_test, 
                                  joined_g_21=joined_g_21)

multilabel_confusion_matrix(y_target, third_approach_y_test, labels=[0, 1, 2])
```




    array([[[721, 275],
            [190, 138]],
    
           [[446, 274],
            [179, 425]],
    
           [[858,  74],
            [254, 138]]])




```python
accuracy_score(y_target, joined_g_21['label'])
```




    0.2756797583081571




```python
precision_score(y_target, joined_g_21['label'], average='weighted')
```




    0.5467943162033351



Conclusions

For the three classification methods selected, the best obtained accuracy was for logistic regression. Achieved precision is above 0.5. It should be borne in mind that campaign discontinuation is an aggressive operation. With an increased threshold, we obtain more possible exits for campaigns that were misplaced. However, based on further research, it appears that out of 3 closed advertising campaigns, one was rightly closed, and others were not. This method can be used for conducting risk-free campaigns. For example when we minimize the risk of unsuccessful results and do not agree to keep running a campaign that does not have a high level of predicted success.


# Fifth approach: use additional parameters c_XX to predict the campaign target group

This approach is the connection of the third one and c params. In this case, the c_28 feature is not used (prior is it unknown when the campaign is launched). The created model was made by using the GradientBoostingClassifier to **predict if the campaign should be launched for the group (one of them, or none of them)**

Preprocessing c_XX params


```python
# X is the difference between first and second group feature
X = campaign.filter(regex = '^g1_21|g2_21')

# y is the target variable
X = campaign.filter(regex = '^(?!c_28)c_')
y = campaign[['target']]

c_Xtrain, c_Xtest, c_ytrain, c_ytest = train_test_split(X, y, test_size = 0.2, random_state = 42)
```


```python
# c_ parameters process
c_Xtrain_minmax = scale_values(c_Xtrain)
c_Xtrain_minmax, features_to_drop_c = drop_correlated_features(c_Xtrain_minmax)
```


```python
c_Xtrain_minmax.describe()
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
      <th>c_1</th>
      <th>c_2</th>
      <th>c_3</th>
      <th>c_4</th>
      <th>c_5</th>
      <th>c_6</th>
      <th>c_7</th>
      <th>c_8</th>
      <th>c_9</th>
      <th>c_10</th>
      <th>c_11</th>
      <th>c_12</th>
      <th>c_13</th>
      <th>c_14</th>
      <th>c_15</th>
      <th>c_16</th>
      <th>c_22</th>
      <th>c_23</th>
      <th>c_24</th>
      <th>c_25</th>
      <th>c_26</th>
      <th>c_27</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
      <td>5296.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.649838</td>
      <td>0.502660</td>
      <td>0.501970</td>
      <td>0.379437</td>
      <td>0.184479</td>
      <td>0.201662</td>
      <td>0.186367</td>
      <td>0.200906</td>
      <td>0.366538</td>
      <td>0.469635</td>
      <td>0.530020</td>
      <td>0.450418</td>
      <td>0.464033</td>
      <td>0.555183</td>
      <td>0.453239</td>
      <td>0.527277</td>
      <td>0.476340</td>
      <td>0.525231</td>
      <td>0.500661</td>
      <td>0.540350</td>
      <td>0.514888</td>
      <td>0.529449</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.078580</td>
      <td>0.175948</td>
      <td>0.174259</td>
      <td>0.162854</td>
      <td>0.387911</td>
      <td>0.401279</td>
      <td>0.389439</td>
      <td>0.400716</td>
      <td>0.227989</td>
      <td>0.122383</td>
      <td>0.113747</td>
      <td>0.107156</td>
      <td>0.105564</td>
      <td>0.123941</td>
      <td>0.101785</td>
      <td>0.098069</td>
      <td>0.125665</td>
      <td>0.115529</td>
      <td>0.082022</td>
      <td>0.122678</td>
      <td>0.114175</td>
      <td>0.066214</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.601375</td>
      <td>0.391304</td>
      <td>0.391304</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.170619</td>
      <td>0.405797</td>
      <td>0.475000</td>
      <td>0.400000</td>
      <td>0.408451</td>
      <td>0.500000</td>
      <td>0.403226</td>
      <td>0.486842</td>
      <td>0.394525</td>
      <td>0.455303</td>
      <td>0.500000</td>
      <td>0.468022</td>
      <td>0.444111</td>
      <td>0.490484</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.652921</td>
      <td>0.478261</td>
      <td>0.478261</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.377950</td>
      <td>0.449275</td>
      <td>0.550000</td>
      <td>0.451852</td>
      <td>0.464789</td>
      <td>0.557692</td>
      <td>0.451613</td>
      <td>0.526316</td>
      <td>0.477129</td>
      <td>0.524228</td>
      <td>0.500000</td>
      <td>0.544114</td>
      <td>0.519365</td>
      <td>0.529412</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.701031</td>
      <td>0.608696</td>
      <td>0.608696</td>
      <td>0.458333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.562930</td>
      <td>0.536232</td>
      <td>0.600000</td>
      <td>0.496296</td>
      <td>0.507042</td>
      <td>0.615385</td>
      <td>0.500000</td>
      <td>0.578947</td>
      <td>0.554696</td>
      <td>0.598987</td>
      <td>0.500000</td>
      <td>0.614504</td>
      <td>0.580805</td>
      <td>0.568348</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# For the test set process to drop redundant columns
indexes = c_Xtest.index

# Add training set to the test, to make better scaling
c_test = c_Xtest.append(c_Xtrain, ignore_index = True)
c_Xtest_minmax = scale_values(c_test)

# Drop rows from the training set
c_Xtest_minmax = c_Xtest_minmax.iloc[indexes] 

# Drop features 
c_Xtest_minmax.drop(features_to_drop_c, axis = 1, inplace = True)
```


```python
c_Xtest_minmax
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
      <th>c_1</th>
      <th>c_2</th>
      <th>c_3</th>
      <th>c_4</th>
      <th>c_5</th>
      <th>c_6</th>
      <th>c_7</th>
      <th>c_8</th>
      <th>c_9</th>
      <th>c_10</th>
      <th>c_11</th>
      <th>c_12</th>
      <th>c_13</th>
      <th>c_14</th>
      <th>c_15</th>
      <th>c_16</th>
      <th>c_22</th>
      <th>c_23</th>
      <th>c_24</th>
      <th>c_25</th>
      <th>c_26</th>
      <th>c_27</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>96</th>
      <td>0.670103</td>
      <td>0.608696</td>
      <td>0.434783</td>
      <td>0.416667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.598998</td>
      <td>0.623188</td>
      <td>0.5500</td>
      <td>0.547445</td>
      <td>0.408451</td>
      <td>0.576923</td>
      <td>0.258065</td>
      <td>0.605263</td>
      <td>0.215120</td>
      <td>0.608343</td>
      <td>0.5</td>
      <td>0.609119</td>
      <td>0.574976</td>
      <td>0.525365</td>
    </tr>
    <tr>
      <th>994</th>
      <td>0.604811</td>
      <td>0.434783</td>
      <td>0.782609</td>
      <td>0.583333</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.531604</td>
      <td>0.391304</td>
      <td>0.5875</td>
      <td>0.452555</td>
      <td>0.549296</td>
      <td>0.480769</td>
      <td>0.661290</td>
      <td>0.355263</td>
      <td>0.711773</td>
      <td>0.291270</td>
      <td>0.5</td>
      <td>0.665812</td>
      <td>0.525615</td>
      <td>0.473125</td>
    </tr>
    <tr>
      <th>1400</th>
      <td>0.649485</td>
      <td>0.521739</td>
      <td>0.434783</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.607934</td>
      <td>0.405797</td>
      <td>0.5875</td>
      <td>0.459854</td>
      <td>0.464789</td>
      <td>0.557692</td>
      <td>0.451613</td>
      <td>0.526316</td>
      <td>0.456890</td>
      <td>0.524228</td>
      <td>0.5</td>
      <td>0.547542</td>
      <td>0.527949</td>
      <td>0.529412</td>
    </tr>
    <tr>
      <th>865</th>
      <td>0.608247</td>
      <td>0.695652</td>
      <td>0.391304</td>
      <td>0.458333</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.023845</td>
      <td>0.478261</td>
      <td>0.6250</td>
      <td>0.518248</td>
      <td>0.507042</td>
      <td>0.576923</td>
      <td>0.419355</td>
      <td>0.486842</td>
      <td>0.453445</td>
      <td>0.499239</td>
      <td>0.5</td>
      <td>0.752380</td>
      <td>0.696666</td>
      <td>0.621067</td>
    </tr>
    <tr>
      <th>6095</th>
      <td>0.707904</td>
      <td>0.478261</td>
      <td>0.434783</td>
      <td>0.291667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.460720</td>
      <td>0.492754</td>
      <td>0.6250</td>
      <td>0.525547</td>
      <td>0.436620</td>
      <td>0.634615</td>
      <td>0.387097</td>
      <td>0.539474</td>
      <td>0.423508</td>
      <td>0.544828</td>
      <td>0.5</td>
      <td>0.343235</td>
      <td>0.340730</td>
      <td>0.500116</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>4706</th>
      <td>0.707904</td>
      <td>0.478261</td>
      <td>0.652174</td>
      <td>0.500000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.464967</td>
      <td>0.405797</td>
      <td>0.5625</td>
      <td>0.445255</td>
      <td>0.366197</td>
      <td>0.615385</td>
      <td>0.596774</td>
      <td>0.605263</td>
      <td>0.601209</td>
      <td>0.543221</td>
      <td>0.5</td>
      <td>0.637394</td>
      <td>0.574294</td>
      <td>0.485308</td>
    </tr>
    <tr>
      <th>1272</th>
      <td>0.680412</td>
      <td>0.565217</td>
      <td>0.869565</td>
      <td>0.791667</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.142904</td>
      <td>0.565217</td>
      <td>0.5875</td>
      <td>0.540146</td>
      <td>0.760563</td>
      <td>0.442308</td>
      <td>0.725806</td>
      <td>0.500000</td>
      <td>0.639388</td>
      <td>0.412323</td>
      <td>0.5</td>
      <td>0.516780</td>
      <td>0.544211</td>
      <td>0.559017</td>
    </tr>
    <tr>
      <th>6309</th>
      <td>0.676976</td>
      <td>0.869565</td>
      <td>0.347826</td>
      <td>0.583333</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.567412</td>
      <td>0.420290</td>
      <td>0.6500</td>
      <td>0.503650</td>
      <td>0.450704</td>
      <td>0.326923</td>
      <td>0.451613</td>
      <td>0.486842</td>
      <td>0.399996</td>
      <td>0.414049</td>
      <td>1.0</td>
      <td>0.409835</td>
      <td>0.463331</td>
      <td>0.539959</td>
    </tr>
    <tr>
      <th>4479</th>
      <td>0.477663</td>
      <td>0.565217</td>
      <td>0.304348</td>
      <td>0.250000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.532125</td>
      <td>0.782609</td>
      <td>0.8625</td>
      <td>0.810219</td>
      <td>0.619718</td>
      <td>0.538462</td>
      <td>0.629032</td>
      <td>0.434211</td>
      <td>0.566057</td>
      <td>0.351052</td>
      <td>0.5</td>
      <td>0.629507</td>
      <td>0.669657</td>
      <td>0.581510</td>
    </tr>
    <tr>
      <th>6037</th>
      <td>0.615120</td>
      <td>0.869565</td>
      <td>0.478261</td>
      <td>0.708333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.543473</td>
      <td>0.536232</td>
      <td>0.4375</td>
      <td>0.437956</td>
      <td>0.718310</td>
      <td>0.250000</td>
      <td>0.580645</td>
      <td>0.328947</td>
      <td>0.580161</td>
      <td>0.479086</td>
      <td>0.5</td>
      <td>0.588110</td>
      <td>0.493371</td>
      <td>0.483194</td>
    </tr>
  </tbody>
</table>
<p>1324 rows × 22 columns</p>
</div>



Processing params g1 and g2


```python
df_groups = campaign.filter(regex = '^(?!g1_21)g1_|g2_')

# X is the first and second group features
X = df_groups.filter(regex = '^(?!g2_21)')

# y is the target variable
y = campaign['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# For the training set get proper groups as the one of the column in the dataframe
# to use it in modeling as the one of the feature
g1_Xtrain = X_train.filter(regex = '^g1_')
g1_Xtrain['group'] = 1
g2_Xtrain = X_train.filter(regex = '^g2_')
g2_Xtrain['group'] = 2
g2_Xtrain.columns = g1_Xtrain.columns
g12_Xtrain = g1_Xtrain.append(g2_Xtrain, ignore_index = True)

# Scale the value other for other than group column
g12_Xtrain_without_groups = g12_Xtrain.loc[:, g12_Xtrain.columns != 'group']
g12_Xtrain_without_groups_minmax = scale_values(g12_Xtrain_without_groups)

# Drop features by the correlation as in the previous approaches
g12_Xtrain_without_groups_minmax.drop(features_to_drop, axis = 1, inplace = True)
g12_Xtrain_without_groups_minmax['group'] = g12_Xtrain['group']

c_Xtest_minmax_appended = c_Xtrain_minmax.append(c_Xtrain_minmax, ignore_index = True)
y_train_appended=y_train.append(y_train, ignore_index = True)

result = pd.concat([g12_Xtrain_without_groups_minmax, c_Xtest_minmax_appended], axis = 1)
result
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:16: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      app.launch_new_instance()
    




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
      <th>g1_1</th>
      <th>g1_2</th>
      <th>g1_3</th>
      <th>g1_5</th>
      <th>g1_7</th>
      <th>g1_9</th>
      <th>g1_10</th>
      <th>g1_11</th>
      <th>g1_12</th>
      <th>g1_13</th>
      <th>g1_15</th>
      <th>g1_16</th>
      <th>g1_17</th>
      <th>g1_18</th>
      <th>g1_19</th>
      <th>g1_20</th>
      <th>group</th>
      <th>c_1</th>
      <th>c_2</th>
      <th>c_3</th>
      <th>c_4</th>
      <th>c_5</th>
      <th>c_6</th>
      <th>c_7</th>
      <th>c_8</th>
      <th>c_9</th>
      <th>c_10</th>
      <th>c_11</th>
      <th>c_12</th>
      <th>c_13</th>
      <th>c_14</th>
      <th>c_15</th>
      <th>c_16</th>
      <th>c_22</th>
      <th>c_23</th>
      <th>c_24</th>
      <th>c_25</th>
      <th>c_26</th>
      <th>c_27</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.031211</td>
      <td>0.090909</td>
      <td>0.085106</td>
      <td>0.315789</td>
      <td>0.023256</td>
      <td>0.134615</td>
      <td>0.373333</td>
      <td>0.478261</td>
      <td>0.426573</td>
      <td>0.518286</td>
      <td>0.378995</td>
      <td>0.302635</td>
      <td>0.110128</td>
      <td>0.357061</td>
      <td>0.010509</td>
      <td>0.006738</td>
      <td>1</td>
      <td>0.759450</td>
      <td>0.565217</td>
      <td>0.608696</td>
      <td>0.541667</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.463285</td>
      <td>0.463768</td>
      <td>0.5625</td>
      <td>0.466667</td>
      <td>0.436620</td>
      <td>0.557692</td>
      <td>0.435484</td>
      <td>0.500000</td>
      <td>0.430904</td>
      <td>0.439718</td>
      <td>0.50</td>
      <td>0.383686</td>
      <td>0.289097</td>
      <td>0.430257</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.006242</td>
      <td>0.581818</td>
      <td>0.489362</td>
      <td>0.105263</td>
      <td>0.209302</td>
      <td>0.307692</td>
      <td>0.613333</td>
      <td>0.695652</td>
      <td>0.657343</td>
      <td>0.987124</td>
      <td>0.875567</td>
      <td>0.790637</td>
      <td>0.456739</td>
      <td>0.247622</td>
      <td>0.169272</td>
      <td>0.522046</td>
      <td>1</td>
      <td>0.525773</td>
      <td>0.739130</td>
      <td>0.217391</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.602615</td>
      <td>0.695652</td>
      <td>0.6500</td>
      <td>0.637037</td>
      <td>0.577465</td>
      <td>0.519231</td>
      <td>0.370968</td>
      <td>0.500000</td>
      <td>0.460707</td>
      <td>0.458539</td>
      <td>0.50</td>
      <td>0.776924</td>
      <td>0.712731</td>
      <td>0.516251</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.038702</td>
      <td>0.072727</td>
      <td>0.021277</td>
      <td>0.789474</td>
      <td>0.139535</td>
      <td>0.096154</td>
      <td>0.360000</td>
      <td>0.521739</td>
      <td>0.440559</td>
      <td>0.495890</td>
      <td>0.396688</td>
      <td>0.355862</td>
      <td>0.306257</td>
      <td>0.424941</td>
      <td>0.008189</td>
      <td>0.005462</td>
      <td>1</td>
      <td>0.721649</td>
      <td>0.173913</td>
      <td>0.565217</td>
      <td>0.125000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.528451</td>
      <td>0.420290</td>
      <td>0.4875</td>
      <td>0.400000</td>
      <td>0.521127</td>
      <td>0.634615</td>
      <td>0.500000</td>
      <td>0.552632</td>
      <td>0.638856</td>
      <td>0.665150</td>
      <td>0.50</td>
      <td>0.595076</td>
      <td>0.569098</td>
      <td>0.541500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.011236</td>
      <td>0.127273</td>
      <td>0.212766</td>
      <td>0.052632</td>
      <td>0.046512</td>
      <td>0.038462</td>
      <td>0.386667</td>
      <td>0.608696</td>
      <td>0.496503</td>
      <td>0.688240</td>
      <td>0.589805</td>
      <td>0.485894</td>
      <td>0.351539</td>
      <td>0.178269</td>
      <td>0.008386</td>
      <td>0.010620</td>
      <td>1</td>
      <td>0.618557</td>
      <td>0.608696</td>
      <td>0.304348</td>
      <td>0.291667</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.061574</td>
      <td>0.449275</td>
      <td>0.6625</td>
      <td>0.518519</td>
      <td>0.492958</td>
      <td>0.480769</td>
      <td>0.483871</td>
      <td>0.513158</td>
      <td>0.605387</td>
      <td>0.471031</td>
      <td>0.50</td>
      <td>0.582383</td>
      <td>0.565722</td>
      <td>0.548148</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.053683</td>
      <td>0.054545</td>
      <td>0.085106</td>
      <td>0.736842</td>
      <td>0.186047</td>
      <td>0.230769</td>
      <td>0.293333</td>
      <td>0.449275</td>
      <td>0.370629</td>
      <td>0.410594</td>
      <td>0.341043</td>
      <td>0.192402</td>
      <td>0.188375</td>
      <td>0.866021</td>
      <td>0.014462</td>
      <td>0.007263</td>
      <td>1</td>
      <td>0.701031</td>
      <td>0.391304</td>
      <td>0.608696</td>
      <td>0.375000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.367451</td>
      <td>0.420290</td>
      <td>0.5875</td>
      <td>0.459259</td>
      <td>0.408451</td>
      <td>0.653846</td>
      <td>0.387097</td>
      <td>0.631579</td>
      <td>0.355344</td>
      <td>0.839257</td>
      <td>0.50</td>
      <td>0.437114</td>
      <td>0.470586</td>
      <td>0.544236</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>10587</th>
      <td>0.048190</td>
      <td>0.236364</td>
      <td>0.148936</td>
      <td>0.631579</td>
      <td>0.255814</td>
      <td>0.307692</td>
      <td>0.413333</td>
      <td>0.492754</td>
      <td>0.454545</td>
      <td>0.597994</td>
      <td>0.304135</td>
      <td>0.091550</td>
      <td>0.437615</td>
      <td>0.034129</td>
      <td>0.009804</td>
      <td>0.544515</td>
      <td>2</td>
      <td>0.632302</td>
      <td>0.434783</td>
      <td>0.521739</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.006704</td>
      <td>0.362319</td>
      <td>0.4625</td>
      <td>0.355556</td>
      <td>0.309859</td>
      <td>0.576923</td>
      <td>0.403226</td>
      <td>0.539474</td>
      <td>0.346838</td>
      <td>0.590610</td>
      <td>0.50</td>
      <td>0.576057</td>
      <td>0.579194</td>
      <td>0.562927</td>
    </tr>
    <tr>
      <th>10588</th>
      <td>0.088639</td>
      <td>0.054545</td>
      <td>0.063830</td>
      <td>0.842105</td>
      <td>0.093023</td>
      <td>0.096154</td>
      <td>0.333333</td>
      <td>0.521739</td>
      <td>0.426573</td>
      <td>0.473578</td>
      <td>0.248480</td>
      <td>0.034365</td>
      <td>0.189734</td>
      <td>0.004683</td>
      <td>0.004149</td>
      <td>0.575597</td>
      <td>2</td>
      <td>0.707904</td>
      <td>0.434783</td>
      <td>0.565217</td>
      <td>0.375000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.420290</td>
      <td>0.6375</td>
      <td>0.488889</td>
      <td>0.422535</td>
      <td>0.615385</td>
      <td>0.500000</td>
      <td>0.565789</td>
      <td>0.557498</td>
      <td>0.670585</td>
      <td>0.50</td>
      <td>0.582988</td>
      <td>0.548703</td>
      <td>0.523986</td>
    </tr>
    <tr>
      <th>10589</th>
      <td>0.031211</td>
      <td>0.345455</td>
      <td>0.404255</td>
      <td>0.578947</td>
      <td>0.627907</td>
      <td>0.519231</td>
      <td>0.266667</td>
      <td>0.420290</td>
      <td>0.342657</td>
      <td>0.655062</td>
      <td>0.145563</td>
      <td>0.090816</td>
      <td>0.555584</td>
      <td>0.195746</td>
      <td>0.017510</td>
      <td>0.285733</td>
      <td>2</td>
      <td>0.728522</td>
      <td>0.391304</td>
      <td>0.391304</td>
      <td>0.166667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.332643</td>
      <td>0.275362</td>
      <td>0.4250</td>
      <td>0.288889</td>
      <td>0.239437</td>
      <td>0.769231</td>
      <td>0.193548</td>
      <td>0.697368</td>
      <td>0.144985</td>
      <td>0.760116</td>
      <td>0.50</td>
      <td>0.548612</td>
      <td>0.505204</td>
      <td>0.511785</td>
    </tr>
    <tr>
      <th>10590</th>
      <td>0.041199</td>
      <td>0.181818</td>
      <td>0.191489</td>
      <td>0.210526</td>
      <td>0.232558</td>
      <td>0.134615</td>
      <td>0.373333</td>
      <td>0.579710</td>
      <td>0.475524</td>
      <td>0.646146</td>
      <td>0.218729</td>
      <td>0.128592</td>
      <td>0.385993</td>
      <td>0.006565</td>
      <td>0.021818</td>
      <td>0.314923</td>
      <td>2</td>
      <td>0.701031</td>
      <td>0.260870</td>
      <td>0.347826</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.155430</td>
      <td>0.333333</td>
      <td>0.5875</td>
      <td>0.414815</td>
      <td>0.436620</td>
      <td>0.557692</td>
      <td>0.338710</td>
      <td>0.407895</td>
      <td>0.353050</td>
      <td>0.382946</td>
      <td>0.50</td>
      <td>0.314940</td>
      <td>0.418599</td>
      <td>0.556587</td>
    </tr>
    <tr>
      <th>10591</th>
      <td>0.101124</td>
      <td>0.290909</td>
      <td>0.063830</td>
      <td>0.684211</td>
      <td>0.116279</td>
      <td>0.211538</td>
      <td>0.426667</td>
      <td>0.434783</td>
      <td>0.433566</td>
      <td>0.607195</td>
      <td>0.237533</td>
      <td>0.105594</td>
      <td>0.300744</td>
      <td>0.021602</td>
      <td>0.011679</td>
      <td>0.472244</td>
      <td>2</td>
      <td>0.697595</td>
      <td>0.869565</td>
      <td>0.565217</td>
      <td>0.791667</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.012099</td>
      <td>0.550725</td>
      <td>0.5000</td>
      <td>0.474074</td>
      <td>0.492958</td>
      <td>0.576923</td>
      <td>0.516129</td>
      <td>0.500000</td>
      <td>0.529212</td>
      <td>0.508897</td>
      <td>0.75</td>
      <td>0.663827</td>
      <td>0.629111</td>
      <td>0.547209</td>
    </tr>
  </tbody>
</table>
<p>10592 rows × 39 columns</p>
</div>




```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Try GradientBoosting with the GridSearchCV to find best params
gbc = GradientBoostingClassifier()
params = {'max_depth':[3, 5, 7],
         'max_leaf_nodes': [3, 5, 7, 8],
         'random_state': [0]}

gs_gbc = GridSearchCV(gbc,
                      param_grid=params,
                      scoring = 'accuracy',
                      cv = 5)
gs_gbc.fit(result, y_train_appended)
gs_gbc.score(result, y_train_appended)
```




    0.6649358006042296




```python
gs_gbc.best_params_
```




    {'max_depth': 7, 'max_leaf_nodes': 8, 'random_state': 0}




```python
import sklearn.metrics

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Try GradientBoosting with the GridSearchCV to find best params
gbc = GradientBoostingClassifier()
params = {'max_depth':[3, 5, 7],
         'max_leaf_nodes': [3, 5, 7, 8],
         'random_state': [0]}

scorer = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average = 'weighted')

gs_gbc = GridSearchCV(gbc,
                      param_grid=params,
                      scoring = scorer,
                      cv = 5)
gs_gbc.fit(result, y_train_appended)
gs_gbc.score(result, y_train_appended)
```




    0.6417661839384359




```python
from sklearn.pipeline import Pipeline

gbc = GradientBoostingClassifier()
params = {'model__max_depth': [7],
          'model__max_leaf_nodes': [8],
          'model__random_state': [0]}

pipe = Pipeline([
    ('model', gbc)])
pipe.fit(result, y_train_appended)
pipe.score(result, y_train_appended)
```




    0.6502077039274925




```python
from sklearn.externals import joblib
# Export model to pickle
joblib.dump(pipe, '/content/gdrive/My Drive/customer_segments_predicting/model.pkl', compress = 1)
```




    ['/content/gdrive/My Drive/customer_segments_predicting/model.pkl']




```python
dtree_predictions = pipe.predict(result)
multilabel_confusion_matrix(y_train_appended, dtree_predictions)
```




    array([[[7546,  368],
            [2001,  677]],
    
           [[3563, 2085],
            [ 741, 4203]],
    
           [[6370, 1252],
            [ 963, 2007]]])




```python
# For the test set get proper groups as the one of the column in the dataframe
g1_Xtest = X_test.filter(regex = '^g1_')
g1_Xtest['group'] = 1
g2_Xtest = X_test.filter(regex = '^g2_')
g2_Xtest['group'] = 2
g2_Xtest.columns = g1_Xtest.columns
g12_Xtest = g1_Xtest.append(g2_Xtest, ignore_index = True)

# Scale the value other for other than group column
g12_Xtest_without_groups = g12_Xtest.loc[:, g12_Xtest.columns != 'group']
g12_Xtest_without_groups_minmax = scale_values(g12_Xtest_without_groups)

# Drop features 
g12_Xtest_without_groups_minmax.drop(features_to_drop, axis = 1, inplace = True)
g12_Xtest_without_groups_minmax['group'] = g12_Xtest['group']

c_Xtest_minmax_appended = c_Xtest_minmax.append(c_Xtest_minmax, ignore_index = True)
y_test_extended=y_test.append(y_test, ignore_index = True)

result = pd.concat([g12_Xtest_without_groups_minmax, c_Xtest_minmax_appended], axis = 1)
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """
    


```python
result
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
      <th>g1_1</th>
      <th>g1_2</th>
      <th>g1_3</th>
      <th>g1_5</th>
      <th>g1_7</th>
      <th>g1_9</th>
      <th>g1_10</th>
      <th>g1_11</th>
      <th>g1_12</th>
      <th>g1_13</th>
      <th>g1_15</th>
      <th>g1_16</th>
      <th>g1_17</th>
      <th>g1_18</th>
      <th>g1_19</th>
      <th>g1_20</th>
      <th>group</th>
      <th>c_1</th>
      <th>c_2</th>
      <th>c_3</th>
      <th>c_4</th>
      <th>c_5</th>
      <th>c_6</th>
      <th>c_7</th>
      <th>c_8</th>
      <th>c_9</th>
      <th>c_10</th>
      <th>c_11</th>
      <th>c_12</th>
      <th>c_13</th>
      <th>c_14</th>
      <th>c_15</th>
      <th>c_16</th>
      <th>c_22</th>
      <th>c_23</th>
      <th>c_24</th>
      <th>c_25</th>
      <th>c_26</th>
      <th>c_27</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.064654</td>
      <td>0.471698</td>
      <td>0.250000</td>
      <td>0.789474</td>
      <td>0.473684</td>
      <td>0.78</td>
      <td>0.347826</td>
      <td>0.258065</td>
      <td>0.301587</td>
      <td>0.466754</td>
      <td>0.527830</td>
      <td>0.406485</td>
      <td>0.234711</td>
      <td>0.624920</td>
      <td>0.991564</td>
      <td>0.000006</td>
      <td>1</td>
      <td>0.670103</td>
      <td>0.608696</td>
      <td>0.434783</td>
      <td>0.416667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.598998</td>
      <td>0.623188</td>
      <td>0.5500</td>
      <td>0.547445</td>
      <td>0.408451</td>
      <td>0.576923</td>
      <td>0.258065</td>
      <td>0.605263</td>
      <td>0.215120</td>
      <td>0.608343</td>
      <td>0.5</td>
      <td>0.609119</td>
      <td>0.574976</td>
      <td>0.525365</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.028212</td>
      <td>0.075472</td>
      <td>0.068182</td>
      <td>0.789474</td>
      <td>0.105263</td>
      <td>0.18</td>
      <td>0.347826</td>
      <td>0.500000</td>
      <td>0.420635</td>
      <td>0.476618</td>
      <td>0.542316</td>
      <td>0.548157</td>
      <td>0.290984</td>
      <td>0.612185</td>
      <td>0.011070</td>
      <td>0.006066</td>
      <td>1</td>
      <td>0.604811</td>
      <td>0.434783</td>
      <td>0.782609</td>
      <td>0.583333</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.531604</td>
      <td>0.391304</td>
      <td>0.5875</td>
      <td>0.452555</td>
      <td>0.549296</td>
      <td>0.480769</td>
      <td>0.661290</td>
      <td>0.355263</td>
      <td>0.711773</td>
      <td>0.291270</td>
      <td>0.5</td>
      <td>0.665812</td>
      <td>0.525615</td>
      <td>0.473125</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.031249</td>
      <td>0.169811</td>
      <td>0.204545</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.10</td>
      <td>0.463768</td>
      <td>0.629032</td>
      <td>0.547619</td>
      <td>0.760785</td>
      <td>0.770007</td>
      <td>0.762640</td>
      <td>0.489349</td>
      <td>0.282508</td>
      <td>0.006738</td>
      <td>0.011109</td>
      <td>1</td>
      <td>0.649485</td>
      <td>0.521739</td>
      <td>0.434783</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.607934</td>
      <td>0.405797</td>
      <td>0.5875</td>
      <td>0.459854</td>
      <td>0.464789</td>
      <td>0.557692</td>
      <td>0.451613</td>
      <td>0.526316</td>
      <td>0.456890</td>
      <td>0.524228</td>
      <td>0.5</td>
      <td>0.547542</td>
      <td>0.527949</td>
      <td>0.529412</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.240791</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.157895</td>
      <td>0.22</td>
      <td>0.275362</td>
      <td>0.403226</td>
      <td>0.333333</td>
      <td>0.304603</td>
      <td>0.419284</td>
      <td>0.143940</td>
      <td>0.000000</td>
      <td>0.740383</td>
      <td>0.010185</td>
      <td>0.001827</td>
      <td>1</td>
      <td>0.608247</td>
      <td>0.695652</td>
      <td>0.391304</td>
      <td>0.458333</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.023845</td>
      <td>0.478261</td>
      <td>0.6250</td>
      <td>0.518248</td>
      <td>0.507042</td>
      <td>0.576923</td>
      <td>0.419355</td>
      <td>0.486842</td>
      <td>0.453445</td>
      <td>0.499239</td>
      <td>0.5</td>
      <td>0.752380</td>
      <td>0.696666</td>
      <td>0.621067</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.007865</td>
      <td>0.132075</td>
      <td>0.136364</td>
      <td>0.315789</td>
      <td>0.131579</td>
      <td>0.02</td>
      <td>0.405797</td>
      <td>0.677419</td>
      <td>0.539683</td>
      <td>0.682300</td>
      <td>0.437235</td>
      <td>0.392411</td>
      <td>0.648636</td>
      <td>0.253907</td>
      <td>0.005274</td>
      <td>0.005713</td>
      <td>1</td>
      <td>0.707904</td>
      <td>0.478261</td>
      <td>0.434783</td>
      <td>0.291667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.460720</td>
      <td>0.492754</td>
      <td>0.6250</td>
      <td>0.525547</td>
      <td>0.436620</td>
      <td>0.634615</td>
      <td>0.387097</td>
      <td>0.539474</td>
      <td>0.423508</td>
      <td>0.544828</td>
      <td>0.5</td>
      <td>0.343235</td>
      <td>0.340730</td>
      <td>0.500116</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>2643</th>
      <td>0.195238</td>
      <td>0.358491</td>
      <td>0.181818</td>
      <td>0.894737</td>
      <td>0.684211</td>
      <td>0.58</td>
      <td>0.347826</td>
      <td>0.322581</td>
      <td>0.333333</td>
      <td>0.495380</td>
      <td>0.333437</td>
      <td>0.064179</td>
      <td>0.548333</td>
      <td>0.294708</td>
      <td>0.000035</td>
      <td>0.509550</td>
      <td>2</td>
      <td>0.707904</td>
      <td>0.478261</td>
      <td>0.652174</td>
      <td>0.500000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.464967</td>
      <td>0.405797</td>
      <td>0.5625</td>
      <td>0.445255</td>
      <td>0.366197</td>
      <td>0.615385</td>
      <td>0.596774</td>
      <td>0.605263</td>
      <td>0.601209</td>
      <td>0.543221</td>
      <td>0.5</td>
      <td>0.637394</td>
      <td>0.574294</td>
      <td>0.485308</td>
    </tr>
    <tr>
      <th>2644</th>
      <td>0.180054</td>
      <td>0.207547</td>
      <td>0.068182</td>
      <td>0.421053</td>
      <td>0.026316</td>
      <td>0.18</td>
      <td>0.405797</td>
      <td>0.500000</td>
      <td>0.452381</td>
      <td>0.563571</td>
      <td>0.108872</td>
      <td>0.061645</td>
      <td>0.211112</td>
      <td>0.008455</td>
      <td>0.009804</td>
      <td>0.190610</td>
      <td>2</td>
      <td>0.680412</td>
      <td>0.565217</td>
      <td>0.869565</td>
      <td>0.791667</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.142904</td>
      <td>0.565217</td>
      <td>0.5875</td>
      <td>0.540146</td>
      <td>0.760563</td>
      <td>0.442308</td>
      <td>0.725806</td>
      <td>0.500000</td>
      <td>0.639388</td>
      <td>0.412323</td>
      <td>0.5</td>
      <td>0.516780</td>
      <td>0.544211</td>
      <td>0.559017</td>
    </tr>
    <tr>
      <th>2645</th>
      <td>0.095023</td>
      <td>0.056604</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.184211</td>
      <td>0.22</td>
      <td>0.289855</td>
      <td>0.403226</td>
      <td>0.341270</td>
      <td>0.347356</td>
      <td>0.113753</td>
      <td>0.035606</td>
      <td>0.565291</td>
      <td>0.004568</td>
      <td>0.002454</td>
      <td>0.239836</td>
      <td>2</td>
      <td>0.676976</td>
      <td>0.869565</td>
      <td>0.347826</td>
      <td>0.583333</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.567412</td>
      <td>0.420290</td>
      <td>0.6500</td>
      <td>0.503650</td>
      <td>0.450704</td>
      <td>0.326923</td>
      <td>0.451613</td>
      <td>0.486842</td>
      <td>0.399996</td>
      <td>0.414049</td>
      <td>1.0</td>
      <td>0.409835</td>
      <td>0.463331</td>
      <td>0.539959</td>
    </tr>
    <tr>
      <th>2646</th>
      <td>0.037323</td>
      <td>0.528302</td>
      <td>0.590909</td>
      <td>0.157895</td>
      <td>0.394737</td>
      <td>0.44</td>
      <td>0.463768</td>
      <td>0.693548</td>
      <td>0.579365</td>
      <td>0.974649</td>
      <td>0.480618</td>
      <td>0.132745</td>
      <td>0.373406</td>
      <td>0.035374</td>
      <td>0.105928</td>
      <td>0.597813</td>
      <td>2</td>
      <td>0.477663</td>
      <td>0.565217</td>
      <td>0.304348</td>
      <td>0.250000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.532125</td>
      <td>0.782609</td>
      <td>0.8625</td>
      <td>0.810219</td>
      <td>0.619718</td>
      <td>0.538462</td>
      <td>0.629032</td>
      <td>0.434211</td>
      <td>0.566057</td>
      <td>0.351052</td>
      <td>0.5</td>
      <td>0.629507</td>
      <td>0.669657</td>
      <td>0.581510</td>
    </tr>
    <tr>
      <th>2647</th>
      <td>0.076802</td>
      <td>0.075472</td>
      <td>0.000000</td>
      <td>0.894737</td>
      <td>0.210526</td>
      <td>0.18</td>
      <td>0.304348</td>
      <td>0.500000</td>
      <td>0.396825</td>
      <td>0.417564</td>
      <td>0.102256</td>
      <td>0.076615</td>
      <td>0.559152</td>
      <td>0.003697</td>
      <td>0.003607</td>
      <td>0.160365</td>
      <td>2</td>
      <td>0.615120</td>
      <td>0.869565</td>
      <td>0.478261</td>
      <td>0.708333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.543473</td>
      <td>0.536232</td>
      <td>0.4375</td>
      <td>0.437956</td>
      <td>0.718310</td>
      <td>0.250000</td>
      <td>0.580645</td>
      <td>0.328947</td>
      <td>0.580161</td>
      <td>0.479086</td>
      <td>0.5</td>
      <td>0.588110</td>
      <td>0.493371</td>
      <td>0.483194</td>
    </tr>
  </tbody>
</table>
<p>2648 rows × 39 columns</p>
</div>




```python
def divide_into_groups(first_half: bool, 
                       y_test_extended: pd.DataFrame,
                       result: pd.DataFrame) -> pd.DataFrame:
  """
  Divide into 2 group (g1 and g2). If first_half==True, that means the g1 group is
  processing

  """
  if first_half == True:
    half_result = result[:int(len(result)/2)]
    y_test_extended = y_test_extended[:int(len(result)/2)]
  else:
    half_result = result[int(len(result)/2):]
    y_test_extended = y_test_extended[int(len(result)/2):]

  return half_result, y_test_extended
```


```python
def rescale_probability(pipe, half_result: pd.DataFrame, 
                        y_train: pd.DataFrame) -> pd.DataFrame:
  """
  The probability is divided by counts of the value. 

  """

  prediction_probability = pipe.predict_proba(half_result)
  prediction_probability = pd.DataFrame(prediction_probability)

  prediction_probability[0] = prediction_probability[0]/y_train.value_counts()[0]
  prediction_probability[1] = prediction_probability[1]/y_train.value_counts()[1]
  prediction_probability[2] = prediction_probability[2]/y_train.value_counts()[2]

  prediction_probability['result'] = prediction_probability.idxmax(axis=1)

  predictions_half = pipe.predict(half_result)

  return predictions_half
```


```python
"""
As the groups are appended, the next step is to unappend it (divide into 2 group 
# (g1 and g2). Additionally, divide by the value count of the target variable in 
the training set to scale the probability, and avoid overestimating group 1 as the result.
"""
first_half_results, y_test_extended = (
    divide_into_groups(first_half=True, y_test_extended=y_test_extended, result=result))

predictions_first_half = (
    rescale_probability(pipe=pipe, half_result=first_half_results, y_train=y_train))

predictions_first_half = pd.DataFrame(predictions_first_half)
predictions_first_half.rename(columns={ predictions_first_half.columns[0]: "g1"}, inplace = True)
```


```python
predictions_second_half, y_test_extended = (
    divide_into_groups(first_half=False, y_test_extended=y_test_extended, result=result))

predictions_second_half = (
    rescale_probability(pipe=pipe, half_result=predictions_second_half, y_train=y_train))

predictions_second_half = pd.DataFrame(predictions_second_half)
predictions_second_half.rename(columns={ predictions_second_half.columns[0]: "g2"}, inplace = True)
```


```python
result_predicted = pd.concat([predictions_first_half, predictions_second_half], axis = 1)
df = result_predicted.copy()

# Comparison of probability in both groups to get the final results
conditions = [
    (df['g1'] == 2) & (df['g2'] == 1),
    (df['g1'] == 1) & (df['g2'] == 2),
    (df['g1'] == 1) | (df['g2'] == 1),
    (df['g1'] == 2) | (df['g2'] == 2),
    (df['g1'] == 0) & (df['g2'] == 0)]
choices = [1, 0, 1, 2, 0]
df['result'] = np.select(conditions, choices, default=0)
```


```python
multilabel_confusion_matrix(c_ytest, df['result'])
```




    array([[[925,  71],
            [303,  25]],
    
           [[296, 424],
            [201, 403]],
    
           [[655, 277],
            [268, 124]]])




```python
c_ytest.value_counts()
```




    target
    1         604
    2         392
    0         328
    dtype: int64




```python
df['result'].value_counts()
```




    1    827
    2    401
    0     96
    Name: result, dtype: int64




```python
accuracy_score(c_ytest, df['result'])
```




    0.4169184290030212




```python
precision_score(c_ytest, df['result'], average='weighted')
```




    0.37837231290753803



The accuracy results for the training set are similar for the version without considering the c_ parameters. However, for the test set with the same classifier settings, obtained results are worse. Precision is also relatively low. The idea of not combining columns g1 and g2 into one group may be considered (as it was assumed in most of proposed assumptions). The third approach gave relatively the best results, thus making it the final model used for API purposes.

# API for prediction the target group (based of the third approach)

## Configuring and running the app.py

The API has been created to check the target group to make them a marketing campaign. It was created in python by using the flask. Having kept in mind that the recommended software is Anaconda3, the env can be created by using conda.

Useful commands to prepare environments and install necessary libraries:

`conda create -n customer_segments_predicting python=3.7.9 ipython`

`conda activate customer_segments_predicting`

`conda install pip`

In the requirements.txt there were specified requested libraries and the version of them.

`pip install -r requirements.txt`

To run the application, type the following command:

`python app.py`

requirements.txt:

```
scikit-learn==0.22.2.post1
numpy==1.19.5
pandas==1.1.5
sklearn-pandas==1.8.0
joblib==1.0.1
flask==1.1.2
flask-restful==0.3.9
```



## Sending the request

The request can be sent by the Postman. It is required to install this application, and the settings SSL certificate verification should be disabled.

In the collections, create new ones (method POST). In the address  field's enter the following:

`http://127.0.0.1:5000/predict`

where 127.0.0.1 is your localhost. 

Next, in the Body set raw -> JSON. Paste the group to predict and click Send. 

In the response field, there will be results, e.g.

```
{
    "Prediction": "Group 2 to target"
}
```

![png]({{site.url}}/assets/images/Creating_api_for_requestes_files/predict_model.gif)    



