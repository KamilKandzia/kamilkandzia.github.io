---
layout: default
title: Python
nav_order: 2
has_children: true
---

All projects in this language were done outside of my studies. I had the chance to use an SQLite database, parse commands using the console and predict the seasonality of sales. Additionally, I created my model for customer segmentation that compares two groups and indicates which one should be targeted. In this case, the requests are sent in JSON format by the Postman. 

### Some of my projects:

|Project|Description|
|:---|:---|
|[Prediction model of sales in alcohol stores <br> by using the Prophet](https://kamilkandzia.github.io/prophet/)|Build a prediction model by the <span class="label label-green">prophet</span> to indicate if the credit could be granted to some stores. For the stores, there is information about the revenue of the alcohol sales. The clustering has been implemented to find stores with similar attributes.|
|[Forecasting of sales](https://kamilkandzia.github.io/forecasting/)|The aim was to create a model prediction of the sales for the next three weeks. Currently, the sales forecast is set 3 weeks ahead based on last week’s sales. The Weighted Absolute Percent Error (WAPE) is used for comparison purposes. The whole dataset contains 3 CSV files. <span class="label label-green">prophet</span> <span class="label label-green">pandas</span>|
|[Predicting profitable customer segments](https://kamilkandzia.github.io/customer_segments/)|Models for customer segmentation that compares two groups and indicates which one should be targeted were created. Based on the approaches, five different models have been made. For one of the models (GradientBoostingClassifier) to predict if the campaign should be launched for the group (one of them, or none of them)), the requests are sent in JSON format by the Postman. <span class="label label-green">GradientBoostingClassifier</span> <span class="label label-green">LogisticRegression</span> <span class="label label-green">KNN</span> <span class="label label-green">modeling</span>  <span class="label label-green">pandas</span> <span class="label label-green">numpy</span>|
|[Stock price: jumping out and in <br> of dividend stocks <br> around ex dividend dates](https://kamilkandzia.github.io/stock/)|The payment of dividends by a company is an attractive morsel for a shareholder. Such a company is better perceived because of its attractiveness and its willingness to share its profits with investors. But is it always profitable to own shares when dividends are paid? In this note, I will try to answer this question. For the analysis, I have chosen companies that regularly pay dividends on the Polish stock exchange. <span class="label label-green">pandas</span> <span class="label label-green">BeautifulSoup</span> <span class="label label-green">requests</span> |

