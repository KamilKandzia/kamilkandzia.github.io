---
layout: page
title: API for prediction the target group
permalink: /model_api/
parent: Python
---
# API for prediction the target group (based of the third approach)
This is part of the project that belongs to Predicting Profitable Customer Segments. By using requests, it is possible to check which of the two groups you should target for marketing purposes (or neither of them).

More information about the project:
[link](https://kamilkandzia.github.io/customer_segments/)

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



