from flask import Flask, render_template, url_for, request
from pyspark import SparkContext
from pyspark.sql import SQLContext
import pandas as pd
import numpy as np
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import udf, col, lower, regexp_replace
from pyspark.ml.classification import LogisticRegressionModel
import re
import pyspark 
from pyspark.sql import functions as f
from pyspark.sql import types as t

spark = SparkSession.builder.appName('Youtube Predictive Title').master("local").getOrCreate()
sqlContext = SQLContext(spark)

app = Flask(__name__)


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_1 = LogisticRegressionModel.load('model')	

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        my_prediction = model_1.transform(data)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)




