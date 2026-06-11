import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import col
from pyspark.ml.regression import LinearRegressionModel



spark = SparkSession.builder.appName("PricePredictorInference") \
    .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
    .getOrCreate()

data = [{
    'features' : 315245.0, 
    'label': 208.00
    }]

df = pd.DataFrame(data)
test_df = spark.createDataFrame(data)
test_df.cache()

test_df.show()

column_alias = [col(column) for column in test_df.columns]
spark.udf.register("toVectorUDT", lambda d: Vectors.dense([d]), returnType=VectorUDT())
vec_df = test_df.select(*column_alias).selectExpr("toVectorUDT(features) as features", "label")

vec_df.show()

model = LinearRegressionModel.load("./model/pop_price_model")

prediction = model.transform(vec_df)

prediction.show()

spark.stop()