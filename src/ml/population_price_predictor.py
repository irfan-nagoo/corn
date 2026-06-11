from pyspark.sql import SparkSession 
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


data_file = "./data/pop_price/pop_price.csv"

spark = SparkSession.builder.appName("PricePredictor").getOrCreate()
df = spark.read.csv(data_file, header=True, inferSchema=True).cache()

df.show()

print(f"Data Frame length: {df.count()}")

df = df.replace('null', None)
df = df.dropna()

print(f"Data Frame current length: {df.count()}")

column_alias = [col(column).alias(column.replace(' ', '_')) for column in df.columns]

spark.udf.register("toVectorUDT", lambda d: Vectors.dense([d]), returnType=VectorUDT())
vec_df = df.select(*column_alias).selectExpr("toVectorUDT(2014_Population_estimate) as features", 
                                             "cast(2015_median_sales_price as decimal(10, 2)) as label")
vec_df.show()


lr = LinearRegression()
model = lr.fit(vec_df, {lr.regParam:0.0})

prediction = model.transform(vec_df)

prediction.show()

evaulator = RegressionEvaluator(metricName="rmse")
rmse = evaulator.evaluate(prediction)

print(f"Root mean square of model: {rmse}")

model.write().overwrite().save("./model/pop_price_model")