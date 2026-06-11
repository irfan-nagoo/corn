from pyspark.sql import SparkSession

logFile = "./README.md"
spark = SparkSession.builder.appName("WordCound").getOrCreate()
logData = spark.read.text(logFile).cache()


total_count = logData.filter(logData.value.contains("corn")).count()

print(f"The total count is {total_count}")

spark.stop()