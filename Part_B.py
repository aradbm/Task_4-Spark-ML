import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
import numpy as np
current_dir = os.getcwd()
file_path = os.path.join(current_dir, "prices.txt")
spark = SparkSession.builder.appName("Task_B").getOrCreate()
df = spark.read.text(file_path)
df.show()
spark.stop()
