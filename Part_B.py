import os
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import lit, split, col, udf
from pyspark.sql.types import ArrayType, FloatType
import numpy as np

# create spark session
current_dir = os.getcwd()
file_path = os.path.join(current_dir, "prices.txt")
spark = SparkSession.builder.appName("Task_B").getOrCreate()
df = spark.read.text(file_path)

# define split function
split_me = udf(lambda x: [float(e)
               for e in x.split(",")[:-1]], ArrayType(FloatType()))

# apply split function to the dataframe
df = df.withColumn('x', split_me(col("value")))
df = df.withColumn('y', split(col("value"), ",").getItem(11).cast(FloatType()))
df = df.drop("value")

# collect data & get the total number of rows
mat = df.collect()
total_rows = len(mat)

# use the first 3/4 of the data for training
training_data = mat[:total_rows - 3]

data_x = np.array([m['x'] for m in training_data])
data_y = np.array([m['y'] for m in training_data])

# Set weights, bias and learning rate
w = np.zeros(len(data_x[0]))
b = 0
alpha = 0.001

# Gradient Descent
for iteration in range(112000):
    deriv_b = np.mean((np.dot(data_x, w)+b)-data_y)
    gradient_w = 1.0/len(data_y) * \
        np.dot(((np.dot(data_x, w)+b)-data_y), data_x)
    b -= alpha*deriv_b
    w -= alpha*gradient_w

print("w = ", w)
print("b = ", b)

# test the model
test_data = mat[total_rows - 3:]
test_x = np.array([m['x'] for m in test_data])
test_y = np.array([m['y'] for m in test_data])
predictions = np.dot(test_x, w) + b
mse = np.mean((test_y - predictions) ** 2)
print("Mean Squared Error:" + str(mse))

spark.stop()
