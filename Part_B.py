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

# collect data
mat = df.collect()

# Get the total number of rows
total_rows = len(mat)

# Use the first n-3 rows for training
training_data = mat[:total_rows - 3]

# prepare data_x and data_y for training
data_x = np.array([m['x'] for m in training_data])
data_y = np.array([m['y'] for m in training_data])

# Set weights (w), bias (b) and learning rate (alpha)
# Weight vector should match the number of features
w = np.zeros(len(data_x[0]))
b = 0
alpha = 0.001

# Implement Gradient Descent for training data
for iteration in range(112000):
    deriv_b = np.mean((np.dot(data_x, w)+b)-data_y)
    gradient_w = 1.0/len(data_y) * \
        np.dot(((np.dot(data_x, w)+b)-data_y), data_x)
    b -= alpha*deriv_b
    w -= alpha*gradient_w

print("w = ", w)
print("b = ", b)

# Now, let's test the model on the last 3 lines of the dataset
test_data = mat[total_rows - 3:]

# prepare data_x and data_y for testing
test_x = np.array([m['x'] for m in test_data])
test_y = np.array([m['y'] for m in test_data])

# Get the predictions
predictions = np.dot(test_x, w) + b

# Print the real and predicted values
for real, pred in zip(test_y, predictions):
    print("Real:" + str(real) + " Pred:" + str(pred))

# Calculate the Mean Squared Error
mse = np.mean((test_y - predictions) ** 2)
print(f'Mean Squared Error: {mse}')
