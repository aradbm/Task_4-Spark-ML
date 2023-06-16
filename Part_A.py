import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

current_dir = os.getcwd()
file_path = os.path.join(current_dir, "books.json")
spark = SparkSession.builder.appName("Task_A").getOrCreate()
df = spark.read.json(file_path, multiLine=True)

# 1. books with authors whose name starts with F
authors_with_f = df.filter(df.author.like("F%")).withColumn(
    "years_since_publication", lit(2023 - df.year)).select("title", "author", "years_since_publication")
authors_with_f.show()

# 2. calculate the average page numbers for each author. only for english books
avg_page_numbers = df.filter(df.language == "English").groupBy(
    "author").avg("pages")
avg_page_numbers.show()
spark.stop()
