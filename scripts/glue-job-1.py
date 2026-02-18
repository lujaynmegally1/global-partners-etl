import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Paths - Ensure these end with a slash /
BRONZE_PATH = "s3://globalpartners-bronze/dms/dbo/"
SILVER_PATH = "s3://globalpartners-silver/"

# 1. PROCESS: order_items (Casting User ID, Price, and Dates)
df_items = (spark.read.parquet(f"{BRONZE_PATH}order_items/")
    .dropDuplicates(["order_id", "lineitem_id"])
    .withColumn("user_id", F.coalesce(F.col("user_id"), F.lit("GUEST")))
    .withColumn("item_price", F.col("item_price").cast(DecimalType(12, 2)))
    .withColumn("creation_time_utc", F.to_timestamp("creation_time_utc"))
    .withColumn("order_date", F.to_date("creation_time_utc"))
    .withColumn("item_total", F.col("item_price") * F.col("item_quantity"))
)

# 2. PROCESS: order_item_options (Standardizing Case)
df_options = (spark.read.parquet(f"{BRONZE_PATH}order_item_options/")
    .withColumnRenamed("ORDER_ID", "order_id")
    .withColumnRenamed("LINEITEM_ID", "lineitem_id")
    .withColumnRenamed("OPTION_NAME", "option_name")
    .withColumnRenamed("OPTION_PRICE", "option_price")
    .withColumnRenamed("OPTION_QUANTITY", "option_quantity")
    .withColumn("option_price", F.col("option_price").cast(DecimalType(12, 2)))
    .withColumn("option_total", F.col("option_price") * F.col("option_quantity"))
)

# 3. JOIN: Creating the Master Sales Table
# We join the raw options so that every row in Silver has the specific option_price
df_silver_sales = (df_items
    .join(df_options, ["order_id", "lineitem_id"], "left")
    .fillna(0, subset=["option_total", "option_price"])
    .withColumn("total_line_item_revenue", F.col("item_total") + F.col("option_total"))
)


# 4. PROCESS: date_dim (Handling your DD-MM-YYYY format)
df_date = (spark.read.parquet(f"{BRONZE_PATH}date_dim/")
    .withColumn("date_key", F.to_date(F.col("date_key"), "dd-MM-yyyy"))
)

# 5. WRITE & LOG
print(f"Total processed records for Silver: {df_silver_sales.count()}")

df_silver_sales.write.mode("overwrite").partitionBy("order_date").parquet(f"{SILVER_PATH}sales_fact/")
df_date.write.mode("overwrite").parquet(f"{SILVER_PATH}date_dim/")

job.commit()