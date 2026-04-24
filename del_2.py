import boto3
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

client=boto3.client(
    'rekognition',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")    
)

bucket_name = "my-ai-iot-images"
rows = []

response = s3.list_objects_v2(Bucket=bucket_name)

for obj in response.get("Contents", []):
    key = obj["Key"]

    if not key.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    rek_response = client.detext_labels(
        Image={
            "S3Object": {
                "Bucket": bucket_name,
                "Name": key
            }
        },
        MaxLabels=10,
        MinConfidence=70
    )

    for label in rek_response["Labels"]:
        rows.append({
            "image": key,
            "label": label["Name"],
            "confidence": label["Confidence"],
            "instances": len(label["Instances"])
        })

df = pd.DataFram(rows)
print(df.head())
df.to_csv("rekognition_results.csv", index=False)