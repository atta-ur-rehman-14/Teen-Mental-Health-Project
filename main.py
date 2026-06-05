import pandas as pd
import numpy as np

df = pd.read_csv("Messy_Teen_Mental_Health_Dataset.csv")


print(df.head(10))
print(df.tail(10))
print(df.describe())
print(df.info())
print(pd.isnull(df).sum())

# Fill missing age values with the mean age
df["age"] = df["age"].fillna(df["age"].mean())
# fill missing values of sleep_hours with mean sleep_hours
df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].mean())



df["platform_usage"] = df["platform_usage"].astype(str).str.strip().str.lower()
df["platform_usage"]= df["platform_usage"].replace(["none","null","nan"], np.nan)
df["platform_usage"]= df["platform_usage"].fillna(df["platform_usage"].mode()[0])


# Numpy is imported to calculate the average stress level
average_stress = np.mean(df["stress_level"])
print(average_stress)
# fill missing values
df["depression_label"]= df["depression_label"].fillna(df["depression_label"].mean())
df["addiction_level"]= df["addiction_level"].fillna(df["addiction_level"].mean())
df["anxiety_level"]= df["anxiety_level"].fillna(df["anxiety_level"].mean())
df["stress_level"]= df["stress_level"].fillna(df["stress_level"].mean())
df["physical_activity"]= df["physical_activity"].fillna(df["physical_activity"].mean())
df["academic_performance"]= df["academic_performance"].mean()
df["screen_time_before_sleep"]= df["screen_time_before_sleep"].mean()
df["daily_social_media_hours"] = df["daily_social_media_hours"].mean()

print(pd.isnull(df).sum())