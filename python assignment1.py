import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("job_change_data.csv")

# Show dataset info
print("Columns in dataset:", df.columns.tolist())
print("Dataset shape:", df.shape)
print(df.head(), "\n")


# Step 1: Clean numeric columns

# Convert city_development_index to float
df['city_development_index'] = pd.to_numeric(df['city_development_index'], errors='coerce')

# Convert company_size to numeric
size_mapping = {
    '1-10': 5,
    '10-50': 30,
    '50-99': 75,
    '100-500': 300,
    '500-999': 750,
    '1000-4999': 3000,
    '5000-9999': 7500,
    '10000+': 10000
}
df['company_size'] = df['company_size'].replace(size_mapping)
df['company_size'] = pd.to_numeric(df['company_size'], errors='coerce')

# Convert experience and last_new_job to numeric
# Note: 'last_new_job' has 'never' and '>4'
df['experience'] = pd.to_numeric(df['experience'], errors='coerce')
df['last_new_job'] = df['last_new_job'].replace({'never': 0, '>4': 5})
df['last_new_job'] = pd.to_numeric(df['last_new_job'], errors='coerce')


# Step 2: Filter rows

filtered_df = df[(df['city_development_index'] > 0.8) & (df['company_size'] > 3)]
print("Filtered dataset:\n", filtered_df.head(), "\n")


# Step 3: Select first 10 rows and specific columns using iloc

selected_rows = df.iloc[:10][['experience', 'education_level']]
print("First 10 rows with Experience and Education_Level:\n", selected_rows, "\n")

# Step 4: Group by Relevant_Experience and calculate average city_development_index

grouped_experience = df.groupby('relevent_experience')['city_development_index'].mean()
print("Average city_development_index by relevant_experience:\n", grouped_experience, "\n")

# Step 5: Group by Company_Size and count unique last_new_job entries

grouped_company = df.groupby('company_size')['last_new_job'].nunique()
print("Unique last_new_job counts by company_size:\n", grouped_company, "\n")

# Step 6: Frequency distribution of Company_Type

company_type_counts = df['company_type'].value_counts()
print("Company_Type frequency distribution:\n", company_type_counts, "\n")


# Step 7: Fill missing numerical values with mean

numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Step 8: Drop rows where >50% of data is missing
initial_rows = df.shape[0]
df = df.dropna(thresh=df.shape[1]*0.5)
print(f"Rows before drop: {initial_rows}, after drop: {df.shape[0]}\n")

# Step 9: Query dataset: experience > 10 & company_size == 7
query_df = df[(df['experience'] > 10) & (df['company_size'] == 7)]
print("Query result (experience>10 & company_size=7):\n", query_df, "\n")

# Step 10: Create new feature Experience_Gap

df['experience_gap'] = df['experience'] - df['last_new_job']
print("Dataset with Experience_Gap:\n", df[['experience', 'last_new_job', 'experience_gap']].head(), "\n")


# Step 11: Normalize city_development_index

scaler = MinMaxScaler()
df['city_development_index_norm'] = scaler.fit_transform(df[['city_development_index']])
print("Normalized city_development_index:\n", df[['city_development_index', 'city_development_index_norm']].head(), "\n")

# Step 12: Create new column cdi_per (as % of normalized city development)

df['cdi_per'] = df['city_development_index_norm'] * 100

# Example merge (just merging with itself for demonstration)
df_merged = pd.merge(df, df[['enrollee_id', 'cdi_per']], on='enrollee_id', suffixes=('', '_merged'))
print("Dataset with merged cdi_per insights:\n", df_merged.head())
