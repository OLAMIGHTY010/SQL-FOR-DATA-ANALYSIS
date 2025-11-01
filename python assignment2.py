# Applied Learning Assignment 2: Students' Performance Analysis

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# ===============================
# Load the dataset
# ===============================
students_df = pd.read_excel("students_performance_records.xlsx")

print("Students dataset loaded. Shape:", students_df.shape)
print("First 5 rows:")
print(students_df.head())

# ===============================
# Handle missing or inconsistent data
# ===============================
# Check for missing values
print("\nMissing values per column:")
print(students_df.isnull().sum())

# For simplicity, drop rows with missing GPA or StudyTimeWeekly
students_df = students_df.dropna(subset=['GPA', 'StudyTimeWeekly', 'GradeClass'])

# ===============================
# Bar Chart: Average GPA by Grade Class
# ===============================
avg_gpa_by_grade = students_df.groupby('GradeClass')['GPA'].mean().sort_index()

plt.figure(figsize=(8,5))
colors = plt.cm.viridis(avg_gpa_by_grade / avg_gpa_by_grade.max())  # colormap based on GPA
bars = plt.bar(avg_gpa_by_grade.index.astype(str), avg_gpa_by_grade, color=colors)
plt.xlabel('Grade Class')
plt.ylabel('Average GPA')
plt.title('Average GPA by Grade Class')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add annotations
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval,2),
             ha='center', va='bottom')

plt.show()

# ===============================
# Scatter Matrix: Study Time, Absences, and GPA
# ===============================
scatter_cols = ['StudyTimeWeekly', 'Absences', 'GPA']
scatter_matrix_df = students_df[scatter_cols]

scatter_matrix(scatter_matrix_df, figsize=(10,10), diagonal='hist', color='teal', alpha=0.6)
plt.suptitle('Scatter Matrix: Study Time, Absences, and GPA', fontsize=16)
plt.show()

# ===============================
# Box Plot: Study Time Distribution by Grade Class
# ===============================
plt.figure(figsize=(8,5))
grade_classes = sorted(students_df['GradeClass'].unique())
colors = plt.cm.Set3(np.linspace(0,1,len(grade_classes)))

box = plt.boxplot([students_df[students_df['GradeClass']==g]['StudyTimeWeekly'] for g in grade_classes],
                  patch_artist=True, labels=[str(g) for g in grade_classes])

# Color each box
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.xlabel('Grade Class')
plt.ylabel('Study Time Weekly (hours)')
plt.title('Study Time Distribution by Grade Class')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
