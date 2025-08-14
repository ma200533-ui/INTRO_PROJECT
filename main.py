import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
# Set Seaborn style
sns.set(style="whitegrid")

zip_path = r"C:\Users\IT\Downloads\titanic.zip"

with zipfile.ZipFile(zip_path) as z:
    train = pd.read_csv(z.open("train.csv"))
    test = pd.read_csv(z.open("test.csv"))

print(" Train data:")
print(train.head())

print("\n Test data:")
print(test.head())

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nDataset info:")
print(train.info())
print("\nMissing values:\n", train.isnull().sum())


train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)


plt.figure(figsize=(6,4))
sns.countplot(data=train, x='Survived')
plt.title('Survival Count')
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(data=train, x='Sex', hue='Survived')
plt.title('Survival by Gender')
plt.show()


plt.figure(figsize=(8,5))
sns.histplot(data=train, x='Age', hue='Survived', multiple='stack', bins=30)
plt.title('Age Distribution by Survival')
plt.show()


plt.figure(figsize=(8,5))
sns.histplot(train['Fare'], bins=30, kde=True)
plt.title('Fare Distribution')
plt.show()


plt.figure(figsize=(8,6))
corr = train.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


