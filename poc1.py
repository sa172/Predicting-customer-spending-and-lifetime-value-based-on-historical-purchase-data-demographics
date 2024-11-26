import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import openpyxl
#loading dataset
df = pd.read_excel("D:\\datasets\\Online Retail.xlsx")
# print(df.head())
#converting the invoicedate column into the datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'],format='%m%d%Y %H:%M')

#adding the new column to the dataset
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
print(df.head())

# Setting the  reference date for calculating Recency
reference_date = pd.to_datetime('12/31/2010')

#grouping the dataset by the customerId to calculate rfm value
rfm_table = df.groupby('CustomerID').agg(
    Recency =('InvoiceDate',lambda x: (reference_date - x.max()).days),
    Frequency = ('InvoiceDate' ,'count'),
    Monetary = ('TotalAmount','sum')
).reset_index()
# print(rfm_table)

# renaming the column name for clarity
# rfm_table.columns = ['CustomerID','Recency','Frequency','Monetary']
# print(rfm_table)

#setting up the cltv(customer lifetime value)
rfm_table['CLTV'] = rfm_table['Monetary'] *1.5

rfm_table.dropna(inplace=True)

#Split data into features (RFM) and target (CLTV)
X = rfm_table[['Recency', 'Frequency', 'Monetary']]  # Features are the RFM values
y = rfm_table['CLTV']  # Target is the simulated CLTV
print("Shape of y before split:", y.shape)  

#Print the shapes of X and y before splitting
print("Shape of X before split:", X.shape)  # Should be (n_samples, 3)
print("Shape of y before split:", y.shape)  # Should match X's first dimension


# spliting the data into train-test
X_train, X_test , y_train, y_test = train_test_split(X , y , test_size=0.3,random_state=42)

#debuging because I got the error regarding the mismatch input while tarining the model

# Debug check: Print shapes after the split to make sure they are consistent
print("X_train shape:", X_train.shape)  # Should be (n_train_samples, 3)
print("y_train shape:", y_train.shape)  # Should match X_train's first dimension
print("X_test shape:", X_test.shape)    # Should be (n_test_samples, 3)
print("y_test shape:", y_test.shape)    # Should match X_test's first dimension


# Standarize feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print(X_train_scaled.shape)
# Ensure X_test is in 2D format
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled.shape)

# building the model
lnr_model = LinearRegression()

# train the model
lnr_model.fit(X_train_scaled, y_train)

#Model Evaluation
#Predict on the test set
y_pred = lnr_model.predict(X_test_scaled)
print(y_pred)

#calculating the evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
print(f'RMSE: {rmse}')
print(f'R2 : r2')

#Customer Segmentation
#Define CLTV thresholds for segmentation
high_value_threshold = y.mean() + y.std()  # Define high-value customer threshold
low_value_threshold = y.mean() - y.std()   # Define low-value customer threshold

print("Low Value Threshold:", low_value_threshold)
print("High Value Threshold:", high_value_threshold)

if low_value_threshold >= high_value_threshold:
    raise ValueError("Low value threshold must be less than high value threshold!")

df_test = X_test.copy()
df_test['Predicted_CLTV'] = y_pred
df_test['segment'] = pd.cut(df_test['Predicted_CLTV'],
                            bins=[-np.inf, low_value_threshold, high_value_threshold ,np.inf],
                            labels=['Low Value' , 'Medium Value','High Value']
)
print(df_test[['Recency','Frequency','Monetary','Predicted_CLTV','segment']])

# export this segment value for the further analysis or targeting

df_test.to_excel('Segmented_data.xlsx' , index=False)
