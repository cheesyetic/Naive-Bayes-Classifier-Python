#IMPORT LIBRARY
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
import category_encoders as ce #untuk melakukan encoding (pelabelan instances)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#INPUT DATA
data = pd.read_csv("data.csv")
print("Data Mentah")
print(data.head())

#CHECKING DATA INFO
print()
print("Hasil Pengecekan Informasi Data")
print(data.info())

#CHECKING MISSING VALUE
print()
print("Hasil Pengecekan Missing Value")
print(data.empty)
print()

#SEPARATING INDEPENDENT VARIABLE AND DEPENDENT VARIABLE
print()
#INDEPENDENT VARIABLE
xdf = data.drop(["COVID-19"], axis = 1)
print("Variabel X")
print(xdf.head())
#DEPENDENT VARIABLE
print()
ydf = data["COVID-19"]
print("Variabel Y")
print(ydf.head())

#CONVERTING DEPENDENT VARIABLE INTO NUMBERS
col_names = ['Breathing_Problem', 'Fever', 'Dry_Cough', 'Sore_throat', 'Running_Nose', 'Asthma', 'Chronic_Lung_Disease', 
	'Headache', 'Heart_Disease', 'Diabetes', 'Hyper_Tension', 'Fatigue', 'Gastrointestinal', 'Abroad_travel', 
	'Contact_with_COVID_Patient', 'Attended_Large_Gathering', 'Visited_Public_Exposed_Places', 
	'Family_working_in_Public_Exposed_Places', 'Wearing_Masks', 'Sanitization_from_Market']
xdf.columns = col_names
encoder = ce.OrdinalEncoder(cols=['Breathing_Problem', 'Fever', 'Dry_Cough', 'Sore_throat', 'Running_Nose', 'Asthma', 'Chronic_Lung_Disease', 
	'Headache', 'Heart_Disease', 'Diabetes', 'Hyper_Tension', 'Fatigue', 'Gastrointestinal', 'Abroad_travel', 
	'Contact_with_COVID_Patient', 'Attended_Large_Gathering', 'Visited_Public_Exposed_Places', 
	'Family_working_in_Public_Exposed_Places', 'Wearing_Masks', 'Sanitization_from_Market'])
x = encoder.fit_transform(xdf)
le = preprocessing.LabelEncoder()
y = le.fit_transform(ydf)
print()
print("Hasil Konversi Variabel Y")
print(y)

#PREPARING THE K-FOLD VALIDATION
cv = KFold(n_splits=10, random_state=None)
model = GaussianNB()
score = []
i=1

#DOING THE K-FOLD VALIDATION
for train_index, test_index in cv.split(x) :
	#SPLIT DATA INTO DATA TRAIN AND DATA TEST
	x_train, x_test = x.iloc[train_index,:] ,x.iloc[test_index,:]
	y_train, y_test = y[train_index], y[test_index]
	#TRAIN THE DATA
	model.fit(x_train,y_train)
	#PREDICT THE DATA
	y_predict = model.predict(x_test)
	#CHECKING RESULT
	print()
	print("Hasil K-Fold ke-", i)
	print()
	print("Data Train X")
	print(x_train.head())
	print()
	print("Data Test X")
	print(x_test)
	print()
	print("Data Train Y")
	print(y_train)
	print()
	print("Data Test Y")
	print(y_test)
	print()
	print("Hasil Prediksi")
	print(y_predict)
	print()
	#CHECKING THE PREDICT ACCURACY
	print("CONFUSION MATRIX")
	cm = np.array(confusion_matrix(y_test, y_predict))
	print(cm)
	print()
	acc = accuracy_score(y_predict, y_test)
	score.append(acc)
	i=i+1

#PRINT THE ACCURACY
meanacc = np.mean(score, axis=0)
print("Rata-rata akurasi :", meanacc)