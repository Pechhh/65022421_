import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
 
file_path = 'C:/Users/User/Downloads'
file_name = 'car_data.csv'
 
df = pd.read_csv(file_path + file_name)
 
df.dropna(inplace=True)
df.drop(columns=['User ID'], inplace=True)
 
encode = LabelEncoder()
 
df['Gender'] = encode.fit_transform(df['Age'])
 
x = df.iloc[:,:3]
y = df['Age']
 
x_train, x_test, y_train, y_test = train_test_split(x,y)
 
model = DecisionTreeClassifier()
model.fit(x,y)
 
Train_score = model.score(x, y)
Test_score = model.score(x,y )
print('Accuracy of Train : ', '{:.2f}'.format(Train_score))
print('Accuracy of Test : ', '{:.2f}'.format(Test_score))
 
feature = x.columns.tolist()
data_class = y.tolist()
 
plt.figure(figsize=(25,20))
_ = plot_tree(model,
              feature_names= feature,
              class_names= data_class,
              label= 'all',
              impurity= True,
              precision= 3,
              filled= True,
              rounded= True,
              fontsize= 16
              )
plt.show()
 
feature_importance = model.feature_importances_
feature_name = ['Age','AnnualSalary']
 
sns.set({'figure.figsize' : (11.7,8.27)})
sns.barplot(x = feature_importance,y = feature_name)