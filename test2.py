import pickle
import pandas as pd
from sklearn.svm import SVC
import cx_Oracle
df=pd.read_excel("C:\\Users\\HP\\OneDrive\\Documents\\TED\\test.xlsx")
df=df.dropna()
#print(df.describe())
#path to serialized db of facial embeddings
embeddings ='./output/embeddings.pickle'
list = pickle.loads(open(embeddings, "rb").read())
#print(list)
target= list['names']
#print(target)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(list['embeddings'], list['names'], test_size=0.2, random_state=0)
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(X_train,y_train)
res=recognizer.fit(X_train,y_train)
print(recognizer.score(X_test,y_test))
