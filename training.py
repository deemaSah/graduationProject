# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import cx_Oracle
import numpy as np
import pandas as pd
df=pd.read_excel("C:\\Users\\HP\\OneDrive\\Documents\\TED\\test.xlsx")
df=df.dropna()
input=np.array([df]).transpose()









conStr='system/deema@localhost:1521/orcl'

conn = cx_Oracle.connect(conStr)
cur = conn.cursor()
#path to serialized db of facial embeddings
embeddings ='./output/embeddings.pickle'

#path to output model trained to recognize faces
recognizer_path = './output/recognizer.pickle'

#path to output label encoder
le_path = './output/le.pickle'
sqlTxt='select * from SYSTEM."TDATA" '
cur.execute(sqlTxt)
records = cur.fetchall()
conn.commit()
cur.close()
conn.close()
embeddingsList=[]
IDList=[]
for rec in records:
    embeddingsList.append(rec[1])
    IDList.append(rec[2])



# load the face embeddings
data = pickle.loads(open(embeddings, "rb").read())

# encode the labels
le = LabelEncoder()
labels = le.fit_transform(data["names"])
print(data["embeddings"][0])
list = np.array(embeddingsList)
print(np.array(list[0]))

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], data["names"])
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(df, data
#['names'], test_size=0.2, random_state=0)
#recognizer = SVC(C=1.0, kernel="linear", probability=True)
#recognizer.fit(X_train,y_train)

# get the accuracy

# write the actual face recognition model to disk
f = open(recognizer_path, "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(le_path, "wb")
f.write(pickle.dumps(le))
f.close()