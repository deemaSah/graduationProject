from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import cx_Oracle

embeddings = './output/embeddings.pickle'
recognizer_path = './output/recognizer.pickle'
le_path = './output/le.pickle'

#*************
conStr= 'system/123456@//localhost:1521/orcl'
conn = cx_Oracle.connect(conStr)
print(conn.version)
cur = conn.cursor()

trainingEmployee = []
query = 'select * from "SYSTEM"."EMBADDINGS" '
cur.execute(query)
trainingEmployee = cur.fetchall()

for record in trainingEmployee:
    ID = record[2]
    sqltext = 'select emp_name from "SYSTEM"."EMPLOYEE" where emp_id=:1 '
    cur.execute(sqltext, (ID,))
    arrRecords = cur.fetchall()
    name = cur.fetchall()
    model = SVC(C=1.0, kernel="linear", probability=True)
    model.fit(record[1], name)


#تعديل نصير نجيب الداتا من داتا بيس
#data = pickle.loads(open(embeddings, "rb").read())
#le = LabelEncoder() # encoding labels
#labels = le.fit_transform(data["names"])
#training model
#model = SVC(C=1.0, kernel="linear", probability=True)
#model.fit(data["embeddings"], labels)

# سؤال : اذا بلزم اخزن المودل ع داتا بيس او كافي هيك
f = open(recognizer_path, "wb")
f.write(pickle.dumps(model))
f.close()
f = open(le_path, "wb")
f.write(pickle.dumps(le))
f.close()


