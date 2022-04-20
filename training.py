from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

embeddings = './output/embeddings.pickle'
recognizer_path = './output/recognizer.pickle'
le_path = './output/le.pickle'

#تعديل نصير نجيب الداتا من داتا بيس
data = pickle.loads(open(embeddings, "rb").read())
le = LabelEncoder() # encoding labels
labels = le.fit_transform(data["names"])
#training model
model = SVC(C=1.0, kernel="linear", probability=True)
model.fit(data["embeddings"], labels)

# سؤال : اذا بلزم اخزن المودل ع داتا بيس او كافي هيك
f = open(recognizer_path, "wb")
f.write(pickle.dumps(model))
f.close()
f = open(le_path, "wb")
f.write(pickle.dumps(le))
f.close()
