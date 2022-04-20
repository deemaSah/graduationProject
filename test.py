import cx_Oracle


conStr='system/deema@localhost:1521/orcl'

conn = cx_Oracle.connect(conStr)
cur = conn.cursor()
sqlTxt='select * from "SYSTEM"."EMPLOYEES" '
cur.execute(sqlTxt)
records = cur.fetchall()
for record in records:
    #name = imagePath.split(os.path.sep)[-2]# Extract the person's name from the path
    name = record[1]
    query='select * from "SYSTEM"."TRAININGIMG" where EMP_ID =:1 '
    #print(record[0])
    cur.execute(query,(record[0],))
    images= cur.fetchall()
    print(images)
    import cv2
    import imutils

    for img in images:
     for i in range(1,img.__len__()):
      image = cv2.imread(img[i])
      image = imutils.resize(image, width=600)
      (h, w) = image.shape[:2]
      #cv2.imwrite("cut"+i.__str__()+".jpg", image)
query='INSERT INTO "SYSTEM"."EMBADDINGS" VALUES (0, \'1\',\'1\')'
cur.execute(query)
conn.commit()
cur.close()
conn.close()

