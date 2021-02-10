import cv2
from deepface import DeepFace


def for_image(img):

    attributes = ['age', 'gender', 'race', 'emotion']
    results  = DeepFace.analyze(img,attributes)
    #print(results["age"], " years old ", results["dominant_race"], " ", results["dominant_emotion"], " ",results["gender"])
    #print(json.dumps(results))
    dict1 = {
        "age": results["age"],
        "race": results["dominant_race"],
        "emotion": results["dominant_emotion"],
        "gender": results["gender"]
    }
    print(f'{dict1}')
def for_video(cap):
    while 1:
        ret, img = cap.read()
        result1 = DeepFace.analyze(img, actions=['age', 'race', 'emotion', 'gender'])
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        dict1 = {
            "age": result1["age"],
            "race": result1["dominant_race"],
            "emotion": result1["dominant_emotion"],
            "gender": result1["gender"]
        }
        cv2.putText(img, f'{dict1}', (50, 50), font, 0.75, (0, 0, 255), 1, cv2.LINE_4)

        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
for_video(cap)
#img='a.jpg'
#for_image(img)
