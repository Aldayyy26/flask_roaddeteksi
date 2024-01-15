from flask import Flask, render_template, request, Response, session, redirect, url_for
from flask_session import Session
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import cv2
import io
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import base64

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
  pass

db = SQLAlchemy(model_class=Base)


# Menggabungkan dan mengembalikan jalur path
model_path1 = os.path.join(os.path.dirname(__file__), 'jalan.h5')
#modeldeteksi = load_model(model_path1)

# membuat modul flask 
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///project.db"
# initialize the app with the extension
# menetapkan folder yang akan digunakan sebagai folder statis
app.static_folder = 'static'
db.init_app(app)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

from sqlalchemy import Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

class Review(db.Model):
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    review: Mapped[str] = mapped_column(String)
    hasil: Mapped[str] = mapped_column(String)

with app.app_context():
    db.create_all()



# Membuat kelas untuk mendeteksi object dari OpenCv
RoadCascade = cv2.CascadeClassifier("static/src/cascade3.xml")
models = load_model(model_path1)
label= ['jalan retak','rusak_kecil','rusak_parah','rusak_sedang']




import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()  # Membuat objek lemmatizer
import pickle
import numpy as np

from keras.models import load_model
model = load_model('model/models.h5') #Membuat Objek model chatbot
import json
import random
intents = json.loads(open('model/data.json').read())
words = pickle.load(open('model/texts.pkl','rb'))
classes = pickle.load(open('model/labels.pkl','rb'))

# Sentimen Analisis
import joblib
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

modelSentimen = joblib.load(open(os.path.join(os.path.dirname(__file__), 'model/sentimen/model.pkl'), 'rb'))
cvPickle = joblib.load(open(os.path.join(os.path.dirname(__file__), 'model/sentimen/review.pkl'), 'rb'))



def predict_sentiment(test):
    test = [str(test)]
    test_vector = cvPickle.transform(test).toarray()
    pred = modelSentimen.predict(test_vector)
    return pred[0]


# membuat root chatbot
@app.route("/chatbot", methods=['GET','POST'])
def home_chatbot():
    if request.method == 'POST':
        file = request.files['file']   
    else:
        return render_template("chatbot.html")
        
# Melakukan operasi pembersihan atau pra-pemrosesan pada kalimat
def clean_up_sentence(sentence):
    # melakukan proses tokenisasi untuk berubah menjadi array
    sentence_words = nltk.word_tokenize(sentence)
    # mengembalikan kata ke bentuk dasarnya.
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# Mengembalikan representasi bow dari kalimat input berdasarkan kosa kata yang diberikan.
def bow(sentence, words, show_details=True):
    # Menandai pola setiap kata
    sentence_words = clean_up_sentence(sentence)
    # membuat vektor representasi tas kata untuk suatu kalimat berdasarkan vektor
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # tetapkan 1 jika kata saat ini berada pada posisi kosa kata
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    #Membuat list results yang berisi indeks dan nilai probabilitas dari hasil prediksi yang melebihi ERROR_THRESHOLD.
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]  

    # urutkan berdasarkan kekuatan probabilitas
    results.sort(key=lambda x: x[1], reverse=True) # mengurutkan list
    return_list = [] #membuat list kosong untuk di isi nilai yang dikembalikan
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])}) ##melakukan iterasi hasil prediksi
    return return_list

#mengambil parameter
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model) #untuk mendapatkan hasil prediksi intent dari model berdasarkan input pengguna 
    res = getResponse(ints, intents) #mendapatkan respons berdasarkan hasil prediksi 
    return res

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg') #mengambil nilai dari parameter 
    return chatbot_response(userText)



@app.route("/")
def home():
    return render_template("index.html")


#Mengijinkan files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Berfungsi untuk memuat dan menyiapkan gambar dalam bentuk yang tepat
def read_image(filename):

    img = load_img(filename, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path)
            classes=models.predict(img) 
            index = np.argmax(classes)
            return render_template('predict.html', road = label[index],prob=round(classes[0][index]*100,2), user_image = file_path)
        else:
            return render_template('index.html')


def detect_Road(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Road = RoadCascade.detectMultiScale(
        gray,
        scaleFactor = 1.05,
        minNeighbors = 6,
        minSize = (128,128),
        maxSize = (800,800)
    )
    
    for (x, y, w, h) in Road:
        load = frame[y:y+h, x:x+w]
        load = cv2.resize(load, (128,128))
        z = tf.keras.utils.img_to_array(load)
        z = np.expand_dims(z, axis=0)
        images = np.vstack([z])
        classes = models.predict(images)
        index = np.argmax(classes)
        cv2.putText(frame, label[index], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2) 
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)
    return frame

def gen_frames(): 
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_Road(frame)
            ret, buffer = cv2.imencode('.jpg',frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect')
def detector():
    return render_template('detect.html')

@app.route('/savefile', methods=['POST'])
def savefile():
    data = request.get_json()
    if data and 'image' in data:
        img_data = data['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(img_data)))
        image.save(os.path.join(os.path.dirname(__file__), "static/uploads/capture-camera.png"))

        return 'success', 200
        
    # facedata = os.path.join(os.path.dirname(__file__), 'static/src/cascade3.xml')
    # cascade = cv2.CascadeClassifier(facedata)

    # img = cv2.imread(os.path.join(os.path.dirname(__file__), "capture-camera.png"))

    # minisize = (img.shape[1],img.shape[0])
    # miniframe = cv2.resize(img, minisize)

    # faces = cascade.detectMultiScale(miniframe)

    # output_path = os.path.join(os.path.dirname(__file__), 'detected_face_.jpg')
    # cv2.imwrite(output_path, faces)

    return 'error', 400

@app.route('/predict_capture', methods=['GET', 'POST'])
def predict_capture():
    file_path = os.path.join(os.path.dirname(__file__), "static/uploads/capture-camera.png")
    img = read_image(file_path)
    classes=models.predict(img)
    index = np.argmax(classes)
    return render_template('predict_capture.html', road = label[index],prob=round(classes[0][index]*100,2), sh_img = "capture-camera.png")

@app.route('/sentimen', methods=['GET','POST'])
def sentiment_analisis():
    if request.method == 'POST':
        kata = request.form['review']
        predict = predict_sentiment(kata)

        review = Review(
            review = kata,
            hasil = predict

        )
        db.session.add(review)
        db.session.commit()

        return render_template('sentimen.html', prediksi = predict)

    return render_template('sentimen.html')


@app.route('/dashboard')
def dashboard():
    if session.get('username') is None:
        return redirect(url_for('admin'))
    else :
        review = db.session.execute(db.select(Review).order_by(Review.id)).scalars()
        jumlah = db.session.query(Review.hasil, func.count(Review.hasil)).group_by(Review.hasil).all()
        return render_template('dashboard.html', reviews = review, jmh = jumlah)

@app.route('/admin', methods=['POST', 'GET'])
def admin():
    if request.method == "POST":
        if request.form['username'] == 'admin' and request.form['password'] == 'admin123':
            session['username'] = "admin"
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html')

    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)