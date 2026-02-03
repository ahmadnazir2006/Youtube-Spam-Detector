from flask import Flask,request,jsonify
from flask_cors import CORS
import re
import emoji as emo
import joblib
vectorizer=joblib.load('vectorizer.pkl')
model=joblib.load('spam_model.pkl')
app=Flask(__name__)
CORS(app)
def clean_data(comment): #this function is for cleaning the data by removing the url, emojis, and special characters
    text=str(comment).lower()
    url_pattern = r'(https?://\S+|www\.\S+|watch\?v=\S+)'
    text = re.sub(url_pattern, 'link_token', text)
    text=emo.demojize(text)
    text=text.replace('@', 'at_token')
    text=text.replace('#', 'hash_token')
    text=text.replace('&', 'and_token')
    text=text.replace('*', 'star_token')
    text=text.replace(':',' ')
    text = " ".join(text.split())
    return text
@app.route('/predict',methods=['POST'])
def predict():
    data=request.get_json(force=True)
    if 'text' not in data and not data:
        return jsonify({'error':'text is required'}),400
    text=data['text']
    cleaned_text=clean_data(text)
    
    vectorized_text=vectorizer.transform([cleaned_text])
    
    prediction=model.predict(vectorized_text)
    if prediction[0]==0:
        return jsonify({'prediction':'spam'}),200
    else:
        return jsonify({'prediction':'not spam'}),200

    






if __name__=='__main__':
    app.run(debug=True)