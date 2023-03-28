from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            age=request.form.get('age'),
            sex=request.form.get('sex'),
            cp=request.form.get('cp'),
            trestbps=request.form.get('trestbps'),
            chol=request.form.get('chol'),

            fbs=request.form.get('fbs'),
            restecg=request.form.get('restecg'),
            thalach=request.form.get('thalach'),
            exang=request.form.get('exang'),
            oldpeak=request.form.get('oldpeak'),

            slope=request.form.get('slope'),
            ca=request.form.get('ca'),
            thal=request.form.get('thal'),
            # reading_score=float(request.form.get('writing_score')),
            # writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")   