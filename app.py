from flask import Flask,request,render_template,jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

from src.pipline.predict_pipline import CustomData,PredictPipeline
app=Flask(__name__)
CORS(app)


# (0.8516949930152167, 'Gradient Boosting')

# Routes for Home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template("home.html",results=0)
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        print("Before Predections")


        predict_pipline=PredictPipeline()
        print("mid Prediction")
        result=predict_pipline.predict(pred_df)
        print("after Predection")

        return render_template("home.html",results=result[0])
        
if __name__=="__main__":
    app.run(debug=True)