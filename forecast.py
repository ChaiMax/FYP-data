import pandas as pd 
import streamlit as st
import pickle
import requests 
import json 
from datetime import datetime 
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder

def app():
    api_key = "6e7c498f095552e69ba97aec2f66f199"
    lat = "3.1390"
    lon = "101.6869"
    url = "https://api.openweathermap.org/data/2.5/onecall?lat=%s&lon=%s&exclude=minutely,hourly,alerts&appid=%s&units=metric" % (lat, lon, api_key)

    response = requests.get(url)
    data0 = response.json()
    data0['daily'][0]['dt']

    j=0

    for j in range(len(data0['daily'])):
        testing = datetime.fromtimestamp(data0['daily'][j]['dt'])
        #st.write(testing)
    
    it = 0
    dt_w=[]
    temp=[]
    DTR=[]
    rainfall=[]
    RH=[]
    for it in range(len(data0['daily'])):
        if 'rain' not in list(data0['daily'][it]):
            dt_w.append(data0['daily'][it]['dt'])
            temp.append((data0['daily'][it]['temp']['min']+data0['daily'][it]['temp']['max'])/2)
            DTR.append(data0['daily'][it]['temp']['max']-data0['daily'][it]['temp']['min'])
            rainfall.append(0)
            RH.append(data0['daily'][it]['humidity'])
        
        else:
            dt_w.append(data0['daily'][it]['dt'])
            temp.append((data0['daily'][it]['temp']['min']+data0['daily'][it]['temp']['max'])/2)
            DTR.append(data0['daily'][it]['temp']['max']-data0['daily'][it]['temp']['min'])
            rainfall.append(data0['daily'][it]['rain'])
            RH.append(data0['daily'][it]['humidity'])

        df = pd.DataFrame({"DateTime":dt_w,"temperature":temp,"rainfall":rainfall,"RH":RH,"DTR":DTR})
        #ct = changed time format 
        df_ct=df.copy()
        k=0

        for k in range(len(df_ct['DateTime'])):
            df_ct['DateTime'][k] = datetime.fromtimestamp(df_ct['DateTime'][k])

    #st.write(df_ct)

    num0=0
    df_ct['day']=0
    for num0 in range(len(df_ct['DateTime'])):
        df_ct['day'][num0] = df_ct['DateTime'][num0].day
    
    api_key1 = "c3e141cc65c24d50322e7b81855aaf1a"
    lat1 = "3.1390"
    lon1 = "101.6869"
    url1 = "https://api.openweathermap.org/data/2.5/air_pollution/forecast?lat=%s&lon=%s&appid=%s&units=metric" % (lat1, lon1, api_key1)

    response1 = requests.get(url1)
    data1 = response1.json()
    dt_a=[]
    PM10=[]
    NO2=[]
    O3=[]

    opps=0
    m = 0
    for opps in range(len(data1['list'])):
        dt_a.append(data1['list'][opps]['dt'])
        PM10.append(data1['list'][opps]['components']['pm10'])
        NO2.append(data1['list'][opps]['components']['no2'])
        O3.append(data1['list'][opps]['components']['o3'])

    #df for air pollution 
    df_a = pd.DataFrame({"DateTime":dt_a,"PM10":PM10,"NO2":NO2,"O3":O3})

    for m in range(len(df_a['DateTime'])):
        df_a['DateTime'][m] = datetime.fromtimestamp(df_a['DateTime'][m])
    
    num1 = 0
    df_a['day']=0

    for num1 in range(len(df_a['DateTime'])):
        df_a['day'][num1] = df_a['DateTime'][num1].day

        days = df_a['day'].unique().tolist()
        days

        joke = 0
        joke2 = 0

        avg_pm10_p =[]
        avg_no2_n=[]
        avg_o3_o=[]

        for joke in range(len(days)):
            counter = 0
            avg_pm10=0
            avg_no2=0
            avg_o3=0
            for joke2 in range(len(df_a['day'])):
                if df_a['day'][joke2] == days[joke]:
                    avg_pm10+= df_a['PM10'][joke2]
                    avg_no2+= df_a['NO2'][joke2]
                    avg_o3+= df_a['O3'][joke2]
                    counter+=1
            avg_pm10=avg_pm10/counter
            avg_no2= avg_no2/counter
            avg_o3= avg_o3/counter
            
            avg_pm10_p.append(avg_pm10)
            avg_no2_n.append(avg_no2)
            avg_o3_o.append(avg_o3)

        avg_df_a = pd.DataFrame({"day":days,"PM10":avg_pm10_p,"NO2":avg_no2_n,"O3":avg_o3_o})        
    #st.write(avg_df_a)

    match = 0 
    match2 = 0
    df_ct['PM10']=0
    df_ct['NO2']=0
    df_ct['O3']=0
    for match in range(len(df_ct['day'])):
        for match2 in range(len(avg_df_a['day'])):
            if df_ct['day'][match] == avg_df_a['day'][match2]:
                df_ct['PM10'][match]=avg_df_a['PM10'][match2]
                df_ct['NO2'][match]=avg_df_a['NO2'][match2]
                df_ct['O3'][match]=avg_df_a['O3'][match2]

    #pm10
    bins = [0,50,100,250]
    groups = ["Good","Satisfactory","Moderately Polluted"]
    df_ct['Grouped_PM10'] = pd.cut(df_ct['PM10'],bins,labels=groups)

    #no2
    bins = [0,50,100,150]
    groups = ["Good","Moderate","Unhealthy"]
    df_ct['Grouped_NO2'] = pd.cut(df_ct['NO2'],bins,labels=groups)

    #o3
    bins = [0,50,100,150]
    groups = ["Good","Moderate","Unhealthy"]
    df_ct["Grouped_O3"] = pd.cut(df_ct['O3'], bins, labels= groups)

    #temperature
    bins = [-15,0,17,25,35]
    groups = ["Freeze","Cold","Normal","Warm"]
    df_ct["Grouped_temperature"] = pd.cut(df_ct['temperature'], bins, labels= groups)

    #rainfall
    bins = [-0.1, 1, 12, 96, 192]
    groups = ["No","Slightly","Moderate","Heavy"]
    df_ct['Grouped_rainfall'] = pd.cut(df_ct['rainfall'], bins, labels = groups)


    #DTR
    bins = [0,4.35,8.7,15.45,25]
    groups = ["low","low moderate","moderate","moderate high"]
    df_ct["Grouped_DTR"] = pd.cut(df_ct['DTR'], bins, labels= groups)

    #RH
    bins = [0,30,50,100]
    groups = ["Low","Normal","High"]
    df_ct["Grouped_RH"] = pd.cut(df_ct['RH'], bins, labels= groups)

    df_ct.drop(df_ct.tail(3).index,inplace=True)
    st.title('Predicting Chance of Triggering Eczema(Atopic Dermatitis) Based on Weather Analysis')
    #st.dataframe(df_ct.astype('object'))

    # 13 features 
    # season, sex, age, fever, rainfall, Grouped_age, Grouped_PM10, Grouped_NO2, Grouped_O3, Grouped_temperature, Grouped_DTR, Grouped_RH, Grouped_SCORAD
    # season:{'spring':1.0,'summer':'2.0','fall':3.0,'winter':4.0}
    # sex:{1.0:male, 0.0:female}
    st.subheader('Please input the personal attributes below to generate the prediction result.')
    st.subheader('Forecasted and Current weather are ready to be input automatically.')

    sex = st.selectbox("sex: ",['male','female'])
    st.write("Sex: ",sex)

    age = st.slider("Select Age",1,7)
    st.text('Selected:{}'.format(age))

    adscore = st.slider("SCORAD",0,100)
    st.text('Selected:{}'.format(adscore))

    season = "summer"

    fever = st.radio("Fever",("No","Yes"))
    st.write(fever)

    new_df = df_ct.copy()
    #age
    new_df['Grouped_age']=age
    new_df['season']=season
    new_df['sex']=sex
    new_df['age']=age
    new_df['fever']=fever
    new_df['Grouped_SCORAD']=''
    if((adscore>=0) & (adscore <25)):
        new_df['Grouped_SCORAD']='Mild'
    elif ((adscore>25) & (adscore<=50)):
        new_df['Grouped_SCORAD']='Moderate'
    else: new_df['Grouped_SCORAD']='Severe'
    
    
    new_df.drop('temperature',axis=1,inplace=True)
    new_df.drop('DateTime',axis=1,inplace=True)
    new_df.drop('RH',axis=1,inplace=True)
    new_df.drop('NO2',axis=1,inplace=True)
    new_df.drop('O3',axis=1,inplace=True)
    new_df.drop('PM10',axis=1,inplace=True)
    new_df.drop('DTR',axis=1,inplace=True)
    new_df.drop('Grouped_rainfall',axis=1,inplace=True)
    new_df.drop('day',axis=1,inplace=True)

    encoding = new_df.copy()
    encoding.drop('rainfall',axis=1,inplace=True)
    encoding.drop('Grouped_age',axis=1,inplace=True)
    encoding.drop('age',axis=1,inplace=True)
    
    fever_int = {'fever':{'Yes':0,'No':1}}
    season_int = {'season':{'summer':2.0}}
    sex_int = {'sex':{'male':1,'female':0}}
    scoread_int = {'Grouped_SCORAD':{'Mild':0,'Moderate':1,'Severe':2}}
    temp_int = {'Grouped_temperature':{'Warm':3,'Normal:':2}}
    pm10_int={'Grouped_PM10':{"Good":0,"Satisfactory":2,"Moderately Polluted":1}}
    no2_int={'Grouped_NO2':{"Good":0,"Moderate":1,"Unhealthy":2}}
    o3_int={'Grouped_O3':{"Good":0,"Moderate":1,"Unhealthy":2}}
    dtr_int={'Grouped_DTR':{"low":0,"low moderate":1,"moderate":2,"moderate high":3}}
    rh_int={'Grouped_RH':{"Low":1,"Normal":2,"High":0}}


    encoding = encoding.replace(fever_int)
    encoding = encoding.replace(season_int)
    encoding = encoding.replace(sex_int)
    encoding = encoding.replace(scoread_int)
    encoding = encoding.replace(temp_int)
    encoding = encoding.replace(pm10_int)
    encoding = encoding.replace(no2_int)
    encoding = encoding.replace(o3_int)
    encoding = encoding.replace(dtr_int)
    encoding = encoding.replace(rh_int)
    encoding['rainfall']=new_df['rainfall']
    encoding['Grouped_age']=new_df['Grouped_age']
    encoding['age']=new_df['age']
    encoding = encoding.astype(int)

    #st.write(new_df.astype('object'))
    #st.write(encoding)
    pkl_filename = "random_forest_model_SMOTE.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    #score = pickle_model.score(df,y)
    #print("Test score: {0:.2f} %".format(100 * score))
    Ypredict = pickle_model.predict(encoding)
    #st.write(Ypredict)

    st.subheader('Current and Forecasted Weather and Air Quality')
    #final prediction outcome
    #df_ct['Prediction']= Ypredict
    st.write(df_ct.astype('object'))

    st.subheader('Prediction Result Based on Date')
    final = pd.DataFrame({'DateTime':df_ct['DateTime'],'Prediction':Ypredict})
    st.write(final)
    st.text('1 representing there is chance of triggering the Eczema(AD)')
    st.text('0 representing there is no chance of triggering the Eczema(AD)')
