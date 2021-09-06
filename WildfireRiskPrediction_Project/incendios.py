#An unsupervised density-based clustering technique is used to predict the risk of wildfire, having as use case the fires of Ponteareas, Galicia in 2017. Code was written following the method proposed in Salehi et al 'Dynamic and Robust Wildﬁre Risk Prediction System: An Unsupervised Approach' IBM Research paper. The model achieved an AUC of 0.98 compared to the current way of estimating fire risk, the Forest Fire Danger Index (FFDI) which obtained an AUC of 0.76. Looking at the Precision-Recall curve (more appropriate for rare events) the obtained Average Precison Recall score was 0.75, far better than the 0.26 score for the FFDI. 

#PAPER LINK: https://www.researchgate.net/publication/305997301_Dynamic_and_Robust_Wildfire_Risk_Prediction_System_An_Unsupervised_Approach
#Weather Data: https://www.meteogalicia.gal/observacion/estacionshistorico/consultar.action

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import chi2
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score


#calculate mahalanobis distance between observation and cluster, if observation out of the *percentile* covered by hyperelipsoid then is considered to be anomalous
def mahalanomalous(observation,cluster_set,percentile=0.999995):
    dof=len(observation)
    anomaly_threshold=chi2.ppf(percentile,dof)
    anomaly=0
    for c in cluster_set:
        left_term=np.dot(observation-c[1],sp.linalg.pinv(c[2])) #use pseudo-inverse to avoid singular cov matrix error
        mahalanobis_distance=np.dot(left_term,(observation-c[1]).T)
        if mahalanobis_distance>anomaly_threshold:
            anomaly=1
    #FFDI filter to avoid rare cold days
    if 2*np.exp(-0.45+0.987*np.log(0.191/3.52*(observation[0]+104))-0.0345*observation[1]+0.0338*observation[2]+0.0234*observation[3])<5:
        anomaly=0
    return anomaly

#weight function as inverse of Hausdorff distance between cluster sets
def weight(current_cluster_set,historical_cluster_set):
    min_dis=[]
    for c in current_cluster_set:
        dis=[]
        for h in historical_cluster_set:
            distance=np.linalg.norm(c[1]-h[1]) #+np.linalg.norm(c[2]-h[2])
            dis.append(distance)
        min_dis.append(min(dis))
    hausdorff_distance=max(min_dis)
        
    return 1/hausdorff_distance

#Density based clustering function
def cluster(df):
    #optimal eps: https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc

    X=StandardScaler().fit_transform(df)
    dbscan=DBSCAN(eps=1.5) 
    clusters=dbscan.fit(X)
    cluster_labels=clusters.labels_
    cluster_params=[]
    for label in np.unique(cluster_labels):
        if label!=-1:
            cluster=X[cluster_labels==label]
            centroid=np.mean(cluster,axis=0)
            cov_matrix=np.cov(cluster.T)
            cluster_params.append((label,centroid,cov_matrix))
    return X,cluster_params


ponteareas=pd.read_csv('Ponteareas.csv')
#Timestamp as index
ponteareas.set_index(pd.DatetimeIndex(ponteareas['Instante lectura']),inplace=True)
#Drop unnecesary columns
ponteareas=ponteareas.drop(['Chuvia','Instante lectura'],axis=1)
#Change of units to compute FFDI (Forest Fire Danger Index):
#Windspeed in km/h
ponteareas['Velocidade do vento a 10m']=ponteareas['Velocidade do vento a 10m']*3600/1000
#Soil moisture in kPa
ponteareas['Humidade do solo']=ponteareas['Humidade do solo']*1000*9.8/1000

#Contexts: Day and Night.Day from 09:00 to 21:00 and Night from 21:00 to 09:00 
first_period=(ponteareas.index > '2017-09-26 09:00:00.0') & (ponteareas.index <= '2017-09-26 21:00:00.0')
W=len(ponteareas[first_period])
#eliminate some few data points to obtain an exact number of days and nights
ponteareas=ponteareas[(ponteareas.index > '2017-09-26 09:00:00.0') & (ponteareas.index <= '2017-10-24 21:00:00.0')]


intervalos=np.array_split(ponteareas, int(len(ponteareas)/W))
X,first_day_c=cluster(intervalos[1])
C_days=[first_day_c]
X,first_night_c=cluster(intervalos[0])
C_nights=[first_night_c]
risk_all=[]
t=2
while t<(len(intervalos)):
    if bool(t%2): #Days
        meteo=intervalos[t]
        #cluster current interval
        X,current_cluster_set=cluster(meteo)
        #obtain weights with past cluster sets
        weights_list=[]
        for historical_cluster_set in C_days:
            weights_list.append(weight(current_cluster_set,historical_cluster_set))
        weights_list.append(1)
        risk_list=[]
        #compute risk coefficient
        for observation in X:
            anomaly_list=[]
            for c in C_days:
                anomaly_list.append(mahalanomalous(observation,c))
            anomaly_list.append(mahalanomalous(observation,current_cluster_set))
            risk_index=np.sum(np.array(weights_list)*np.array(anomaly_list))/np.sum(weights_list)
            risk_list.append(risk_index)
        risk_all.append(pd.Series(risk_list))
        #append new cluster set and keep only last 10 day periods
        C_days.append(current_cluster_set)
        if len(C_days)>10:
            C_days.pop(0)
            
    else: #Nights
        meteo=intervalos[t]
        #cluster current interval
        X,current_cluster_set=cluster(meteo)
        #obtain weights with past cluster sets
        weights_list=[1]
        for historical_cluster_set in C_nights:
            weights_list.append(weight(current_cluster_set,historical_cluster_set))
        
        risk_list=[]
        #compute risk coefficient
        for observation in X:
            anomaly_list=[]
            for c in C_nights:
                anomaly_list.append(mahalanomalous(observation,c))
            anomaly_list.append(mahalanomalous(observation,current_cluster_set))
            risk_index=np.dot(weights_list,anomaly_list)/np.sum(weights_list)
            risk_list.append(risk_index)
        risk_all.append(pd.Series(risk_list))
        #append new cluster set and keep only last 10 day periods
        C_nights.append(current_cluster_set)
        if len(C_nights)>10:
            C_nights.pop(0)
    t=t+1

#Simplified FFDI computation for long periods without rain (see equations 1 and 2 in paper)
ponteareas['FFDI']=2*np.exp(-0.45+0.987*np.log(0.191/3.52*(ponteareas['Humidade do solo']+104))-0.0345*ponteareas['Humidade relativa media a\xa01.5m']+0.0338*ponteareas['Temperatura media a\xa01.5m']+0.0234*ponteareas['Velocidade do vento a 10m'])
#Fire happened between 14 and 16 of October 2017
ponteareas['incendio']=0
duracion_incendio=(ponteareas.index > '2017-10-14 22:00:00.0') & (ponteareas.index <= '2017-10-16 21:00:00.0')
ponteareas.loc[duracion_incendio,'incendio']=1
#Compare smoothed risk coefficients with FFDIs
risk_df= pd.concat(risk_all).reset_index(drop=True)
smooth_risk=lowess(risk_df,np.array(risk_df.index),return_sorted=False)
fig1, ax1 = plt.subplots(1, 1, figsize = (10, 10))
sns.lineplot(ponteareas.tail(len(risk_df)).index,smooth_risk,ax=ax1)
sns.lineplot(ponteareas.tail(len(risk_df)).index,ponteareas.tail(len(risk_df)).FFDI/100,ax=ax1)
sns.lineplot(ponteareas[ponteareas.incendio==1].index,0,ax=ax1)
plt.legend(['Model','FFDI/100','Fire Duration'])

#Obtain and plot roc curves for smoothed risk coefficients and FFDIs
fpr_1,tpr_1,threshold_1=roc_curve(ponteareas.incendio.tail(len(risk_df)),smooth_risk)
fpr_2,tpr_2,threshold_2=roc_curve(ponteareas.incendio.tail(len(risk_df)),ponteareas.tail(len(risk_df)).FFDI/100)

fig2, ax2 = plt.subplots(1, 1, figsize = (10, 10))
ax2=plt.plot(fpr_1,tpr_1)
ax2=plt.plot(fpr_2,tpr_2)

plt.xlabel('Specifity')
plt.ylabel('Sensitivity')
plt.legend(['Model','FFDI'])

#Obtain and plot Precision-Recall curves for smoothed risk coefficients and FFDIs
prec_1,rec_1,threshold_1=precision_recall_curve(ponteareas.incendio.tail(len(risk_df)),smooth_risk)
prec_2,rec_2,threshold_2=precision_recall_curve(ponteareas.incendio.tail(len(risk_df)),ponteareas.tail(len(risk_df)).FFDI/100)

fig3, ax3 = plt.subplots(1, 1, figsize = (10, 10))
ax3=plt.plot(prec_1,rec_1)
ax3=plt.plot(prec_2,rec_2)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(['Model','FFDI'])

#Obtain auc score and average precision recall score for both models
auc_1=roc_auc_score(ponteareas.incendio.tail(len(risk_df)),smooth_risk)
prec_1=average_precision_score(ponteareas.incendio.tail(len(smooth_risk)),smooth_risk)

auc_2=roc_auc_score(ponteareas.incendio.tail(len(risk_df)),ponteareas.tail(len(risk_df)).FFDI/100)
prec_2=average_precision_score(ponteareas.incendio.tail(len(risk_df)),ponteareas.tail(len(risk_df)).FFDI/100)

print(auc_1,prec_1)
print(auc_2,prec_2)
        
    
    
    







