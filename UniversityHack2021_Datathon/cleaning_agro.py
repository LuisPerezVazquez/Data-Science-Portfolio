import pandas as pd
import numpy as np
import seaborn as sns
import re
from unidecode import unidecode

dataset_names=['Dataset1.- DatosConsumoAlimentarioMAPAporCCAA.txt','Dataset2.- Precios Semanales Observatorio de Precios Junta de Andalucia.txt','Dataset3a_Datos_MercaMadrid.txt','Dataset3b_Datos_MercaBarna.txt','Dataset4.- Comercio Exterior de España.txt','Dataset5_Coronavirus_cases.txt']

'''Data1
df1=pd.read_csv('repo/'+dataset_names[0],sep='|')

df1.drop(['Unnamed: 10', 'Unnamed: 11'],axis=1,inplace=True)

float_columns=['Volumen (miles de kg)',
       'Valor (miles de €)', 'Precio medio kg', 'Penetración (%)',
       'Consumo per capita', 'Gasto per capita']

df1[float_columns].replace(',','.',regex=True,inplace=True)
df1[float_columns]=df1[float_columns].astype(float)

month_dict={'Enero':'-01','Febrero':'-02','Marzo':'-03','Abril':'-04','Mayo':'-05','Junio':'-06','Julio':'-07','Agosto':'-08','Septiembre':'-09','Octubre':'-10','Noviembre':'-11','Diciembre':'-12'}

df1['Fecha']=df1['Año'].astype(str)+df1.Mes.apply(lambda x: month_dict[x])

df1['Datetime']=pd.DatetimeIndex(df1['Fecha'])
df1.set_index(df1.Datetime,inplace=True)

df1.to_csv('Dataset1_limpio.csv')
'''
##############################################################
'''
df5=pd.read_csv('repo/'+dataset_names[5],sep='|')

df5.dropna(inplace=True)

df5.rename(columns={df5.columns[0]:'dt',df5.columns[6]:'country',df5.columns[7]:df5.columns[7].lower(),df5.columns[8]:'code',df5.columns[9]:'population',df5.columns[10]:'continent',df5.columns[11]:'incidencia_100k'}, inplace=True)

df5.population=df5.population.astype(int)

df5.incidencia_100k=df5.incidencia_100k.str.replace(',','.').astype(float)

df5.dt=pd.to_datetime(df5.dt,format='%d/%m/%Y')

#spain=df5.loc[df5.code=='ESP',['dt','cases','deaths','incidencia_100k']]
#sns.lineplot(x='dt',y='incidencia_100k',data=spain)
#sp=spain.set_index(spain.dt).sort_index().rolling('7d',min_periods=1).mean()
#sns.lineplot(x='dt',y='deaths',data=sp.drop(sp[sp.deaths<=0].index).reset_index())

df5.to_csv('limpios/dataset5_limpio.csv')
'''
##############################################################
'''
df4=pd.read_csv('repo/'+dataset_names[4],sep='|')

df4.rename(columns={'PRODUCT':'products'},inplace=True) #product es un atributo de pd.DataFrame
df4.columns=df4.columns.str.lower()

#partner
df4.drop('partner',axis=1,inplace=True)

#value
df4=df4[df4.value!=':']
df4.value=df4.value.str.replace(' ','') #no estoy seguro de esto
df4.value=df4.value.astype(int)
#reporter
df4.reporter=df4.reporter.apply(lambda x: re.sub(r' +\([^)]*\)', '',x))
df4.reporter[df4.reporter=='European Union - 27 countries']='European Union'

#indicators
df4.indicators.replace('QUANTITY_IN_100KG','quantity_100kg',inplace=True)
df4.indicators.replace('VALUE_IN_EUROS','euros',inplace=True)

#product
def limpiar_tipo(tipo):
     quitar = ["fresh or chilled ",'"fresh or dried ',',']
     for p in quitar:
         tipo = tipo.lower().replace(p,"")
     tipo = tipo.split("(")[0]
     tipo = tipo.split('"')[0]
     return tipo
df4['clean_products']=df4.products.apply(lambda x : limpiar_tipo(x))



#period
month_dict={'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}

df4_total=df4[df4.period.str.split('.').apply(lambda x: len(x))==3] #hay datos que son agregados de todo el año y su period es del estilo 'Jan.-Dec. 2018'
df4_total['year']=df4_total.period.str.split('.').apply(lambda x: x[-1])
df4_total.drop('period',axis=1,inplace=True)

df4=df4[df4.period.str.split('.').apply(lambda x: len(x))!=3]

df4['month']=df4.period.str.split('.').apply(lambda x: month_dict[x[0]])
df4['year']=df4.period.str.split('.').apply(lambda x: x[-1])
df4['dt']=pd.to_datetime(df4.year+'-'+df4.month)
df4.drop('period',axis=1,inplace=True)

df4_pivoted=df4.pivot_table('value',['dt','reporter','products','flow'], 'indicators').reset_index()

#sns.lineplot(x='dt',y='value',data=df4[(df4.clean_products=='bananas fresh ')&(df4.indicators=='euros')],hue='flow')
df4.to_csv('limpios/dataset4_limpio.csv')
df4_total.to_csv('limpios/dataset4_totales.csv')
df4_pivoted.to_csv('limpios/dataset4_pivotado.csv')
'''

##########################################################

df3a=pd.read_csv('repo/'+dataset_names[2],sep='|')

df3a.rename(columns={'product':'products'},inplace=True) #product es un atributo de pd.DataFrame
df3a.columns=df3a.columns.str.lower()


df3a.drop(df3a[df3a.products=='Ã\x91AME O YAME'].index,inplace=True)

df3a.drop('unidad',axis=1,inplace=True) #solo tiene kg

float_columns=['price_min','price_mean','price_max']

df3a[float_columns]=df3a[float_columns].replace(',','.',regex=True)
df3a[float_columns]=df3a[float_columns].astype(float)

df3a['dt']=pd.to_datetime((df3a.year).astype(str)+'-'+(df3a.month).astype(str))

df3a.origen[df3a.origen=='LA CORUÃ\x91A']='LA CORUNA'
df3a.origen[df3a.origen=='TÃ\x9aNEZ']='TUNEZ'
df3a.origen[df3a.origen=='GRAN BRETAÃ\x91A']='GRAN BRETANA'


provincias_sucias = np.char.lower(np.array(['Alava','Albacete','Alicante','Almería','Asturias','Avila','Badajoz','Barcelona','Burgos','Cáceres',
'Cádiz','Cantabria','Castellón','Ciudad Real','Córdoba','La Coruña','Cuenca','Gerona','Granada','Guadalajara',
'Guipúzcoa','Huelva','Huesca','Baleares','Jaén','León','Lérida','Lugo','Madrid','Málaga','Murcia','Navarra',
'Orense','Palencia','Las Palmas','Pontevedra','La Rioja','Salamanca','Segovia','Sevilla','Soria','Tarragona',
'Sta.Cruz de Tenerife','Teruel','Toledo','Valencia','Valladolid','Vizcaya','Zamora','Zaragoza']))

provincias=[unidecode(p) for p in provincias_sucias]

def es_nacional(x):
    if x.lower() in provincias:
        es=True 
    else:
        es=False
    return es

df3a['nacional']=df3a.origen.apply(lambda x: es_nacional(x))



''' para europeo o no?
europa=['FRANCIA', 'PORTUGAL', 
       'ITALIA',  'BELGICA', 'HOLANDA',
       'ESTADOS UNIDOS', 'HONDURAS', 'IRLANDA', 
       'GRAN BRETAÃ\x91A', 'GRECIA',  'POLONIA', 'ALEMANIA', 'DINAMARCA', 
       'AUSTRIA',
       'SUIZA',  'BULGARIA',
        'OTROS PAIS. EUROPEOS']
'''

################################
df3b=pd.read_csv('repo/'+dataset_names[3],sep='|')

df3b.rename(columns={'product':'products'},inplace=True) #product es un atributo de pd.DataFrame
df3b.columns=df3b.columns.str.lower()


df3b.drop(df3b[df3b.products=='Ã\x91AME O YAME'].index,inplace=True)

df3b.drop('unidad',axis=1,inplace=True) #solo tiene kg

float_columns=['price_mean']

df3b[float_columns]=df3b[float_columns].replace(',','.',regex=True)
df3b[float_columns]=df3b[float_columns].astype(float)

df3b['dt']=pd.to_datetime((df3b.year).astype(str)+'-'+(df3b.month).astype(str))

df3b.origen[df3b.origen=='LA CORUÃ\x91A']='LA CORUNA'
df3b.origen[df3b.origen=='TÃ\x9aNEZ']='TUNEZ'
df3b.origen[df3b.origen=='GRAN BRETAÃ\x91A']='GRAN BRETANA'


provincias_sucias = np.char.lower(np.array(['Alava','Albacete','Alicante','Almería','Asturias','Avila','Badajoz','Barcelona','Burgos','Cáceres',
'Cádiz','Cantabria','Castellón','Ciudad Real','Córdoba','La Coruña','Cuenca','Gerona','Granada','Guadalajara',
'Guipúzcoa','Huelva','Huesca','Baleares','Jaén','León','Lérida','Lugo','Madrid','Málaga','Murcia','Navarra',
'Orense','Palencia','Las Palmas','Pontevedra','La Rioja','Salamanca','Segovia','Sevilla','Soria','Tarragona',
'Sta.Cruz de Tenerife','Teruel','Toledo','Valencia','Valladolid','Vizcaya','Zamora','Zaragoza']))

provincias=[unidecode(p) for p in provincias_sucias]


df3b['nacional']=df3b.origen.apply(lambda x: es_nacional(x))





df3a=pd.DataFrame(df3a.groupby(['dt','products']).mean()).reset_index()

df3aprev=df3a[['dt','price_mean','volumen','products']]

df3aprev.dt=pd.to_datetime(df3a.dt)+pd.DateOffset(months=1)

df3a_join=df3a.merge(right=df3aprev,suffixes=('','_prev'),how='left',left_on=['dt','products'],right_on=['dt','products'])

df3a_join['price_change']=(self_join['price_mean']-df3a_join['price_mean_prev'])/df3a_join['price_mean_prev']



df3a_join.loc[df3a_join['price_change']<0,'price_alert']='Bajada de precio (menor al 0%)'
df3a_join.loc[(df3a_join['price_change']>=0)&(df3a_join['price_change']<0.05),'price_alert']='Subida de precio pequeña(0-5%)'
df3a_join.loc[(df3a_join['price_change']>0.05)&(df3a_join['price_change']<0.15),'price_alert']='Subida de precio moderada(5-15%)'
df3a_join.loc[(df3a_join['price_change']>0.15),'price_alert']='Subida de precio alta(superior al 15%)'
df3a_join.loc[df3a_join['price_change'].isna(),'price_alert']='Sin Datos'

df3a_join['Mercado']='MercaMadrid'



product_series=df3b.producto.str.split(' ').apply(lambda x: x[0])
product_series[df3b.producto.str.split(' ').apply(lambda x: x[0])=='FRUTAS']=df3b.producto[df3b.producto.str.split(' ').apply(lambda x: x[0])=='FRUTAS'].str.split(' ').apply(lambda x: x[1])

df3b.producto=product_series
df3b.producto[df3b['producto']=='MANZANA']='MANZANAS'
df3b.producto[df3b['producto']=='MELOCOTON']='MELOCOTONES'
df3b.producto[df3b['producto']=='COL']='COLES BRUSELAS'
df3b.producto[df3b['producto']=='MANDARINA']='MANDARINAS'
df3b.producto[df3b['producto']=='JUDIA']='JUDIAS'
df3b.producto[df3b['producto']=='HIGO']='HIGOS'

##poner todo en plural
singular_1=[p[:-1] for p in df3a_join.producto.unique()]
singular_2=[p[:-2] for p in df3a_join.producto.unique()]
df3b_join.producto[df3b_join.producto.isin(singular_1)]=df3b_join.producto[df3b_join.producto.isin(singular_1)]+'S'
df3b_join.producto[df3b_join.producto.isin(singular_2)]=df3b_join.producto[df3b_join.producto.isin(singular_2)]+'ES'

singular_1=[p[:-1] for p in df3b_join.producto.unique()]
singular_2=[p[:-2] for p in df3b_join.producto.unique()]
df3a_join.producto[df3a_join.producto.isin(singular_1)]=df3a_join.producto[df3a_join.producto.isin(singular_1)]+'S'
df3a_join.producto[df3a_join.producto.isin(singular_2)]=df3a_join.producto[df3a_join.producto.isin(singular_2)]+'ES'

df3b_final=df3b_join[df3b_join.producto.isin(df3a_join.producto.unique())]
df3a_final=df3a_join[df3a_join.producto.isin(df3b_join.producto.unique())]

df3b_final.to_csv('limpios/dataset3b_filtrado.csv')
df3a_final.to_csv('limpios/dataset3a_filtrado.csv')


df3=pd.concat([df3a_final,df3b_final])

df3.to_csv('limpios/dataset3.csv')


#####################################

df2.dt=df2.dt.astype('datetime64[M]')

####################################

productos=df1[df1.dt>'2020-07'].producto.unique()


df1.loc[df1['producto']=='PATATAS','misc']='Esta frase sobre las patatas es la hostia'
df3b.producto[df3b['producto']=='MELOCOTON']='MELOCOTONES'
df3b.producto[df3b['producto']=='COL']='COLES BRUSELAS'
df3b.producto[df3b['producto']=='MANDARINA']='MANDARINAS'
df3b.producto[df3b['producto']=='JUDIA']='JUDIAS'
df3b.producto[df3b['producto']=='HIGO']='HIGOS'




