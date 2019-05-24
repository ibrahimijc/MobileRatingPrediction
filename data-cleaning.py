

import pandas as pd
import numpy as np

def remove_outlier(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3-q1
    low  = q1-1.5*iqr
    high = q3+1.5*iqr
    return df.loc[(df[col] > low) & (df[col] < high)]


df = pd.read_csv('data.csv')

print(df.shape)
df = df[df.columns[df.isnull().sum() < 1500]]
print(df.shape)

print(df.info())

df.drop('Unnamed: 0',axis=1,inplace=True)
df.drop('Features:',axis=1,inplace=True)
df.drop('Weight:',axis=1,inplace=True)
df.drop('None',axis=1,inplace=True)
df.drop('Other features:',axis=1,inplace=True)
df.drop('Other:',axis=1,inplace=True)
df.drop('Notifications:',axis=1,inplace=True)
df.drop('Video recording:',axis=1,inplace=True)


df['Bluetooth:'] = pd.to_numeric(df['Bluetooth:'].astype(str).str[:3], errors='coerce')
df['Bluetooth:'] = df['Bluetooth:'].fillna(2)
print(df['Bluetooth:'].value_counts(dropna=False))



df['Display size:'] = pd.to_numeric(df['Display size:'].astype(str).str[:4], errors='coerce')
df['Display size:'] = df['Display size:'].fillna(df['Display size:'].mean())
print(df['Display size:'].value_counts(dropna=False))

df['Main camera:'] = df['Main camera:'].str.replace('megapixels', '') 
df['Main camera:'] = df['Main camera:'].str.replace('megapixel', '')
df['Main camera:'] = df['Main camera:'].str.replace('VGA', '')
df['Main camera:'] = df['Main camera:'].str.replace('CIF \(288x352\)', '')
df['Main camera:'] = df['Main camera:'].str.replace('megapixles SVGA', '')
df['Main camera:'] = df['Main camera:'].str.replace('megapixles S', '')
df['Main camera:'] = pd.to_numeric(df['Main camera:'])
df['Main camera:'] = df['Main camera:'].fillna(df['Main camera:'].mean())
print(df['Main camera:'].value_counts(dropna=False))



df['USB:'] = df['USB:'].str.replace('USB', '') 
df['USB:'] = df['USB:'].str.replace('Yes', '2.0') 
df['USB:'] = df['USB:'].fillna(2.0)
df['USB:'] = pd.to_numeric(df['USB:'])
print(df['USB:'].value_counts(dropna=False))


df['Resolution:'] = df['Resolution:'].str.replace('pixels', '') 
seperated_Res = df['Resolution:'].str.split('x',n=1,expand=True)
df['Resolution:h'] = seperated_Res[0].fillna(np.nan)
df['Resolution:w'] = seperated_Res[1].fillna(np.nan)
df.drop('Resolution:',axis=1,inplace=True)



seperated_Dimensions_both_units = df['Dimensions:'].str.split('inches',n=1,expand=True)

Dim_inches = seperated_Dimensions_both_units[0]
seperated_Dim = Dim_inches.str.split('x',n=2,expand=True)
print(seperated_Dim)

df['Dimension:1'] = seperated_Dim[0]
df['Dimension:2'] = seperated_Dim[1]
df['Dimension:3'] = seperated_Dim[2]

df.drop('Dimensions:',axis=1,inplace=True)


df['Technology:'], levels = pd.factorize(df['Technology:'])
df['Rear:'], levels2 = pd.factorize(df['Rear:'])
df['Headphones connector:'],levels3 = pd.factorize(df['Headphones connector:'])

print(df['Headphones connector:'].value_counts(dropna=False))

df['4G'] = np.where(df['GSM:'].str.contains('1900'), '1','0' )
df['3G'] = np.where(df['GSM:'].str.contains('1800'), '1','0' )
df['2G'] = np.where(df['GSM:'].str.contains('850'), '1','0' )


df.drop('GSM:',axis=1,inplace=True)





df['Headphones connector:'] = pd.to_numeric(df['Headphones connector:'],errors='coerce')
df['Dimension:3'] = pd.to_numeric(df['Dimension:3'],errors='coerce')
df['Dimension:2'] = pd.to_numeric(df['Dimension:2'],errors='coerce')
df['Dimension:1'] = pd.to_numeric(df['Dimension:1'],errors='coerce')
df['Resolution:h'] = pd.to_numeric(df['Resolution:h'],errors='coerce')
df['Resolution:w'] = pd.to_numeric(df['Resolution:w'],errors='coerce')
df['4G'] = pd.to_numeric(df['4G'],errors='coerce')
df['3G'] = pd.to_numeric(df['3G'],errors='coerce')
df['2G'] = pd.to_numeric(df['2G'],errors='coerce')

df['Rear:'] = np.where(df['Rear:']==-1, None, df['Rear:'])
df['Technology:'] = np.where(df['Technology:']==-1, None, df['Technology:'])
df['Headphones connector:'] = np.where(df['Headphones connector:']==-1, None, df['Headphones connector:'])
df = df.dropna()
print(df.isnull().sum())

print(df.shape)

df['rating_cat'] = pd.cut(df.rating,
                     bins=[0, 3, 5, 7, 8,9],
                     labels=["Poor", "Average", "Good","Very Good","Best"])

df = df.dropna()
print(df.columns)
df.columns = ['bluetooth', 'display_size','headphone','main_camera','rear_camera','display_technology','usb','rating','res_height','res_width','dim_1','dim_2','dim_3','4g','3g','2g','rating_cat']
#df = remove_outlier(df,'main_camera')
#df = remove_outlier(df,'display_size')
df.to_csv('clean.csv')