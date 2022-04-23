import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


df = pd.read_csv('ts.csv')
dfT = df.sort_values('tiempo')  #ordeno por tiempo
dfT.head()
df_muestra = dfT.sample(200)
plt.scatter(df_muestra['tiempo'], df_muestra['magnitud'])
plt.show()
##regresion lineal
RegLineal = LinearRegression()
RegLineal.fit(dfT[['tiempo']], dfT['magnitud'])
print('el resultado del coef es=',RegLineal.coef_)
print('el resultado del inter es=',RegLineal.intercept_)

x_linspace = np.linspace(dfT['tiempo'].min(), dfT['tiempo'].max(), 100).reshape(-1, 1)
y_pred = RegLineal.predict(x_linspace)
np.linspace(dfT['tiempo'].min(), dfT['tiempo'].max(), 100).shape
plt.plot(x_linspace, y_pred, 'r-')
plt.scatter(dfT['tiempo'], dfT['magnitud'])
plt.xlabel('tiempo')
plt.ylabel('magnitud')
plt.show()

pol= PolynomialFeatures(degree=1)
model= LinearRegression()
x_poly= dfT[['tiempo']].values.reshape(-1, 1)
dfT_poly= pol.fit_transform(x_poly)
model.fit(dfT_poly, dfT['magnitud'])

dfT_poly_pred= pol.fit_transform(x_linspace)

x_linspace = np.linspace(dfT['tiempo'].min(), dfT['tiempo'].max(), 100).reshape(-1, 1)
y_pred = model.predict(dfT_poly_pred)

plt.plot(x_linspace, y_pred, 'r--')
plt.scatter(dfT['tiempo'], dfT['magnitud'])
plt.title('Prediccion con polinomios de grado 1')
plt.legend(['Prediccion', 'Datos'])
plt.xlabel('tiempo')
plt.ylabel('magnitud')
plt.show()


pol= PolynomialFeatures(degree=3)
model= LinearRegression()
x_poly= dfT[['tiempo']].values.reshape(-1, 1)
dfT_poly= pol.fit_transform(x_poly)
model.fit(dfT_poly, dfT['magnitud'])

dfT_poly_pred= pol.fit_transform(x_linspace)

x_linspace = np.linspace(dfT['tiempo'].min(), dfT['tiempo'].max(), 100).reshape(-1, 1)
y_pred = model.predict(dfT_poly_pred)

#penalizacion lasso y ridge
ridgereg = Ridge(alpha=10,normalize=True)
ridgereg.fit(dfT,dfT['magnitud'])
y_pred1 = ridgereg.predict(dfT)

modelo = Lasso(alpha=0.024,normalize=True)
modelo.fit(dfT,dfT['magnitud'])
y_pred2 = modelo.predict(dfT)

plt.tight_layout()
plt.plot(x_linspace, y_pred, 'r')
plt.plot(dfT['tiempo'],y_pred1, 'b--')
plt.plot(dfT['tiempo'],y_pred2, 'g-')
plt.scatter(dfT['tiempo'], dfT['magnitud'], color='black')
plt.title('Prediccion con polinomios de grado 3')
plt.legend(['Prediccion','Ridge','Lasso','Datos'])
plt.xlabel('tiempo')
plt.ylabel('magnitud')
plt.show()


pol= PolynomialFeatures(degree=5)
model= LinearRegression()
x_poly= dfT[['tiempo']].values.reshape(-1, 1)
dfT_poly= pol.fit_transform(x_poly)
model.fit(dfT_poly, dfT['magnitud'])

dfT_poly_pred= pol.fit_transform(x_linspace)

x_linspace = np.linspace(dfT['tiempo'].min(), dfT['tiempo'].max(), 100).reshape(-1, 1)
y_pred = model.predict(dfT_poly_pred)

plt.plot(x_linspace, y_pred, 'r--')
plt.scatter(dfT['tiempo'], dfT['magnitud'])
plt.title('Prediccion con polinomios de grado 5')
plt.legend(['Prediccion', 'Datos'])
plt.xlabel('tiempo')
plt.ylabel('magnitud')
plt.show()

pol= PolynomialFeatures(degree=7)
model= LinearRegression()
x_poly= dfT[['tiempo']].values.reshape(-1, 1)
dfT_poly= pol.fit_transform(x_poly)
model.fit(dfT_poly, dfT['magnitud'])

dfT_poly_pred= pol.fit_transform(x_linspace)

x_linspace = np.linspace(dfT['tiempo'].min(), dfT['tiempo'].max(), 100).reshape(-1, 1)
y_pred = model.predict(dfT_poly_pred)

#Penalizacion lasso y ridge )
ridgereg = Lasso(alpha=0.0275,normalize=True)
ridgereg.fit(dfT,dfT['magnitud'])
y_pred1 = ridgereg.predict(dfT)


#plt.plot(x_linspace, y_pred, 'r--')
plt.tight_layout()
plt.plot.size=10
plt.plot(dfT['tiempo'],y_pred1, 'r--')
plt.scatter(dfT['tiempo'], dfT['magnitud'])
plt.title('Prediccion con polinomios de grado 7')
plt.legend(['Prediccion', 'Datos'])
plt.xlabel('tiempo')
plt.ylabel('magnitud')
plt.show()
