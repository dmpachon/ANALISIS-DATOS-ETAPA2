#!/usr/bin/env python
# coding: utf-8

# In[55]:


#cargan las librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#llamamos el dataset
datos = pd.read_csv('framingham.csv')
datos.head(5)


# In[73]:


#se toman las variables con las que se va a trabajar y mostrarmos el top 5
datos[['glucose','diabetes']].head()


# In[74]:


#Grafica de dispersion
datos[['glucose','diabetes']].plot.scatter(x='glucose',y='diabetes')


# In[76]:


#cargamos la libreria del modelo de regresion logistica
from sklearn.linear_model import LogisticRegression

# Creamos un modelo de regresión logística

modelo = LogisticRegression()

# Ajustamos el modelo a los datos
x = datos[['glucose']]  
y = datos['diabetes']  

# se elimnan las filas vacias
x = x.dropna() 
y = y.loc[x.index] 

modelo.fit(x, y)

# imprime los parametros

print(f"Pendiente (w): {modelo.coef_}")
print(f"Intercepto (b): {modelo.intercept_}")


# In[77]:


#Grafica de regresión logística 

# puntos de la recta
x = np.linspace(0,datos['heartRate'].max(),100)
y = 1/(1+np.exp(-(w*x+b)))

# grafica de la recta
datos.plot.scatter(x='glucose',y='diabetes')
plt.plot(x, y, '-r') 
plt.ylim(0,datos['diabetes'].max()*1.1)
plt.show()


# In[ ]:




