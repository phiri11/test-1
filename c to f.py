from pyexpat import model
from xml.dom.minidom import ReadOnlySequentialNamedNodeMap
import tensorflow as tf
import numpy as np


celsius = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

#capa = tf.keras.layers.Dense(units=1, input_shape=[1])
#model = tf.keras.Sequential([capa])

hide1 = tf.keras.layers.Dense(units=3, input_shape=[1])
hide2 = tf.keras.layers.Dense(units=3)
hide3 = tf.keras.layers.Dense(units=3)
exit = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([hide1, hide2, hide3, exit])

model.compile(
    loss='mean_squared_error', 
    optimizer=tf.keras.optimizers.Adam(0.1))

print('comenzando a entrenar')
historial = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print('entrenamiento finalizado')

import matplotlib.pyplot as plt
plt.xlabel('Epocas')
plt.ylabel('Error')
plt.plot(historial.history['loss'])
plt.show()

print('predicciones')
resultado = model.predict([100.0])
print('el resultado es: ' + str(resultado) + ' grados fahrenheit')

print('variables internas del modelo')
print(model.get_weights())
