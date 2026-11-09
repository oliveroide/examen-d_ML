# examen-d_ML

Nombre: Oliver Ruiz Beltran
Expediente: 223204115
a = 5
b = 1
Librerias que se ocupan instalar
import numpy as np
import matplotlib.pyplot as plt
import time
import mpmath

a)
¿En qué n se separa de 2ⁿ? En n = 3: se alcanzan 7 de 8 etiquetaciones.
¿Coincide con d_VC? Sí. Con d_VC = 2 la separación debe aparecer en d_VC + 1 = 3. El conteo empírico es 1 + n(n+1)/2 y coincide con Sauer para todo n de 1 a 12.

n= 1 empírico= 2 Sauer= 2 2^n=2
n= 2 empírico= 4 Sauer= 4 2^n=4
n= 3 empírico= 7 Sauer= 7 2^n=8
n= 4 empírico= 11 Sauer= 11 2^n=16
n= 5 empírico= 16 Sauer= 16 2^n=32
n= 6 empírico= 22 Sauer= 22 2^n=64
n= 7 empírico= 29 Sauer= 29 2^n=128
n= 8 empírico= 37 Sauer= 37 2^n=256
n= 9 empírico= 46 Sauer= 46 2^n=512
n=10 empírico= 56 Sauer= 56 2^n=1024
n=11 empírico= 67 Sauer= 67 2^n=2048
n=12 empírico= 79 Sauer= 79 2^n=4096

b)
¿Cuántos cortes hay? Con n = 2000 son 2,001,000.
¿Cuánto tarda con n = 2000? 0.3s
n cortes t_exhaustivo t_erm mismo E_in
100 5050 0.0002 s 0.0012 s True
500 125250 0.0038 s 0.0257 s True
1000 500500 0.0097 s 0.0951 s True
2000 2001000 0.0663 s 0.2947 s True

¿Es el mínimo exacto? ¿Cómo afecta a los resultados?
Si, es exacto, porque prueba todos los bloques consecutivos en el orden por radio. No hay aproximación que afecte los resultados.

c)
¿Qué fracción se reproduce bien con float64?
El 100 % hasta m = 15; después cae hasta 1/200 en m = 25.

¿Desde qué m falla?
Desde m = 16 (164/200).
¿Por qué falla?
Por representación numérica.
De la computadora: con mpmath y 100 cifras se reproducen 200/200 etiquetaciones para todo m de 1 a 25.
m float64 mpmath
1 200/200 200/200
2 200/200 200/200
3 200/200 200/200
4 200/200 200/200
5 200/200 200/200
6 200/200 200/200
7 200/200 200/200
8 200/200 200/200
9 200/200 200/200
10 200/200 200/200
11 200/200 200/200
12 200/200 200/200
13 200/200 200/200
14 200/200 200/200
15 200/200 200/200
16 164/200 200/200
17 126/200 200/200
18 76/200 200/200
19 39/200 200/200
20 27/200 200/200
21 21/200 200/200
22 15/200 200/200
23 3/200 200/200
24 5/200 200/200
25 1/200 200/200

¿Qué pasa cuando A crece y n es chico?
E_in baja de 0.150 a 0.100, pero E_out sube a ≈ 0.49, por lo que es azar

¿Por qué limitar A funciona como regularización?
Porque reduce la capacidad de la clase sin límite en ω, la clase tiene d_VC infinita y puede ajustar cualquier etiquetación.

n A omega E_in E_out(ruido) E_out(limpio)
20 30 18.84 0.150 0.151 0.001
20 300 18.84 0.150 0.151 0.001
20 3000 627.47 0.100 0.494 0.498
20 30000 627.47 0.100 0.494 0.498
50 30 18.47 0.160 0.188 0.051
50 300 18.47 0.160 0.188 0.051
50 3000 18.47 0.160 0.188 0.051
50 30000 18.47 0.160 0.188 0.051
200 30 18.81 0.150 0.154 0.005
200 300 18.81 0.150 0.154 0.005
200 3000 18.81 0.150 0.154 0.005
200 30000 18.81 0.150 0.154 0.005

d)
Lo que aprendiste

La dimensión VC dice que lo que controla la generalización es la capacidad de la clase y no el número de parámetros y en lo que se queda corta es que no dice cuántos datos hacen falta porque es una garantía del peor caso
