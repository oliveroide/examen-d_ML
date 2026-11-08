# examen-d_ML

Nombre: Oliver Ruiz Beltran
Expediente: 223204115
a = 5
b = 1

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

¿Cuántos cortes hay? Con n = 2000 son 2,001,000.
¿Cuánto tarda con n = 2000? 0.3s
n cortes t_exhaustivo t_erm mismo E_in
100 5050 0.0002 s 0.0012 s True
500 125250 0.0038 s 0.0257 s True
1000 500500 0.0097 s 0.0951 s True
2000 2001000 0.0663 s 0.2947 s True
