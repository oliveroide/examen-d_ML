import numpy as np
import matplotlib.pyplot as plt

SEMILLA = 223204115   # expediente 
import time
import mpmath
# ---------- Clase de anillos: h(x) = +1 si r1 <= ||x|| <= r2, -1 en otro caso ----------

def error(puntos, y, r1, r2):
    """Fracción de errores del anillo [r1, r2]."""
    ro = np.linalg.norm(puntos, axis=1)
    prediccion = np.where((ro >= r1) & (ro <= r2), 1, -1)
    return np.mean(prediccion != y)


def realizable(puntos, y):
    """True si algún anillo produce estas etiquetas (y es un arreglo de +1/-1).
    Al ordenar por radio, los positivos deben formar un solo bloque."""
    ro = np.linalg.norm(puntos, axis=1)
    orden = np.argsort(ro)
    ro, y = ro[orden], y[orden]

    for i in range(len(ro) - 1):              # mismo radio con etiquetas distintas
        if ro[i] == ro[i + 1] and y[i] != y[i + 1]:
            return False

    positivos = np.flatnonzero(y == 1)        # posiciones de los +1
    if len(positivos) == 0:
        return True
    return positivos[-1] - positivos[0] + 1 == len(positivos)   # sin huecos


def erm(puntos, y):
    """Prueba todos los bloques consecutivos (O(n^2)) y se queda con el de mayor suma de y."""
    ro = np.linalg.norm(puntos, axis=1)
    orden = np.argsort(ro)
    ro, y = ro[orden], y[orden]

    mejor = 0                                  
    for i in range(len(y)):                    # el bloque empieza en i
        suma = 0
        for j in range(i, len(y)):             # y termina en j
            suma += y[j]
            if suma > mejor:
                mejor, r1, r2 = suma, ro[i], ro[j]
    return r1, r2

def erm_exhaustivo(puntos, y):
    """Prueba TODOS los bloques consecutivos con sumas prefijas (O(n^2))."""
    ro = np.linalg.norm(puntos, axis=1)
    orden = np.argsort(ro)
    ro, y = ro[orden], y[orden]

    prefijas = np.concatenate(([0], np.cumsum(y)))
    suma = np.triu(prefijas[None, 1:] - prefijas[:-1, None])   # suma[i, j] = suma de y[i..j], i <= j
    i, j = np.unravel_index(np.argmax(suma), suma.shape)
    if suma[i, j] <= 0:                      
        return 1, 0
    return ro[i], ro[j]

# ---------- (a) Contar etiquetaciones ----------

def experimento_a():
    rng = np.random.RandomState(SEMILLA)
    ns = list(range(1, 13))
    empirico = []
    for n in ns:
        # Las 2^n etiquetaciones con -1/+1, una por fila.
        etiquetaciones = 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) - 1
        mejor = 0
        for _ in range(100):                  # 100 acomodos de puntos
            puntos = rng.normal(0, 1, size=(n, 2))
            cuenta = sum(realizable(puntos, y) for y in etiquetaciones)
            mejor = max(mejor, cuenta)        # m_H(n) es el máximo sobre acomodos
        empirico.append(mejor)
        print(f"n={n:2d}  empírico={mejor:3d}  Sauer={1 + n + n * (n - 1) // 2:3d}  2^n={2 ** n}")

    sauer = [1 + n + n * (n - 1) // 2 for n in ns]
    plt.figure(figsize=(7, 4.5))
    plt.plot(ns, [2 ** n for n in ns], label="2^n")
    plt.plot(ns, sauer, label="Cota de Sauer")
    plt.plot(ns, empirico, "o", label="Conteo empírico")
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("Número de etiquetaciones")
    plt.title("Etiquetaciones realizables con anillos")
    plt.legend()
    plt.grid(True)
    plt.savefig("a_etiquetaciones.png")
    plt.show()


# ---------- (b) Curva de aprendizaje con ERM ----------

def experimento_b():
    rng = np.random.RandomState(SEMILLA)

    def muestra(n):
        """n puntos uniformes en [-1,1]^2 etiquetados con el anillo objetivo."""
        puntos = rng.normal(0, 1, size=(n, 2))
        ro = np.linalg.norm(puntos, axis=1)
        return puntos, np.where((ro >= 1) & (ro <= 2), 1, -1)

    def con_ruido(y):
        """Invierte exactamente el 10 % de las etiquetas."""
        y = y.copy()
        malas = rng.choice(len(y), size=round(0.1 * len(y)), replace=False)
        y[malas] = -y[malas]
        return y

    tamanos = [10, 20, 50, 100, 200, 500, 1000, 2000]
    dif_prom, dif_desv, cotas = [], [], []
    print("   n    E_in  E_out(ruido)   dif    desv  E_out(limpio)  cota P1")
    for n in tamanos:
        e_in, e_out, e_limpio = [], [], []
        for _ in range(30):                   # 30 repeticiones
            x, limpias = muestra(n)
            y = con_ruido(limpias)
            r1, r2 = erm(x, y)
            e_in.append(error(x, y, r1, r2))

            x_prueba, limpias_prueba = muestra(10000)
            e_out.append(error(x_prueba, con_ruido(limpias_prueba), r1, r2))
            e_limpio.append(error(x_prueba, limpias_prueba, r1, r2))

        dif = np.array(e_out) - np.array(e_in)
        cota = np.sqrt(8 / n * np.log(4 * (1 + 2 * n ** 2 + n) / 0.05))
        dif_prom.append(dif.mean())
        dif_desv.append(dif.std())
        cotas.append(cota)
        print(f"{n:5d}  {np.mean(e_in):.4f}  {np.mean(e_out):.4f}      "
              f"{dif.mean():.4f}  {dif.std():.4f}  {np.mean(e_limpio):.4f}        {cota:.3f}")

    plt.figure(figsize=(7, 4.5))
    plt.plot(tamanos, dif_prom, "o-", label="Eout - Ein (promedio)")
    plt.plot(tamanos, cotas, "o-", label="Cota P1")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("Eout - Ein")
    plt.title("Curva de aprendizaje")
    plt.legend()
    plt.grid(True)
    plt.savefig("b_curva_aprendizaje.png")
    plt.show()
    
def comparar_tiempos():
    rng = np.random.default_rng(SEMILLA)
    print("      n   cortes     t_exhaustivo   t_erm   mismo E_in")
    for n in [100, 500, 1000, 2000]:
        puntos = rng.normal(0, 1, size=(n, 2))
        ro = np.linalg.norm(puntos, axis=1)
        y = np.where((ro >= 1) & (ro <= 2), 1, -1)
        malas = rng.choice(n, size=round(0.1 * n), replace=False)
        y[malas] = -y[malas]

        t0 = time.time()
        r_exh = erm_exhaustivo(puntos, y)
        t_exh = time.time() - t0

        t0 = time.time()
        r_erm = erm(puntos, y)
        t_erm = time.time() - t0

        igual = np.isclose(error(puntos, y, *r_exh), error(puntos, y, *r_erm))
        print(f"{n:7d}  {n * (n + 1) // 2:8d}  {t_exh:12.4f} s  {t_erm:9.4f} s   {igual}")

# ---------- (c)(i) Construcción P3: float64 contra precisión arbitraria ----------

def experimento_c1():
    rng = np.random.default_rng(SEMILLA)
    mpmath.mp.dps = 100                       # 100 cifras decimales
    ms = list(range(1, 26))
    frac_float, frac_mp = [], []
    print("  m   float64   mpmath")
    for m in ms:
        ok_float, ok_mp = 0, 0
        for _ in range(200):                  # 200 etiquetaciones por cada m
            y = rng.choice([-1, 1], size=m).tolist()
            # 1 + sum c_i 10^i con c_i = (1 - y_i)/2, entero exacto de Python
            entero = 1 + sum((1 - yi) // 2 * 10 ** i for i, yi in enumerate(y, 1))

            omega = np.pi * entero            # float64
            ok_float += all((np.sin(omega * 10.0 ** -i) > 0) == (yi == 1)
                            for i, yi in enumerate(y, 1))

            omega_mp = mpmath.pi * entero    
            ok_mp += all((mpmath.sin(omega_mp * mpmath.mpf(10) ** -i) > 0) == (yi == 1)
                         for i, yi in enumerate(y, 1))
        frac_float.append(ok_float / 200)
        frac_mp.append(ok_mp / 200)
        print(f"{m:3d}   {ok_float:3d}/200   {ok_mp:3d}/200")

    plt.figure(figsize=(7, 4.5))
    plt.plot(ms, frac_float, "o-", label="float64")
    plt.plot(ms, frac_mp, "o-", label="mpmath")
    plt.xlabel("m")
    plt.ylabel("Fracción reproducida")
    plt.title("Construcción P3")
    plt.legend()
    plt.grid(True)
    plt.savefig("c1_float64_vs_mpmath.png")
    plt.show()


# ---------- (c)(ii) Buscar omega en una rejilla y limitar A ----------

def experimento_c2():
    rng = np.random.default_rng(SEMILLA)
    paso = 0.01                               

    def f(x):
        return np.where(np.sin(6 * np.pi * x) > 0, 1, -1)

    def con_ruido(y):
        """Invierte exactamente el 15 % de las etiquetas."""
        y = y.copy()
        malas = rng.choice(len(y), size=round(0.15 * len(y)), replace=False)
        y[malas] = -y[malas]
        return y

    def mejor_omega(x, y, A):
        """omega en {paso, 2 paso, ..., A} con menor E_in (en empates, el menor omega)."""
        k_max = round(A / paso)
        mejor, menor_error = None, len(x) + 1
        for inicio in range(1, k_max + 1, 20000):      
            omegas = paso * np.arange(inicio, min(inicio + 20000, k_max + 1))
            prediccion = np.where(np.sin(np.outer(omegas, x)) > 0, 1, -1)
            errores = np.sum(prediccion != y, axis=1)
            i = np.argmin(errores)                     # primer mínimo = menor omega
            if errores[i] < menor_error:
                mejor, menor_error = omegas[i], errores[i]
        return mejor, menor_error / len(x)

    tamanos = [20, 50, 200]
    limites = [30, 300, 3000, 30000]
    e_limpio_por_n = {n: [] for n in tamanos}
    print("   n      A     omega    E_in  E_out(ruido)  E_out(limpio)")
    for n in tamanos:
        x = rng.uniform(0, 1, n)              # misma muestra para todos los A
        y = con_ruido(f(x))
        x_prueba = rng.uniform(0, 1, 10000)
        limpias_prueba = f(x_prueba)
        y_prueba = con_ruido(limpias_prueba)
        for A in limites:
            omega, e_in = mejor_omega(x, y, A)
            prediccion = np.where(np.sin(omega * x_prueba) > 0, 1, -1)
            e_ruido = np.mean(prediccion != y_prueba)
            e_limpio = np.mean(prediccion != limpias_prueba)
            e_limpio_por_n[n].append(e_limpio)
            print(f"{n:4d}  {A:5d}  {omega:8.2f}  {e_in:.3f}  {e_ruido:.3f}         {e_limpio:.3f}")

    plt.figure(figsize=(7, 4.5))
    for n in tamanos:
        plt.plot(limites, e_limpio_por_n[n], "o-", label=f"n = {n}")
    plt.xscale("log")
    plt.xlabel("A")
    plt.ylabel("Eout (contra f sin ruido)")
    plt.title("Efecto del límite A")
    plt.legend()
    plt.grid(True)
    plt.savefig("c2_efecto_A.png")
    plt.show()


if __name__ == "__main__":
    experimento_a()
    experimento_b()
    comparar_tiempos()
    experimento_c1()
    experimento_c2()