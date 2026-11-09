import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import check_grad, minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from statsmodels.miscmodels.ordinal_model import OrderedModel

SEMILLA = 223204115   
URL = "https://archive.ics.uci.edu/static/public/1/abalone.zip"
K = 5                 
NUMERICAS = ["Length", "Diameter", "Height", "Whole", "Shucked", "Viscera", "Shell"]
ATRIBUTOS = NUMERICAS + ["Sex_M", "Sex_F"]     


# =====================================================================
# (a) Datos
# =====================================================================


def cargar_datos():
    """Lee Abalone y elimina alturas inválidas y crea los 5 niveles."""

    datos = pd.read_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "abalone.data"),
        header=None,
        names=[
            "Sex", "Length", "Diameter", "Height", "Whole",
            "Shucked", "Viscera", "Shell", "Rings"
        ]
    )

    print("Filas leídas:", len(datos))
    print("Filas con altura = 0 (se eliminan):", (datos["Height"] == 0).sum())

    datos = datos[datos["Height"] > 0].reset_index(drop=True)

    datos["Nivel"] = np.digitize(datos["Rings"], [8, 10, 12, 15]) + 1

    datos["Sex_M"] = (datos["Sex"] == "M").astype(int)
    datos["Sex_F"] = (datos["Sex"] == "F").astype(int)

    print("Filas finales:", len(datos))
    print("Conteo por Sex:", datos["Sex"].value_counts().to_dict())
    print("Conteo por nivel:", datos["Nivel"].value_counts().sort_index().to_dict())

    return datos


def preparar(datos):
    """Prueba 25 % estratificada y estandarización con la media y desviación del entrenamiento."""
    entrena, prueba = train_test_split(datos, test_size=0.25, stratify=datos["Nivel"],
                                       random_state=SEMILLA)
    media = entrena[NUMERICAS].mean()
    desv = entrena[NUMERICAS].std()

    def matriz(tabla):
        X = tabla[ATRIBUTOS].copy()
        X[NUMERICAS] = (X[NUMERICAS] - media) / desv
        return X.to_numpy(), tabla["Nivel"].to_numpy()

    Xe, ye = matriz(entrena)
    Xp, yp = matriz(prueba)
    print("Entrenamiento:", len(ye), " Prueba:", len(yp))
    return Xe, ye, Xp, yp


# =====================================================================
# (b) Modelo ordinal logístico propio
# =====================================================================
# P(Y <= k | x) = sigmoide(theta_k - w'x).  Parámetros: w (d) y u (K-1), con
# theta_1 = u_0 y theta_k = theta_{k-1} + exp(u_{k-1}), así los umbrales quedan ordenados.

def umbrales(u):
    return np.cumsum(np.concatenate(([u[0]], np.exp(u[1:]))))


def nll_y_gradiente(parametros, X, y):
    """Log-verosimilitud negativa promedio y su gradiente (y toma valores 1..K)."""
    d = X.shape[1]
    w, u = parametros[:d], parametros[d:]
    theta_ext = np.concatenate(([-np.inf], umbrales(u), [np.inf]))   # theta_0 = -inf, theta_K = +inf
    eta = X @ w
    a = theta_ext[y] - eta                    # theta_k - w'x
    b = theta_ext[y - 1] - eta                # theta_{k-1} - w'x
    P = np.maximum(expit(a) - expit(b), 1e-12)
    nll = -np.mean(np.log(P))

    # Gradiente respecto a w:  (1 - sigmoide(a) - sigmoide(b)) x
    factor = 1 - expit(a) - expit(b)
    g_w = X.T @ factor / len(y)

    # Gradiente respecto a cada umbral theta_1..theta_{K-1}
    d_a = expit(a) * (1 - expit(a)) / P
    d_b = expit(b) * (1 - expit(b)) / P
    g_theta = np.zeros(K + 1)
    np.add.at(g_theta, y, d_a)                # theta_k aparece como cota superior
    np.add.at(g_theta, y - 1, -d_b)           # theta_{k-1} aparece como cota inferior
    g_theta = -g_theta[1:K] / len(y)

    # Regla de la cadena: de theta a u
    g_u = np.cumsum(g_theta[::-1])[::-1] * np.concatenate(([1], np.exp(u[1:])))
    return nll, np.concatenate([g_w, g_u])


def nll(parametros, X, y):
    return nll_y_gradiente(parametros, X, y)[0]


def gradiente(parametros, X, y):
    return nll_y_gradiente(parametros, X, y)[1]


def ajustar_ordinal(X, y):
    """Devuelve (w, theta). Inicia con w = 0 y umbrales a partir de las proporciones."""
    d = X.shape[1]
    acumulada = np.cumsum(np.bincount(y, minlength=K + 1)[1:]) / len(y)      # P(Y <= k)
    theta0 = np.log(acumulada[:-1] / (1 - acumulada[:-1]))
    u0 = np.concatenate(([theta0[0]], np.log(np.diff(theta0))))
    resultado = minimize(nll_y_gradiente, np.concatenate([np.zeros(d), u0]), args=(X, y),
                         jac=True, method="L-BFGS-B",
                         options={"maxiter": 5000, "gtol": 1e-10, "ftol": 1e-14})
    return resultado.x[:d], umbrales(resultado.x[d:])


def probabilidades_ordinal(X, w, theta):
    """Matriz (n, K) con P(Y = k | x)."""
    theta_ext = np.concatenate(([-np.inf], theta, [np.inf]))
    acumulada = expit(theta_ext[None, :] - (X @ w)[:, None])
    return np.diff(acumulada, axis=1)


def verificar_ordinal(Xe, ye):
    d = Xe.shape[1]
    rng = np.random.default_rng(SEMILLA)
    p = rng.normal(0, 0.5, size=d + K - 1)
    error = check_grad(nll, gradiente, p, Xe, ye)
    print(f"check_grad: error = {error:.2e}  (norma del gradiente = {np.linalg.norm(gradiente(p, Xe, ye)):.3f})")

    w, theta = ajustar_ordinal(Xe, ye)
    sm = OrderedModel(ye, Xe, distr="logit").fit(method="bfgs", disp=False, maxiter=5000, gtol=1e-7)
    w_sm = np.asarray(sm.params)[:d]
    theta_sm = umbrales(np.asarray(sm.params)[d:])      # statsmodels usa la misma parametrización
    print(f"Máx |w - w_statsmodels| = {np.max(np.abs(w - w_sm)):.2e}")
    print(f"Máx |theta - theta_statsmodels| = {np.max(np.abs(theta - theta_sm)):.2e}")


# =====================================================================
# (c) Los seis modelos
# =====================================================================

def moda(P):
    return np.argmax(P, axis=1) + 1


def mediana(P):
    """Primer nivel donde la probabilidad acumulada llega a 0.5."""
    return np.argmax(np.cumsum(P, axis=1) >= 0.5, axis=1) + 1


def redondear(prediccion):
    return np.clip(np.round(prediccion), 1, K).astype(int)


def metricas(y, prediccion):
    return (accuracy_score(y, prediccion), mean_absolute_error(y, prediccion),
            cohen_kappa_score(y, prediccion, weights="quadratic"))


# Cada modelo: ajustar(X, y, **parametros) -> modelo;  predecir(modelo, X) -> {variante: niveles}

def ajustar_m1(X, y):
    return LinearRegression().fit(X, y)


def predecir_m1(m, X):
    return {"redondeo": redondear(m.predict(X))}


def ajustar_m2(X, y, C):
    return LogisticRegression(C=C, max_iter=2000).fit(X, y)


def predecir_m2(m, X):
    P = m.predict_proba(X)
    return {"moda": moda(P), "mediana": mediana(P)}


def ajustar_m3(X, y):
    return ajustar_ordinal(X, y)


def predecir_m3(m, X):
    P = probabilidades_ordinal(X, *m)
    return {"moda": moda(P), "mediana": mediana(P)}


def clasificador(learning_rate, max_leaf_nodes):
    return HistGradientBoostingClassifier(learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                                          max_iter=200, random_state=SEMILLA)


def ajustar_m4(X, y, learning_rate, max_leaf_nodes):
    return clasificador(learning_rate, max_leaf_nodes).fit(X, y)


def predecir_m4(m, X):
    P = m.predict_proba(X)
    return {"moda": moda(P), "mediana": mediana(P)}


def ajustar_m5(X, y, learning_rate, max_leaf_nodes):
    return HistGradientBoostingRegressor(learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                                         max_iter=200, random_state=SEMILLA).fit(X, y)


def predecir_m5(m, X):
    return {"redondeo": redondear(m.predict(X))}


def ajustar_m6(X, y, learning_rate, max_leaf_nodes):
    """K-1 clasificadores, el k-ésimo estima P(Y > k | x)."""
    return [clasificador(learning_rate, max_leaf_nodes).fit(X, y > k) for k in range(1, K)]


def probabilidades_m6(modelos, X):
    return np.column_stack([m.predict_proba(X)[:, 1] for m in modelos])     # (n, K-1)


def predecir_m6(modelos, X):
    return {"conteo": 1 + np.sum(probabilidades_m6(modelos, X) > 0.5, axis=1)}


BOOSTING = list(ParameterGrid({"learning_rate": [0.05, 0.1], "max_leaf_nodes": [8, 31]}))
MODELOS = {
    "M1 lineal": (ajustar_m1, predecir_m1, [{}]),
    "M2 multinomial": (ajustar_m2, predecir_m2, list(ParameterGrid({"C": [0.01, 0.1, 1, 10]}))),
    "M3 ordinal": (ajustar_m3, predecir_m3, [{}]),
    "M4 boosting clasif.": (ajustar_m4, predecir_m4, BOOSTING),
    "M5 boosting regresor": (ajustar_m5, predecir_m5, BOOSTING),
    "M6 boosting K-1": (ajustar_m6, predecir_m6, BOOSTING),
}


def validacion_cruzada(ajustar, predecir, parametros, X, y):
    """MAE promedio en 5 pliegues estratificados, para cada variante de predicción."""
    pliegues = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    errores = {}
    for ent, val in pliegues.split(X, y):
        modelo = ajustar(X[ent], y[ent], **parametros)
        for variante, prediccion in predecir(modelo, X[val]).items():
            errores.setdefault(variante, []).append(mean_absolute_error(y[val], prediccion))
    return {variante: np.mean(e) for variante, e in errores.items()}


def comparar_modelos(Xe, ye, Xp, yp):
    filas, finales, ajustados = [], {}, {}
    for nombre, (ajustar, predecir, rejilla) in MODELOS.items():
        cv = [validacion_cruzada(ajustar, predecir, par, Xe, ye) for par in rejilla]
        for variante in cv[0]:
            mejor = int(np.argmin([c[variante] for c in cv]))       # elegido por MAE en validación
            if (nombre, mejor) not in ajustados:
                ajustados[(nombre, mejor)] = ajustar(Xe, ye, **rejilla[mejor])
            modelo = ajustados[(nombre, mejor)]
            prediccion = predecir(modelo, Xp)[variante]              # la prueba se usa una sola vez
            exactitud, mae, qwk = metricas(yp, prediccion)
            etiqueta = f"{nombre} ({variante})"
            filas.append((etiqueta, str(rejilla[mejor]), exactitud, mae, qwk))
            finales[etiqueta] = (modelo, predecir, variante, prediccion)
    tabla = pd.DataFrame(filas, columns=["Modelo", "Hiperparámetros", "Exactitud", "MAE", "QWK"])
    print(tabla.round(4).to_string(index=False))
    print("Mejor por exactitud:", tabla.loc[tabla["Exactitud"].idxmax(), "Modelo"])
    print("Mejor por MAE:      ", tabla.loc[tabla["MAE"].idxmin(), "Modelo"])
    print("Mejor por QWK:      ", tabla.loc[tabla["QWK"].idxmax(), "Modelo"])
    return tabla, finales


def matriz_de_confusion(tabla, finales, yp):
    mejor = tabla.loc[tabla["MAE"].idxmin(), "Modelo"]
    matriz = confusion_matrix(yp, finales[mejor][3], labels=list(range(1, K + 1)))
    print(f"Matriz de confusión de {mejor} (filas: nivel real, columnas: nivel predicho)")
    print(pd.DataFrame(matriz, index=range(1, K + 1), columns=range(1, K + 1)))

    plt.imshow(matriz, cmap="Blues")
    for i in range(K):
        for j in range(K):
            color = "white" if matriz[i, j] > matriz.max() / 2 else "black"
            plt.text(j, i, matriz[i, j], ha="center", va="center", color=color)
    plt.xticks(range(K), range(1, K + 1))
    plt.yticks(range(K), range(1, K + 1))
    plt.xlabel("Nivel predicho")
    plt.ylabel("Nivel real")
    plt.title(f"Matriz de confusión: {mejor}")
    plt.savefig("confusion.png")
    plt.show()


def fraccion_no_decreciente(finales, Xp):
    """Fracción de datos de prueba donde P(Y > k) no decrece en k (en M6)."""
    P = probabilidades_m6(finales["M6 boosting K-1 (conteo)"][0], Xp)
    fraccion = np.mean(np.any(np.diff(P, axis=1) > 0, axis=1))
    print(f"M6: P(Y > k) no decrece en k en {100 * fraccion:.2f} % de los datos de prueba")


# =====================================================================
# (d) ¿Se cumplen los momios proporcionales?
# =====================================================================

def importancia_permutacion(predecir, X, y, repeticiones=10):
    """Aumento promedio del MAE al barajar cada columna."""
    rng = np.random.default_rng(SEMILLA)
    base = mean_absolute_error(y, predecir(X))
    importancia = []
    for j in range(X.shape[1]):
        aumentos = []
        for _ in range(repeticiones):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            aumentos.append(mean_absolute_error(y, predecir(Xp)) - base)
        importancia.append(np.mean(aumentos))
    return np.array(importancia)


def momios_proporcionales(Xe, ye, Xp, yp, tabla, finales):
    w, theta = finales["M3 ordinal (moda)"][0]
    # K-1 logísticas binarias (sin regularizar): P(Y > k) = sigmoide(w'x - theta_k), igual signo que w
    betas = np.array([LogisticRegression(C=np.inf, max_iter=5000).fit(Xe, ye > k).coef_[0]
                      for k in range(1, K)])
    top = np.argsort(-np.abs(w))[:4]

    comparacion = pd.DataFrame({"Variable": [ATRIBUTOS[j] for j in top], "w ordinal": w[top],
                                "exp(w)": np.exp(w[top])})
    for k in range(1, K):
        comparacion[f"beta P(Y>{k})"] = betas[k - 1, top]
    print(comparacion.round(3).to_string(index=False))

    plt.figure()
    for j in top:
        linea, = plt.plot(range(1, K), betas[:, j], "o-", label=ATRIBUTOS[j])
        plt.axhline(w[j], color=linea.get_color(), linestyle="--")
    plt.xticks(range(1, K))
    plt.xlabel("k  (logística para P(Y > k))")
    plt.ylabel("Coeficiente")
    plt.title("Coeficientes por k (puntos) y w del ordinal (línea punteada)")
    plt.legend()
    plt.grid(True)
    plt.savefig("momios.png")
    plt.show()

    boosting = tabla[tabla["Modelo"].str.startswith(("M4", "M5", "M6"))]
    mejor = boosting.loc[boosting["MAE"].idxmin(), "Modelo"]
    modelo, predecir, variante, _ = finales[mejor]
    importancia = importancia_permutacion(lambda X: predecir(modelo, X)[variante], Xp, yp)
    ranking = pd.DataFrame({"Variable": ATRIBUTOS, "|w| ordinal": np.abs(w),
                            "Importancia (aumento de MAE)": importancia})
    print(f"Importancia por permutación del mejor boosting: {mejor}")
    print(ranking.sort_values("Importancia (aumento de MAE)", ascending=False).round(4).to_string(index=False))

    plt.figure()
    plt.barh(ATRIBUTOS, importancia)
    plt.xlabel("Aumento del MAE al barajar la variable")
    plt.title(f"Importancia por permutación: {mejor}")
    plt.grid(True)
    plt.savefig("permutacion.png")
    plt.show()


if __name__ == "__main__":
    datos = cargar_datos()
    Xe, ye, Xp, yp = preparar(datos)
    verificar_ordinal(Xe, ye)
    tabla, finales = comparar_modelos(Xe, ye, Xp, yp)
    matriz_de_confusion(tabla, finales, yp)
    fraccion_no_decreciente(finales, Xp)
    momios_proporcionales(Xe, ye, Xp, yp, tabla, finales)