"""
=============================================================================
EQUIPO 8 — MÉTODOS NUMÉRICOS: GAUSS-JORDAN CON PIVOTEO PARCIAL (MAESTRO)
=============================================================================
Este archivo contiene la suite de producción y experimentos analíticos.
Cada línea de código está comentada en profundidad para explicar su fundamento
algebraico, comportamiento en memoria y funcionamiento en Python / NumPy.
=============================================================================
"""

# Importa NumPy para manipulación matricial de alta eficiencia y cálculo numérico
import numpy as np

# Importa el módulo pyplot de Matplotlib para generar gráficos científicos y de Machine Learning
import matplotlib.pyplot as plt

# Importa el módulo 'time' para medir tiempos de ejecución en CPU (benchmarking de milisegundos)
import time

# Configura las opciones de impresión de arreglos NumPy en la consola
# 'precision=6': Fija en 6 los decimales mostrados
# 'suppress=True': Muestra números sumamente pequeños (menores al épsilon de formato) como 0.0
np.set_printoptions(precision=6, suppress=True)


# =============================================================================
# 1. IMPLEMENTACIÓN DE GAUSS-JORDAN SIN PIVOTEO (VERSIÓN FRÁGIL / EDUCATIVA)
# =============================================================================
# Función que implementa Gauss-Jordan directo sobre la diagonal sin verificar magnitudes
def gauss_jordan_sin_pivoteo(A, b):
    """
    Resuelve el sistema Ax = b usando eliminación de Gauss-Jordan directa.
    ADVERTENCIA: Si un elemento pivote en la diagonal es 0 o cercano a 0,
    el algoritmo falla por división entre cero o amplificación de error de redondeo.
    """
    # Convierte A a una matriz de tipo float64 de doble precisión en una copia nueva en RAM
    A = np.array(A, dtype=np.float64, copy=True)
    
    # Convierte b a float64 y le da forma de vector columna bidimensional de (n, 1)
    b = np.array(b, dtype=np.float64, copy=True).reshape(-1, 1)
    
    # Obtiene la dimensión n del sistema (número de ecuaciones)
    n = len(b)
    
    # Construye la matriz aumentada [A | b] concatenando horizontalmente A y b
    Ab = np.hstack([A, b])
    
    # Itera sobre cada columna del sistema desde col = 0 hasta n - 1
    for col in range(n):
        # Toma directamente como pivote el elemento que se encuentra en la diagonal principal
        pivote = Ab[col, col]
        
        # 'np.isclose(pivote, 0.0, atol=1e-15)': Verifica si el pivote es nulo con tolerancia absoluta de 10^-15
        if np.isclose(pivote, 0.0, atol=1e-15):
            # Lanza una excepción ZeroDivisionError informando que el método sin pivoteo colapsó
            raise ZeroDivisionError(f"Pivote nulo o casi nulo detectado en fila {col}. "
                                    f"Gauss-Jordan sin pivoteo no puede continuar.")
        
        # Normaliza la fila actual dividiendo todos sus coeficientes entre el pivote diagonal
        Ab[col] = Ab[col] / pivote
        
        # Itera sobre todas las filas para hacer ceros arriba y abajo del pivote
        for fila in range(n):
            # Evita procesar la fila pivote actual
            if fila != col:
                # Almacena el coeficiente a eliminar en la columna actual
                factor = Ab[fila, col]
                # Resta a la fila el múltiplo de la fila pivote normalizada: R_fila <- R_fila - factor * R_col
                Ab[fila] -= factor * Ab[col]
                
    # Extrae y retorna la última columna de la matriz aumentada, donde quedó la solución x
    x = Ab[:, -1]
    return x


# =============================================================================
# 2. IMPLEMENTACIÓN ROBUSTA: GAUSS-JORDAN CON PIVOTEO PARCIAL
# =============================================================================
# Función robusta con selección del máximo absoluto en cada columna
def gauss_jordan_con_pivoteo(A, b, verbose=False):
    """
    Resuelve el sistema Ax = b mediante Gauss-Jordan con Pivoteo Parcial.
    Estrategia de Pivoteo Parcial:
    En cada columna k, busca el elemento con mayor valor absoluto en las filas i >= k.
    Si está en una fila distinta a k, intercambia las filas.
    Esto minimiza el error de redondeo y evita divisiones por cero.
    """
    # Convierte A a float64 y genera copia en memoria para no modificar datos originales
    A = np.array(A, dtype=np.float64, copy=True)
    
    # Convierte b a float64 y le da formato de columna (n, 1) mediante reshape
    b = np.array(b, dtype=np.float64, copy=True).reshape(-1, 1)
    
    # Calcula el número de incógnitas del sistema
    n = len(b)
    
    # Concatena A y b horizontalmente para crear la matriz aumentada [A | b]
    Ab = np.hstack([A, b])
    
    # Si verbose es True, muestra la matriz antes de procesarla
    if verbose:
        print("\n--- MATRIZ AUMENTADA INICIAL [A | b] ---")
        print(Ab)
        
    # Bucle que recorre cada una de las columnas k de la matriz
    for k in range(n):
        # 1. BÚSQUEDA DEL PIVOTE PARCIAL:
        # 'Ab[k:, k]': Toma la subcolumna k desde la fila k hasta la última fila n-1
        # 'np.abs(...)': Convierte los elementos a valor absoluto para medir su magnitud
        # 'np.argmax(...)': Encuentra el índice relativo del valor máximo
        # 'k + ...': Desplaza el índice relativo sumándole k para obtener la fila absoluta en Ab
        fila_max = k + np.argmax(np.abs(Ab[k:, k]))
        
        # Extrae el valor numérico del pivote máximo localizado
        pivote_val = Ab[fila_max, k]
        
        # Comprueba si el pivote máximo es prácticamente 0 (dentro de tolerancia 10^-14)
        if np.isclose(pivote_val, 0.0, atol=1e-14):
            # Si el valor más grande es cero, la matriz es singular (no invertible)
            raise ValueError(f"El sistema es singular o numéricamente no invertible. "
                             f"Pivote máximo en columna {k} es ~0.")
        
        # 2. INTERCAMBIO DE FILAS (SWAP) si el pivote no está en la fila actual k
        if fila_max != k:
            # Utiliza Fancy Indexing para intercambiar la fila k con la fila fila_max en un paso atómico
            Ab[[k, fila_max]] = Ab[[fila_max, k]]
            # Si verbose está activado, imprime el detalle del intercambio en consola
            if verbose:
                print(f"\n[PIVOTEO] Intercambiando Fila {k} con Fila {fila_max} (Pivote = {pivote_val:.4f})")
                print(Ab)
                
        # 3. NORMALIZACIÓN DE LA FILA PIVOTE (genera un 1.0 en la diagonal principal)
        # Divide toda la fila k entre su elemento diagonal Ab[k, k]
        Ab[k] = Ab[k] / Ab[k, k]
        
        # 4. ELIMINACIÓN HACIA ADELANTE Y HACIA ATRÁS (genera ceros en toda la columna k)
        for i in range(n):
            # Salta la fila pivote k
            if i != k:
                # Almacena el factor multiplicador correspondiente
                factor = Ab[i, k]
                # Resta a la fila i el factor multiplicado por la fila pivote k
                Ab[i] = Ab[i] - factor * Ab[k]
                
        # Si verbose está activado, muestra la matriz con la columna k completamente reducida
        if verbose:
            print(f"\n--- Columna {k} reducida ---")
            print(Ab)
            
    # Extrae la última columna de la matriz reducida, correspondiente a la solución del sistema
    x = Ab[:, -1]
    
    # Retorna el vector solución
    return x


# =============================================================================
# 3. CÁLCULO EXPLÍCITO DE LA MATRIZ INVERSA MEDIANTE [A | I] -> [I | A^-1]
# =============================================================================
# Función que calcula A^-1 resolviendo simultáneamente las n columnas canónicas
def inversa_gauss_jordan(A):
    """
    Calcula A^-1 transformando [A | I] en [I | A^-1] usando Gauss-Jordan.
    """
    # Convierte A a tipo float64 y crea una copia independiente
    A = np.array(A, dtype=np.float64, copy=True)
    
    # Obtiene el número de filas de A
    n = A.shape[0]
    
    # 'assert': Valida que la matriz sea estrictamente cuadrada (filas == columnas)
    assert A.shape[0] == A.shape[1], "La matriz debe ser cuadrada."
    
    # 'np.eye(n)': Genera la matriz identidad de tamaño n x n
    # 'np.hstack([A, np.eye(n)])': Concatena A con I creando la super-matriz aumentada [A | I] de (n x 2n)
    AI = np.hstack([A, np.eye(n)])
    
    # Itera sobre cada columna k para reducir el bloque izquierdo a la identidad
    for k in range(n):
        # Búsqueda del pivote parcial en la columna k del bloque izquierdo
        fila_max = k + np.argmax(np.abs(AI[k:, k]))
        
        # Validación de no-singularidad
        if np.isclose(AI[fila_max, k], 0.0, atol=1e-14):
            raise ValueError("La matriz es singular, no tiene inversa.")
            
        # Si el pivote no está en la fila k, ejecuta el intercambio en toda la super-fila
        if fila_max != k:
            AI[[k, fila_max]] = AI[[fila_max, k]]
            
        # Normaliza la fila pivote completa de tamaño 2n dividiendo entre el pivote diagonal
        AI[k] = AI[k] / AI[k, k]
        
        # Elimina los elementos superiores e inferiores en las 2n columnas
        for i in range(n):
            if i != k:
                # Resta la fila pivote multiplicada por el factor
                AI[i] -= AI[i, k] * AI[k]
                
    # 'AI[:, n:]': Toma todas las filas y extrae las columnas desde la posición n hasta el final (2n)
    # En ese bloque derecho quedó calculada exactamente la matriz inversa A^-1
    A_inv = AI[:, n:]
    
    # Retorna la matriz inversa
    return A_inv


# =============================================================================
# 4. MÉTRICAS NUMÉRICAS: RESIDUO vs ERROR HACIA ADELANTE
# =============================================================================
# Función que evalúa la estabilidad numérica calculando la norma del residuo y error relativo
def analizar_solucion(A, b, x_calculado, x_real=None, nombre_metodo="Método"):
    """
    Calcula el vector de residuo r = b - A x_hat y su norma euclidiana ||r||_2.
    Si se conoce x_real, también calcula el error hacia adelante relativo.
    """
    # Convierte A a flotante de 64 bits
    A = np.array(A, dtype=np.float64)
    # Aplana b a un vector 1D
    b = np.array(b, dtype=np.float64).flatten()
    # Aplana x_calculado a un vector 1D
    x_hat = np.array(x_calculado, dtype=np.float64).flatten()
    
    # 'r = b - A @ x_hat': Calcula el vector residuo (discrepancia en el espacio de salida)
    r = b - A @ x_hat
    
    # 'np.linalg.norm(r, 2)': Calcula la norma L2 (euclidiana) del residuo: sqrt(sum(r_i^2))
    norma_residuo = np.linalg.norm(r, 2)
    
    # Imprime encabezado del método evaluado
    print(f"\n[{nombre_metodo}]")
    # Muestra el vector solución obtenido
    print(f"  -> Solución x_hat       : {x_hat}")
    # Muestra la norma euclidiana del residuo en notación científica
    print(f"  -> Norma Residuo ||r||_2: {norma_residuo:.3e}")
    
    # Evalúa si el usuario proporcionó una solución teórica exacta de referencia
    if x_real is not None:
        # Aplana la solución exacta a 1D
        x_real = np.array(x_real, dtype=np.float64).flatten()
        # Calcula el error absoluto hacia adelante: ||x_real - x_hat||_2
        error_absoluto = np.linalg.norm(x_real - x_hat, 2)
        # Calcula la norma de la solución exacta: ||x_real||_2
        norma_real = np.linalg.norm(x_real, 2)
        # Calcula el error relativo hacia adelante: ||x_real - x_hat|| / ||x_real||
        error_relativo = error_absoluto / norma_real if norma_real != 0 else error_absoluto
        # Imprime el error relativo en notación científica
        print(f"  -> Error hacia adelante relativo ||x - x_hat|| / ||x||: {error_relativo:.3e}")
        # Retorna ambas métricas
        return norma_residuo, error_relativo
        
    # Si no se pasó x_real, retorna solo el residuo
    return norma_residuo, None


# =============================================================================
# 5. CASOS DE PRUEBA OBLIGATORIOS Y EXPERIMENTOS
# =============================================================================
# Demostración 1 y 2 requeridas por el temario oficial
def demo_casos_obligatorios():
    # Imprime separador visual
    print("=" * 80)
    print("DEMOSTRACIÓN 1: CASO REGULAR (3x3)")
    print("=" * 80)
    # Matriz regular bien condicionada de 3x3
    A_reg = np.array([
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0]
    ])
    # Vector de términos independientes correspondiente
    b_reg = np.array([8.0, -11.0, -3.0])
    
    # Resuelve con nuestro algoritmo de Gauss-Jordan con pivoteo parcial
    x_pivot = gauss_jordan_con_pivoteo(A_reg, b_reg, verbose=False)
    # Resuelve con la función oficial de NumPy (referencia LAPACK)
    x_numpy = np.linalg.solve(A_reg, b_reg)
    
    # Analiza y muestra los resultados de ambos métodos
    analizar_solucion(A_reg, b_reg, x_pivot, x_real=x_numpy, nombre_metodo="Gauss-Jordan con Pivoteo")
    analizar_solucion(A_reg, b_reg, x_numpy, nombre_metodo="numpy.linalg.solve")
    
    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN 2: CASO QUE REQUIERE INTERCAMBIO DE FILAS (Pivote Cero)")
    print("=" * 80)
    # Matriz donde el primer coeficiente diagonal es cero exacto: A[0,0] = 0.0
    A_swap = np.array([
        [0.0, 2.0, 1.0],
        [1.0, -1.0, 1.0],
        [2.0, 1.0, -1.0]
    ])
    b_swap = np.array([5.0, 1.0, 4.0])
    
    print("\nIntentando con Gauss-Jordan SIN pivoteo:")
    # Bloque try-except para capturar la falla esperada del método sin pivote
    try:
        gauss_jordan_sin_pivoteo(A_swap, b_swap)
    except ZeroDivisionError as e:
        # Imprime la captura del error por división entre cero
        print(f" [FALLA ESPERADA]: {e}")
        
    print("\nResolviendo con Gauss-Jordan CON pivoteo parcial:")
    # Resuelve con pivoteo y verbose=True para ver el intercambio de filas
    x_swap = gauss_jordan_con_pivoteo(A_swap, b_swap, verbose=True)
    x_numpy_swap = np.linalg.solve(A_swap, b_swap)
    analizar_solucion(A_swap, b_swap, x_swap, x_real=x_numpy_swap, nombre_metodo="Gauss-Jordan con Pivoteo")


# Experimento obligatorio de estrés numérico con perturbación diminuta
def demo_experimento_obligatorio():
    print("\n" + "=" * 80)
    print("EXPERIMENTO OBLIGATORIO: INESTABILIDAD NUMÉRICA (Sin Pivote vs Con Pivote vs Solve)")
    print("=" * 80)
    # Define un valor epsilon diminuto (10^-15), cercano al épsilon de máquina IEEE-754
    eps = 1e-15
    # Matriz donde el coeficiente A[0,0] es epsilon
    A_inestable = np.array([
        [eps, 1.0],
        [1.0, 1.0]
    ])
    # Vector b configurado para que la solución analítica exacta sea estrictamente x = [1.0, 1.0]
    b_inestable = np.array([1.0 + eps, 2.0])
    x_exacto = np.array([1.0, 1.0])
    
    print(f"Sistema 2x2 con epsilon = {eps}:")
    print(f"  [ {eps}   1.0 ] [x1] = [ {1.0+eps} ]")
    print(f"  [ 1.0    1.0 ] [x2]   [ 2.0     ]")
    print(f"Solución real esperada: x = {x_exacto}")
    
    # 1. Prueba de ejecución sin pivoteo
    try:
        x_sin = gauss_jordan_sin_pivoteo(A_inestable, b_inestable)
        analizar_solucion(A_inestable, b_inestable, x_sin, x_real=x_exacto, nombre_metodo="1. Gauss-Jordan SIN Pivoteo")
    except Exception as e:
        print(f"Error sin pivoteo: {e}")
        
    # 2. Prueba de ejecución con pivoteo parcial
    x_con = gauss_jordan_con_pivoteo(A_inestable, b_inestable)
    analizar_solucion(A_inestable, b_inestable, x_con, x_real=x_exacto, nombre_metodo="2. Gauss-Jordan CON Pivoteo")
    
    # 3. Prueba de ejecución con numpy.linalg.solve
    x_np = np.linalg.solve(A_inestable, b_inestable)
    analizar_solucion(A_inestable, b_inestable, x_np, x_real=x_exacto, nombre_metodo="3. numpy.linalg.solve")


# Experimento avanzado sobre mal condicionamiento con Matrices de Hilbert
def demo_matriz_hilbert():
    print("\n" + "=" * 80)
    print("EXPERIMENTO PRO: MATRICES DE HILBERT Y CONDICIONAMIENTO")
    print("=" * 80)
    print("La matriz de Hilbert H_ij = 1 / (i + j + 1) es el ejemplo canónico de matriz mal condicionada.")
    print("Pregunta clave: ¿Puede haber residuo pequeño y error grande? ¡SÍ!\n")
    
    # Lista de dimensiones n a probar (4x4, 6x6, 8x8, 10x10)
    tamanos = [4, 6, 8, 10]
    condiciones = []
    errores_sin = []
    errores_con = []
    errores_np = []
    
    # Itera sobre cada tamaño
    for n in tamanos:
        # Genera la matriz de Hilbert n x n usando comprensión de listas: H[i,j] = 1 / (i + j + 1)
        H = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
        # Fija la solución teórica exacta como un vector de unos: x = [1, 1, ..., 1]
        x_real = np.ones(n)
        # Calcula el vector de términos independientes exacto: b = H * x_real
        b = H @ x_real
        
        # 'np.linalg.cond(H)': Calcula el Número de Condición kappa(H) = ||H|| * ||H^-1||
        cond_H = np.linalg.cond(H)
        condiciones.append(cond_H)
        
        print(f"\n--- Hilbert {n}x{n} | Número de Condición kappa(H) = {cond_H:.2e} ---")
        
        # Ejecuta la versión sin pivoteo
        try:
            x_sin = gauss_jordan_sin_pivoteo(H, b)
            _, err_sin = analizar_solucion(H, b, x_sin, x_real, f"Sin Pivote (n={n})")
            errores_sin.append(err_sin)
        except Exception as e:
            print(f"Sin pivoteo falló: {e}")
            errores_sin.append(np.nan)
            
        # Ejecuta la versión con pivoteo parcial
        x_con = gauss_jordan_con_pivoteo(H, b)
        r_con, err_con = analizar_solucion(H, b, x_con, x_real, f"Con Pivote (n={n})")
        errores_con.append(err_con)
        
        # Ejecuta la función oficial de NumPy
        x_np = np.linalg.solve(H, b)
        _, err_np = analizar_solucion(H, b, x_np, x_real, f"NumPy Solve (n={n})")
        errores_np.append(err_np)
        
    print("\n[CONCLUSIÓN CLAVE]: El pivoteo parcial retarda el desastre numérico, pero cuando kappa(A) > 10^16,")
    print("incluso con residuo ~ 0 en aritmética flotante IEEE-754, el error hacia adelante se amplifica por kappa(A).")


# Demostración del costo computacional de invertir frente a resolver directamente
def demo_por_que_no_invertir():
    print("\n" + "=" * 80)
    print("PREGUNTA CLAVE: ¿POR QUÉ NO RESOLVER Ax = b CALCULANDO PRIMERO A^-1?")
    print("=" * 80)
    # Tamaño de la matriz para el benchmark (800 ecuaciones con 800 incógnitas)
    n = 800
    print(f"Generando sistema denso aleatorio de tamaño {n}x{n}...")
    # Fija una semilla aleatoria para garantizar reproducibilidad exacta
    np.random.seed(42)
    # Genera matriz aleatoria densa y le suma 10 a la diagonal para asegurar diagonal dominante y no singularidad
    A = np.random.rand(n, n) + np.eye(n) * 10
    # Genera vector aleatorio b
    b = np.random.rand(n)
    
    # Opción 1: Resolver directamente el sistema lineal mediante eliminación / LU
    t0 = time.time()
    x_directo = np.linalg.solve(A, b)
    t_directo = time.time() - t0
    
    # Opción 2: Invertir explícitamente la matriz y multiplicar A^-1 por b
    t0 = time.time()
    A_inv = np.linalg.inv(A)
    x_inversa = A_inv @ b
    t_inversa = time.time() - t0
    
    # Calcula la norma de los residuos de ambos enfoques
    res_directo = np.linalg.norm(b - A @ x_directo)
    res_inversa = np.linalg.norm(b - A @ x_inversa)
    
    # Muestra los tiempos de CPU y residuos obtenidos
    print(f"1. Resolver directo (np.linalg.solve) : Tiempo = {t_directo:.4f} s | Residuo = {res_directo:.3e}")
    print(f"2. Invertir y multiplicar (inv(A) @ b) : Tiempo = {t_inversa:.4f} s | Residuo = {res_inversa:.3e}")
    print(f"-> Invertir tomó {t_inversa / t_directo:.2f}x más tiempo y triplica las operaciones elementales (~n^3 vs ~n^3/3).")


# =============================================================================
# 6. APLICACIÓN MODERNA: DATA SCIENCE & MACHINE LEARNING
#    Ecuaciones Normales: (X^T X) beta = X^T y
# =============================================================================
# Demostración del uso de Gauss-Jordan en Inteligencia Artificial y Regresión
def demo_machine_learning_regression():
    print("\n" + "=" * 80)
    print("APLICACIÓN MODERNA: REGRESIÓN EN MACHINE LEARNING (ECUACIONES NORMALES)")
    print("=" * 80)
    print("En Inteligencia Artificial y Ciencia de Datos, entrenar una regresión lineal polinomial")
    print("óptima equivale a resolver el sistema de Ecuaciones Normales:")
    print("       (X^T X) beta = X^T y")
    print("Donde beta son los parámetros (pesos) del modelo que minimizan el error cuadrático medio.")
    
    # Fija semilla aleatoria para generar datos reproducibles
    np.random.seed(101)
    # Número de muestras experimentales observadas (m = 50)
    m = 50
    # Genera 50 puntos equiespaciados entre -3 y 3 para la variable predictora x
    x_data = np.linspace(-3, 3, m)
    # Genera la variable objetivo y con ruido gaussiano: y = 1.5*x^2 - 2*x + 3 + Ruido
    y_data = 1.5 * (x_data**2) - 2.0 * x_data + 3.0 + np.random.normal(0, 1.5, size=m)
    
    # Construye la Matriz de Diseño X de tamaño (m x 3):
    # Columna 0: Término independiente (unos)
    # Columna 1: Término lineal (x)
    # Columna 2: Término cuadrático (x^2)
    X = np.column_stack([np.ones(m), x_data, x_data**2])
    
    # Formula el sistema de Ecuaciones Normales (X^T X) * beta = X^T y
    # Matriz del sistema: A_ml = X^T * X (tamaño 3x3)
    A_ml = X.T @ X
    # Vector de términos independientes: b_ml = X^T * y (tamaño 3x1)
    b_ml = X.T @ y_data
    
    # Resuelve el sistema de pesos beta utilizando NUESTRO Gauss-Jordan con pivoteo parcial
    beta_calculado = gauss_jordan_con_pivoteo(A_ml, b_ml)
    # Resuelve con np.linalg.solve para contrastar
    beta_numpy = np.linalg.solve(A_ml, b_ml)
    
    # Muestra los coeficientes beta hallados
    print(f"\nParámetros calculados con Gauss-Jordan : beta = {beta_calculado}")
    print(f"Parámetros calculados con np.linalg.solve: beta = {beta_numpy}")
    # Comprueba la coincidencia casi perfecta entre ambos cálculos
    print(f"Diferencia máxima entre ambos: {np.max(np.abs(beta_calculado - beta_numpy)):.3e}")
    
    # Genera una curva suave de 200 puntos para evaluar la función aprendida por el modelo
    x_curva = np.linspace(-3.2, 3.2, 200)
    # Construye la matriz de diseño de la curva
    X_curva = np.column_stack([np.ones(len(x_curva)), x_curva, x_curva**2])
    # Calcula las predicciones del modelo: y_pred = X_curva * beta
    y_pred = X_curva @ beta_calculado
    
    # Inicializa una figura en Matplotlib con dimensiones de 9 x 5 pulgadas
    plt.figure(figsize=(9, 5))
    # Dibuja los puntos experimentales dispersos en color carmesí
    plt.scatter(x_data, y_data, color='crimson', label='Datos Experimentales (Muestras de entrenamiento)')
    # Dibuja la curva parabólica continua ajustada por Gauss-Jordan en color azul marino
    plt.plot(x_curva, y_pred, color='navy', linewidth=2.5, 
             label=f'Ajuste ML por Gauss-Jordan: $\\beta_0={beta_calculado[0]:.2f}, \\beta_1={beta_calculado[1]:.2f}, \\beta_2={beta_calculado[2]:.2f}$')
    # Agrega título formal al gráfico
    plt.title("Aplicación en Machine Learning: Regresión Polinomial vía Ecuaciones Normales", fontsize=12, fontweight='bold')
    # Etiqueta del eje horizontal
    plt.xlabel("Variable predictora (x)")
    # Etiqueta del eje vertical
    plt.ylabel("Objetivo (y)")
    # Agrega rejilla de fondo punteada semitransparente
    plt.grid(True, linestyle='--', alpha=0.6)
    # Agrega la leyenda explicativa
    plt.legend()
    # Ajusta los márgenes automáticamente para que no se corten etiquetas
    plt.tight_layout()
    
    # Guarda la figura resultante en formato PNG si el entorno lo permite, o la muestra en pantalla
    grafica_path = "ml_regression_gauss_jordan.png"
    try:
        plt.savefig(grafica_path, dpi=200)
        print(f"[GRÁFICA GENERADA]: Se ha guardado el gráfico explicativo en '{grafica_path}'")
    except Exception as e:
        print(f"[GRÁFICA GENERADA]: Gráfico generado en memoria para visualización en Spyder (Plots)")
    # Cierra la figura en memoria para liberar recursos
    plt.close()


# =============================================================================
# BLOQUE PRINCIPAL DE EJECUCIÓN (F5 EN SPYDER)
# =============================================================================
if __name__ == "__main__":
    # Ejecuta en secuencia cada uno de los módulos de prueba y benchmarking
    demo_casos_obligatorios()
    demo_experimento_obligatorio()
    demo_matriz_hilbert()
    demo_por_que_no_invertir()
    demo_machine_learning_regression()
