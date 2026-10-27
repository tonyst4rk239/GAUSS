"""
=============================================================================
PLANTILLA DE TRABAJO EN VIVO: GAUSS-JORDAN CON PIVOTEO PARCIAL
Materia: Métodos Numéricos — Equipo 8
=============================================================================
Instrucciones para el Live Coding:
  - Cada bloque a programar tiene la marca especial '~>'.
  - Lo que está a la derecha de '~>' es exactamente la línea que debes teclear.
=============================================================================
"""

import numpy as np

# Configuración visual para la consola de Spyder (6 decimales limpios)
np.set_printoptions(precision=6, suppress=True)


# =============================================================================
# BLOQUE 1: GAUSS-JORDAN CON PIVOTEO PARCIAL (ALGORITMO PRINCIPAL)
# =============================================================================
def gauss_jordan_con_pivoteo(A, b, verbose=False):
    """
    Resuelve el sistema Ax = b mediante Gauss-Jordan con Pivoteo Parcial.
    """
    # Conversión de A a float64 (doble precisión) y copia independiente en memoria
    A = np.array(A, dtype=np.float64, copy=True)
    
    # Conversión de b a float64 y forzado a vector columna (n, 1)
    b = np.array(b, dtype=np.float64, copy=True).reshape(-1, 1)
    
    # Número de ecuaciones e incógnitas
    n = len(b)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [TODO 1]: Construir la matriz aumentada [A | b]
    # Apila horizontalmente la matriz A (n x n) y el vector columna b (n x 1)
    # ~> Ab = np.hstack([A, b])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Ab = np.hstack([A, b])  # ~> Reemplaza por: np.hstack([A, b])
    
    if verbose:
        print("\n--- MATRIZ AUMENTADA INICIAL [A | b] ---")
        print(Ab)
        
    # Bucle que recorre cada columna k de la matriz (k = 0, 1, ..., n-1)
    for k in range(n):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # [TODO 2]: Búsqueda del pivote parcial en la columna k
        # Escanear verticalmente desde la diagonal hacia abajo buscando el máximo absoluto
        # Recuerda sumar '+ k' por el desfase del recorte local
        # ~> fila_max = k + np.argmax(np.abs(Ab[k:, k]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        fila_max = k + np.argmax(np.abs(Ab[k:, k]))  # ~> Reemplaza por: k + np.argmax(np.abs(Ab[k:, k]))
        
        # Validación de seguridad: si el pivote máximo es 0, la matriz es singular
        pivote_val = Ab[fila_max, k]
        if np.isclose(pivote_val, 0.0, atol=1e-14):
            raise ValueError(f"Sistema singular: pivote nulo en columna {k}.")
            
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # [TODO 3]: Intercambio atómico de filas (Swap) si fila_max != k
        # Aplica Fancy Indexing para cambiar fila k con fila fila_max
        # ~> if fila_max != k:
        # ~>     Ab[[k, fila_max]] = Ab[[fila_max, k]]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # <-- ESCRIBE AQUÍ EL SWAP:
        if fila_max != k:
            Ab[[k, fila_max]] = Ab[[fila_max, k]]
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # [TODO 4]: Normalizar la fila pivote para tener 1.0 en la diagonal
        # Divide toda la fila k entre su elemento diagonal Ab[k, k]
        # ~> Ab[k] = Ab[k] / Ab[k, k]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # <-- ESCRIBE AQUÍ LA NORMALIZACIÓN:
        Ab[k] = Ab[k] / Ab[k, k]
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # [TODO 5]: Eliminación bidireccional (hacer 0 arriba y abajo de la diagonal)
        # Recorrer todas las filas i != k y restar: Ab[i] - factor * Ab[k]
        # ~> for i in range(n):
        # ~>     if i != k:
        # ~>         factor = Ab[i, k]
        # ~>         Ab[i] = Ab[i] - factor * Ab[k]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # <-- ESCRIBE AQUÍ EL BUCLE DE ELIMINACIÓN:
        for i in range(n):
            if i != k:
                factor = Ab[i, k]
                Ab[i] = Ab[i] - factor * Ab[k]
        
        if verbose:
            print(f"\n--- Columna {k} reducida ---")
            print(Ab)
            
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [TODO 6]: Extraer la solución x del sistema
    # Como el bloque izquierdo ya es la identidad I, leemos la última columna
    # ~> x = Ab[:, -1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    x = Ab[:, -1]  # ~> Reemplaza por: Ab[:, -1]
    
    return x


# =============================================================================
# BLOQUE 2: CÁLCULO Y COMPROBACIÓN DEL RESIDUO
# =============================================================================
def calcular_residuo(A, b, x_calculado):
    """
    Calcula el vector residuo r = b - A x_hat y su norma euclidiana ||r||_2.
    """
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64).flatten()
    x = np.array(x_calculado, dtype=np.float64).flatten()
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [TODO 7]: Calcular el residuo algebraico r = b - A @ x y su norma L2
    # Multiplica la matriz A por la solución x con el operador '@'
    # ~> r = b - A @ x
    # ~> norma = np.linalg.norm(r, 2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    r = b - A @ x     # ~> Reemplaza por: b - A @ x
    norma = np.linalg.norm(r,2)  # ~> Reemplaza por: np.linalg.norm(r, 2)
    
    return r, norma


# =============================================================================
# BLOQUE 3: CASOS DE PRUEBA EN VIVO (EJECUTAR CON F5 EN SPYDER)
# =============================================================================
if __name__ == "__main__":
    print("=== PROBANDO NUESTRO ALGORITMO ===")
    
    # -------------------------------------------------------------------------
    # CASO 1: Sistema regular bien condicionado 3x3
    # -------------------------------------------------------------------------
    A1 = np.array([
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0]
    ])
    b1 = np.array([8.0, -11.0, -3.0])
    
    print("\n[Caso 1] Sistema regular (3x3):")
    # ~> Descomenta estas 4 líneas para probar el Caso 1 en vivo:
    
    # print(f"Norma residuo ||r||_2 : {norma1:.3e}")
    x1 = gauss_jordan_con_pivoteo(A1, b1)
    print("Solución Gauss-Jordan :", x1)
    print("Solución NumPy solve  :", np.linalg.solve(A1, b1))
    r1, norma1 = calcular_residuo(A1, b1, x1)
    # -------------------------------------------------------------------------
    # CASO 2: Sistema con pivote 0 en la diagonal (Swap obligatorio)
    # -------------------------------------------------------------------------
    A2 = np.array([
        [0.0, 2.0, 1.0],
        [1.0, -1.0, 1.0],
        [2.0, 1.0, -1.0]
    ])
    b2 = np.array([5.0, 1.0, 4.0])
    
    print("\n[Caso 2] Sistema con pivote cero en A[0,0] (Swap obligatorio):")
    # ~> Descomenta estas 3 líneas para proyectar el intercambio en vivo:
    x2 = gauss_jordan_con_pivoteo(A2, b2, verbose=True)
    print("Solución calculada    :", x2)
    r2, norma2 = calcular_residuo(A2, b2, x2)
    print(f"Norma residuo ||r||_2 : {norma2:.3e}")
