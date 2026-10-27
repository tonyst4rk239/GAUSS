# ANATOMÍA INTERNA DE CADA FUNCIÓN UTILIZADA EN EL PROYECTO
## Métodos Numéricos — Equipo 8
> **Guía de Defensa Técnica Avanzada:** Qué hace cada función, cómo está programada por dentro (Python, NumPy C-API, BLAS y LAPACK), su complejidad algorítmica y qué responder si el profesor pregunta detalles de bajo nivel.

---

## 📑 ÍNDICE DE FUNCIONES ANALIZADAS

1. [Funciones Propias del Equipo 8](#1-funciones-propias-del-equipo-8)
   - `gauss_jordan_con_pivoteo(A, b)`
   - `gauss_jordan_sin_pivoteo(A, b)`
   - `inversa_gauss_jordan(A)`
   - `calcular_residuo(A, b, x)`
   - `analizar_solucion(A, b, x_calc, x_real)`
2. [Funciones de Estructura y Memoria de NumPy](#2-funciones-de-estructura-y-memoria-de-numpy)
   - `np.array(..., dtype=np.float64, copy=True)`
   - `.reshape(-1, 1)`
   - `np.hstack([A, b])`
   - `Ab[[k, fila_max]] = Ab[[fila_max, k]]` (Fancy Indexing)
   - `Ab[:, -1]` (Slicing básico)
   - `.flatten()`
3. [Operaciones Matemáticas y Vectorizadas (UFuncs y BLAS)](#3-operaciones-matemáticas-y-vectorizadas-ufuncs-y-blas)
   - `np.abs(...)`
   - `np.argmax(...)`
   - `np.isclose(a, b, atol=...)`
   - Operador de resta y multiplicación: `Ab[i] - factor * Ab[k]` (DAXPY)
   - Operador matricial `@` (`np.matmul` / DGEMV)
   - `np.linalg.norm(r, 2)` (DNRM2)
4. [Funciones de Álgebra Lineal Avanzada (LAPACK)](#4-funciones-de-álgebra-lineal-avanzada-lapack)
   - `np.linalg.solve(A, b)` (DGESV / DGETRF + DGETRS)
   - `np.linalg.inv(A)` (DGETRF + DGETRI)
   - `np.linalg.cond(A)` (DGESDD / SVD)
   - `np.eye(n)` y `np.column_stack(...)`

---

# 1. FUNCIONES PROPIAS DEL EQUIPO 8

---

### `gauss_jordan_con_pivoteo(A, b, verbose=False)`
* **Propósito:** Resuelve el sistema lineal general $Ax = b$ convirtiendo $[A | b] \to [I | x]$ con pivoteo parcial.
* **Cómo está programada por dentro:**
  1. **Inicialización de memoria:** Convierte $A$ y $b$ a buffers continuos en C de tipo `float64` (8 bytes por número) y los apila con `hstack` para crear un bloque contiguo en memoria de $n \times (n+1)$.
  2. **Bucle de columnas ($k = 0 \dots n-1$):**
     - *Escaneo vertical:* Extrae el slice de memoria `Ab[k:, k]` sin copiar datos (vista con *stride* de fila). Aplica `abs` y busca el índice máximo con `argmax`.
     - *Validación:* Si el pivote máximo es menor a $10^{-14}$, detiene la ejecución para evitar propagación de singularidad.
     - *Swap de punteros:* Intercambia las filas $k$ y `fila_max` en la tabla de punteros del array de NumPy.
     - *Escalamiento (Normalización):* Divide la fila $k$ completa por su pivote diagonal.
     - *Eliminación bidireccional:* Recorre las filas $i \neq k$. Por cada fila, calcula el factor $m_{ik}$ y ejecuta una resta vectorizada en paralelo (operación tipo AXPY de nivel BLAS-1).
  3. **Extracción:** Devuelve la última columna del bloque de memoria (`Ab[:, -1]`).
* **Complejidad:** 
  - Tiempo: $\mathcal{O}\left(\frac{2}{3}n^3 + n^2\right)$ operaciones de punto flotante (flops).
  - Espacio: $\mathcal{O}(n^2)$ memoria adicional contigua.
* **Si el profesor pregunta:** *"¿Por qué Gauss-Jordan y no LU?"*  
  👉 **Respuesta:** *"Gauss-Jordan produce directamente la matriz identidad sin fase de sustitución hacia atrás; LU es preferible si tenemos múltiples vectores $b$, pero para resolver un sistema único o calcular inversas explícitas, Gauss-Jordan simplifica el flujo algorítmico a un solo bucle bidireccional."*

---

### `gauss_jordan_sin_pivoteo(A, b)`
* **Propósito:** Versión académica frágil que no busca máximos y asume que el elemento diagonal $A[k,k]$ siempre es válido.
* **Cómo está programada por dentro:**
  - Omite la llamada a `argmax` y el intercambio de filas.
  - Comprueba con `np.isclose(Ab[col, col], 0.0, atol=1e-15)`. Si es cero o casi cero, lanza `ZeroDivisionError`.
* **Utilidad en la clase:** Demuestra empíricamente la necesidad del estándar IEEE-754 y cómo un algoritmo sin pivoteo colapsa ante pivotes diminutos $\epsilon = 10^{-15}$.

---

### `inversa_gauss_jordan(A)`
* **Propósito:** Calcula la matriz inversa $A^{-1}$ transformando $[A | I_n] \to [I_n | A^{-1}]$.
* **Cómo está programada por dentro:**
  - Construye una super-matriz aumentada de tamaño $n \times 2n$ mediante `np.hstack([A, np.eye(n)])`.
  - Aplica exactamente la misma lógica de Gauss-Jordan con pivoteo parcial, pero operando simultáneamente sobre las $2n$ columnas.
  - Al finalizar, el bloque izquierdo queda en $I_n$ y el bloque derecho contiene $A^{-1}$, el cual se extrae con el slice `AI[:, n:]`.
* **Complejidad:** $\sim 2n^3$ flops (el triple de resolver un sistema $Ax=b$ directo).

---

### `calcular_residuo(A, b, x_calculado)`
* **Propósito:** Mide qué tan fielmente la solución calculada satisface algebraicamente el sistema original.
* **Fórmula interna:**
  $$r = b - A\hat{x}, \quad \|r\|_2 = \sqrt{\sum_{i=1}^n r_i^2}$$
* **Cómo está programada:**
  - Aplana $b$ y $x$ a vectores 1D con `.flatten()`.
  - Ejecuta la multiplicación matricial $A\hat{x}$ con el operador `@` (`DGEMV`).
  - Resta el vector resultante de $b$.
  - Calcula la norma euclidiana con `np.linalg.norm(r, 2)`.

---

# 2. FUNCIONES DE ESTRUCTURA Y MEMORIA DE NUMPY

---

### `np.array(A, dtype=np.float64, copy=True)`
* **Qué es:** Constructor principal del objeto central de NumPy: el `ndarray` (*n-dimensional array*).
* **Cómo funciona internamente en C:**
  - Reserva un bloque de memoria contigua en el *heap* de C (`malloc`).
  - Cada número flotante de 64 bits (`float64`) ocupa exactamente **8 bytes** contiguos en memoria, siguiendo la especificación IEEE-754.
  - `copy=True`: Fuerza una copia profunda de los datos binarios. Esto garantiza que si dentro del algoritmo modificamos o intercambiamos filas de $A$, la matriz original que el usuario pasó por parámetro no sufra modificaciones no deseadas (*efectos colaterales*).
* **Si el profesor pregunta:** *"¿Qué pasa si no le pongo dtype=np.float64?"*  
  👉 **Respuesta:** *"Si el usuario pasa enteros como `[[2, 1], [3, 4]]`, NumPy creará un arreglo de `int32`. Al dividir en la normalización, Python truncaría los decimales (`1 / 2 = 0`), destruyendo por completo el cálculo numérico."*

---

### `.reshape(-1, 1)`
* **Qué es:** Modificador de la geometría dimensional (*shape*) del arreglo sin alterar los datos subyacentes.
* **Cómo funciona internamente:**
  - En NumPy, los arreglos tienen una estructura llamada *Strides* (pasos de memoria).
  - Un vector unidimensional de $n$ elementos tiene `shape = (n,)` (no tiene noción de columnas).
  - `.reshape(-1, 1)` no mueve ni copia ningún número en la memoria RAM: únicamente actualiza la metadata interna del objeto en C (`PyArrayObject`), diciéndole a NumPy: *"interpréteme este bloque como una matriz de $n$ filas y 1 columna"*.
  - El valor `-1` le indica a NumPy que calcule automáticamente la dimensión restante: $\text{filas} = \frac{\text{total elementos}}{1} = n$.

---

### `np.hstack([A, b])`
* **Qué es:** Concatenación horizontal de bloques matriciales (*horizontal stack*).
* **Cómo funciona internamente:**
  - Invoca la función de bajo nivel `np.concatenate(..., axis=1)`.
  - Verifica que ambas estructuras compartan el mismo número de filas ($n$).
  - Asigna un nuevo bloque continuo de memoria en RAM de tamaño $n \times (n+1) \times 8 \text{ bytes}$.
  - Realiza una copia de memoria a nivel de sistema (`memcpy` en C) copiando las filas de $A$ y pegando a su derecha el elemento de $b$.

---

### `Ab[[k, fila_max]] = Ab[[fila_max, k]]` (Fancy Indexing)
* **Qué es:** Indexación avanzada de arreglos por listas de enteros para intercambio atómico de filas.
* **Cómo funciona internamente en C:**
  - A diferencia del *slicing* tradicional (que crea vistas), el *fancy indexing* en el lado derecho `Ab[[fila_max, k]]` extrae un buffer temporal en C con los datos de las filas en orden inverso.
  - En el lado izquierdo, la rutina interna de NumPy `PyArray_CopyObject` copia ese buffer en las posiciones de memoria correspondientes a las filas `k` y `fila_max`.
  - Todo ocurre a velocidad de lenguaje compilado C, sin necesidad de bucles `for` en Python ni variables auxiliares temporales (`aux = fila1; fila1 = fila2; fila2 = aux`).

---

### `Ab[:, -1]` (Slicing Básico)
* **Qué es:** Extracción de la última columna de una matriz bidimensional.
* **Cómo funciona internamente:**
  - El primer argumento `:` indica *"todas las filas desde la 0 hasta la n-1"*.
  - El segundo argumento `-1` es la indexación negativa en Python, que apunta directamente a la última columna de la matriz.
  - Al ser un slicing básico, no copia datos: genera una vista que apunta al primer elemento de la última columna y se desplaza con un *stride* de fila.

---

### `.flatten()`
* **Qué es:** Aplanamiento de un arreglo multidimensional a un vector 1D continuo.
* **Cómo funciona internamente:**
  - A diferencia de `.ravel()` (que intenta devolver una vista), `.flatten()` siempre reserva un nuevo bloque contiguo de memoria y copia los datos en orden *row-major* (orden C estándar: fila por fila).
  - Lo usamos en `analizar_solucion` para garantizar que la resta de vectores $r = b - A\hat{x}$ no sufra errores de dimensiones dispares como `(n, 1)` contra `(n,)`.

---

# 3. OPERACIONES MATEMÁTICAS Y VECTORIZADAS (UFUNCS Y BLAS)

---

### `np.abs(Ab[k:, k])`
* **Qué es:** Función universal (*ufunc*) que calcula el valor absoluto elemento por elemento.
* **Cómo funciona internamente:**
  - Se ejecuta a nivel de bucles C vectorizados utilizando instrucciones **SIMD** (*Single Instruction, Multiple Data*) del procesador (AVX-2 o AVX-512).
  - En lugar de evaluar elemento por elemento con un `if x < 0: -x`, el procesador borra directamente el bit de signo (el bit 63 en IEEE-754 de 64 bits) mediante una máscara lógica binaria `AND`, calculando el valor absoluto en un solo ciclo de reloj de CPU.

---

### `np.argmax(arreglo)`
* **Qué es:** Reducción que encuentra el índice del valor máximo en un arreglo.
* **Cómo funciona internamente:**
  - Ejecuta una búsqueda lineal en C de orden $\mathcal{O}(m)$, donde $m = n - k$ es la longitud de la subcolumna.
  - Recorre el buffer de memoria guardando en un registro de CPU el valor máximo encontrado y su índice local.
  - Si existen empates (dos valores máximos idénticos), devuelve siempre el índice de la primera aparición.
* **Detalle técnico de nuestra implementación:**
  - Dado que `Ab[k:, k]` empieza en la fila $k$, el índice que devuelve `np.argmax` es relativo (un número entre $0$ y $n - k - 1$).
  - Por eso en nuestro código sumamos obligatoriamente `k + np.argmax(...)` para obtener la fila real dentro de la matriz global.

---

### `np.isclose(a, b, atol=1e-14, rtol=1e-05)`
* **Qué es:** Comparación de igualdad flotante con tolerancia numérica.
* **Cómo funciona internamente:**
  - En punto flotante IEEE-754 **nunca se debe comparar con igualdad exacta (`a == 0.0`)**, debido a los errores residuales de redondeo.
  - `np.isclose` evalúa la desigualdad:
    $$|a - b| \le \text{atol} + \text{rtol} \cdot |b|$$
  - En nuestro código usamos `atol=1e-14` para verificar si un pivote es numéricamente indistinguible del cero absoluto.

---

### `Ab[i] = Ab[i] - factor * Ab[k]` (Operación DAXPY)
* **Qué es:** Combinación lineal elemental de filas.
* **Cómo funciona internamente:**
  - Esta operación elemental corresponde exactamente a la rutina estándar **DAXPY** (*Double precision A times X Plus Y*) de las librerías **BLAS Nivel 1** (*Basic Linear Algebra Subprograms*).
  - Toma el vector fila $k$ cargado en el caché L1 de la CPU, multiplica cada uno de sus elementos por el escalar flotante `factor`, y lo resta del vector fila $i$ en registros vectoriales en paralelo.

---

### Operador Matricial `@` (`np.matmul`)
* **Qué es:** Operador oficial de multiplicación matricial introducido en Python 3.5 (PEP 465).
* **Cómo funciona internamente:**
  - Para matrices y vectores densos en `float64`, NumPy no utiliza bucles `for` de Python: delega el cálculo directamente en la subrutina **DGEMV** (*Double precision General Matrix-Vector product*) de librerías BLAS de alto rendimiento (como Intel MKL o OpenBLAS).
  - Divide la matriz en bloques (*cache tiling*) para maximizar los aciertos en la memoria caché del procesador y evitar cuellos de botella en la memoria RAM.

---

### `np.linalg.norm(r, 2)`
* **Qué es:** Norma euclidiana (norma $L_2$) de un vector.
* **Cómo funciona internamente:**
  - Llama a la rutina **DNRM2** de BLAS.
  - En lugar de hacer ingenuamente `sqrt(sum(r**2))` (lo cual podría causar desbordamiento por *overflow* si los números son muy grandes, o *underflow* si son muy pequeños), utiliza el algoritmo de escalamiento interno de Blue o la función `hypot` para calcular:
    $$\|r\|_2 = s \cdot \sqrt{\sum \left(\frac{r_i}{s}\right)^2}, \quad \text{donde } s = \max |r_i|$$
  - Esto garantiza que la norma se calcule con total precisión flotante sin desbordar el rango numérico.

---

# 4. FUNCIONES DE ÁLGEBRA LINEAL AVANZADA (LAPACK)

---

### `np.linalg.solve(A, b)`
* **Qué es:** El solver profesional oficial de NumPy para resolver sistemas lineales $Ax = b$.
* **Cómo funciona internamente (LAPACK):**
  - NumPy compila y enlaza por debajo la biblioteca en Fortran **LAPACK** (*Linear Algebra Package*).
  - En matrices de punto flotante de 64 bits, invoca la rutina experta **`_gesv`** (específicamente `DGESV`).
  - `DGESV` realiza dos pasos canónicos:
    1. **`DGETRF`:** Factorización LU con pivoteo parcial ($P A = L U$).
    2. **`DGETRS`:** Solución del sistema mediante dos fases triangulares:
       - Sustitución hacia adelante sobre $L y = P b$.
       - Sustitución hacia atrás sobre $U x = y$.
* **Comparativa con nuestro código:**
  - Nuestro algoritmo de Gauss-Jordan con pivoteo parcial llega a la **misma solución exacta con la misma precisión de máquina ($2.22 \times 10^{-16}$)**, pero resolviendo directamente sin almacenar las matrices $L$ y $U$ por separado.

---

### `np.linalg.inv(A)`
* **Qué es:** Inversión matricial explícita.
* **Cómo funciona internamente:**
  - Llama a la combinación de LAPACK **`DGETRF`** (factorización LU) seguida de **`DGETRI`** (cálculo de la inversa a partir de los factores $L$ y $U$).
  - Resuelve el sistema contra las $n$ columnas de la matriz identidad.
* **Por qué demostramos en clase que no debe usarse para resolver $Ax = b$:**
  - Requiere $2n^3$ flops en lugar de los $\frac{2}{3}n^3$ de un solver directo.
  - Duplica el error de redondeo acumulado.
  - Destruye la dispersión (*sparsity*): si $A$ tiene 99% de ceros, $A^{-1}$ es 100% densa.

---

### `np.linalg.cond(A)`
* **Qué es:** Cálculo del Número de Condición $\kappa(A) = \|A\| \cdot \|A^{-1}\|$.
* **Cómo funciona internamente:**
  - Por defecto, calcula la norma 2 inducida:
    $$\kappa_2(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$
  - Para hallar los valores singulares $\sigma$, NumPy ejecuta la rutina **`DGESDD`** de LAPACK (Descomposición en Valores Singulares mediante el algoritmo *Divide and Conquer*).
  - Es el termómetro que mide la sensibilidad del sistema: si $\kappa(A) \approx 1$, el sistema es perfectamente estable; si $\kappa(A) \ge 10^{13}$ (como en Hilbert 10x10), el sistema es hipersensible y pierde 13 dígitos de precisión.

---

### `np.eye(n)` y `np.column_stack(...)`
* **`np.eye(n)`**:
  - Genera una matriz de ceros de tamaño $n \times n$ en C y asigna un valor de `1.0` en cada posición diagonal donde la fila es igual a la columna (`i == j`). La usamos en la inversión para generar $[A | I_n]$.
* **`np.column_stack([col1, col2, col3])`**:
  - Toma una secuencia de arreglos 1D y los apila como columnas contiguas para construir matrices 2D.
  - En nuestra demostración de Machine Learning, la usamos para construir la **Matriz de Diseño $X$**:
    $$X = \begin{bmatrix} 1 & x_1 & x_1^2 \\ 1 & x_2 & x_2^2 \\ \vdots & \vdots & \vdots \\ 1 & x_m & x_m^2 \end{bmatrix}$$
    permitiendo formular el sistema de Ecuaciones Normales $(X^T X)\beta = X^T y$.

---

## 🎯 RESUMEN DE RESPUESTAS RÁPIDAS PARA EL EXAMEN ORAL

| Si el profesor te pregunta... | Tu respuesta técnica exacta: |
|---|---|
| *"¿Qué algoritmo de bajo nivel corre `np.linalg.solve`?"* | *"Ejecuta la rutina `DGESV` de LAPACK escrita en Fortran, que realiza descomposición $PA = LU$ con pivoteo parcial mediante `DGETRF` y luego sustitución triangular doble con `DGETRS`."* |
| *"¿Por qué usaron `@` en vez de `np.dot`?"* | *"El operador `@` (`__matmul__`) es el estándar moderno introducido en Python 3.5. Invoca directamente la rutina BLAS `DGEMV`, aprovechando optimizaciones de memoria caché L1/L2 del procesador."* |
| *"¿Por qué sumas `+ k` en el `argmax`?"* | *"Porque `Ab[k:, k]` es un slice que genera una vista local con índices que inician en cero. Sumar $k$ traslada ese índice local a la posición absoluta de la fila en la matriz original."* |
| *"¿Cómo calcula `np.linalg.norm` la norma euclidiana sin desbordar?"* | *"Llama a `DNRM2` de BLAS, que escala los números antes de elevarlos al cuadrado para evitar *overflow* en números gigantescos o *underflow* en números diminutos."* |
| *"¿Qué complejidad asintótica tiene su Gauss-Jordan?"* | *"Tiene una complejidad temporal de $\mathcal{O}(n^3)$ operaciones elementales de punto flotante y una complejidad espacial de $\mathcal{O}(n^2)$ para la matriz aumentada contigua en memoria."* |
