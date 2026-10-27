# 🧮 Gauss-Jordan con Pivoteo Parcial
> **Métodos Numéricos — Equipo 8**  
> Plantilla de trabajo interactiva para la clase en vivo. Conversión de operaciones elementales por fila en un algoritmo computacional robusto para resolver sistemas lineales $Ax = b$.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tonyst4rk239/GAUSS/blob/main/gauss_jordan_student.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Required-orange.svg)](https://numpy.org/)
[![IDE](https://img.shields.io/badge/IDE-Spyder%20%7C%20Colab-green.svg)](https://www.spyder-ide.org/)

---

## 📌 ¿Cómo seguir la clase?

Tienes dos formas sencillas de trabajar con nosotros en tiempo real:

### Opción A: En la nube con Google Colab (Recomendado para celulares y laptops)
Haz clic en el siguiente botón para abrir el cuaderno interactivo directamente en tu navegador sin instalar nada:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tonyst4rk239/GAUSS/blob/main/gauss_jordan_student.ipynb)

### Opción B: En tu computadora con Spyder
1. Descarga o clona este repositorio:
   ```bash
   git clone https://github.com/tonyst4rk239/GAUSS.git
   ```
2. Abre **Spyder**.
3. En la barra superior: `Archivo` ➔ `Abrir...` y selecciona **`gauss_jordan_student.py`** (o importa el notebook **`gauss_jordan_student.ipynb`**).
4. Iremos completando juntos los bloques marcados con `TODO 1` a `TODO 7`.
5. Presiona **F5** para ejecutar y verificar los resultados en la consola de IPython.

---

## 🚀 Hoja de Ruta de los 7 TODOs

Durante la sesión implementaremos:

1. **TODO 1:** Construcción de la matriz aumentada $[A | b]$ con `np.hstack`.
2. **TODO 2:** Búsqueda del pivote parcial en la columna $k$ con `np.argmax(np.abs(...))`.
3. **TODO 3:** Intercambio atómico de filas con Fancy Indexing de NumPy.
4. **TODO 4:** Normalización unitaria de la fila pivote ($R_k \leftarrow \frac{1}{a_{kk}} R_k$).
5. **TODO 5:** Eliminación bidireccional hacia adelante y hacia atrás ($R_i \leftarrow R_i - m_{ik} R_k$).
6. **TODO 6:** Extracción de la solución del sistema desde la última columna.
7. **TODO 7:** Validación del residuo $r = b - A\hat{x}$ y su norma euclidiana $\|r\|_2$.

---

## 👥 Equipo 8
- Métodos Numéricos
- Repositorio oficial del taller interactivo
