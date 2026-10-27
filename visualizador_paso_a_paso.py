"""
=============================================================================
VISUALIZADOR GRÁFICO PASO A PASO: GAUSS-JORDAN CON PIVOTEO
Herramienta de alto impacto visual para proyectar en Spyder / Clase
=============================================================================
Muestra cómo evoluciona la matriz aumentada [A | b] en cada iteración,
resaltando con colores:
  - El pivote seleccionado
  - El intercambio de filas (swap)
  - La formación progresiva de la matriz identidad
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def graficar_matriz_estado(Ab, titulo, pivote_coord=None, swap_filas=None):
    """
    Dibuja un mapa de calor numérico de la matriz aumentada [A | b].
    """
    filas, cols = Ab.shape
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Mapa de calor
    cax = ax.matshow(Ab, cmap="Blues", alpha=0.75)
    fig.colorbar(cax)
    
    # Escribir los valores numéricos dentro de cada celda
    for i in range(filas):
        for j in range(cols):
            valor = Ab[i, j]
            color_texto = "black"
            peso_texto = "normal"
            
            # Resaltar pivote
            if pivote_coord and pivote_coord == (i, j):
                color_texto = "red"
                peso_texto = "bold"
            
            texto = f"{valor:.2f}" if abs(valor) > 1e-4 or valor == 0 else f"{valor:.1e}"
            ax.text(j, i, texto, ha="center", va="center", color=color_texto,
                    fontweight=peso_texto, fontsize=11)
            
    # Línea divisoria vertical entre A y b
    ax.axvline(x=cols - 1.5, color='crimson', linestyle='--', linewidth=2.5, label='Límite [A | b]')
    
    ax.set_xticks(range(cols))
    ax.set_xticklabels([f"x{i+1}" for i in range(cols-1)] + ["b"])
    ax.set_yticks(range(filas))
    ax.set_yticklabels([f"Fila {i}" for i in range(filas)])
    
    extra = ""
    if swap_filas:
        extra = f"\n[SWAP]: Fila {swap_filas[0]} <---> Fila {swap_filas[1]}"
        
    ax.set_title(f"{titulo}{extra}", fontsize=12, pad=12, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15))
    plt.tight_layout()
    plt.show()


def demo_visualizacion_didactica():
    print("Iniciando visualización gráfica interactiva paso a paso...")
    # Sistema con cero en la diagonal que requiere pivoteo
    A = np.array([
        [0.0, 2.0, 1.0],
        [1.0, -1.0, 1.0],
        [2.0, 1.0, -1.0]
    ])
    b = np.array([5.0, 1.0, 4.0]).reshape(-1, 1)
    n = len(b)
    
    Ab = np.hstack([A, b])
    graficar_matriz_estado(Ab, "Paso 0: Matriz Aumentada Inicial [A | b]")
    
    for k in range(n):
        # 1. Búsqueda de pivote
        fila_max = k + np.argmax(np.abs(Ab[k:, k]))
        swap_info = None
        
        # 2. Swap si es necesario
        if fila_max != k:
            swap_info = (k, fila_max)
            Ab[[k, fila_max]] = Ab[[fila_max, k]]
            graficar_matriz_estado(Ab, f"Paso {k+1}.1: Intercambio de Filas", 
                                  pivote_coord=(k, k), swap_filas=swap_info)
        
        # 3. Normalizar
        Ab[k] = Ab[k] / Ab[k, k]
        
        # 4. Eliminar resto de filas
        for i in range(n):
            if i != k:
                factor = Ab[i, k]
                Ab[i] = Ab[i] - factor * Ab[k]
                
        graficar_matriz_estado(Ab, f"Paso {k+1}.2: Columna {k} reducida a forma canónica", 
                              pivote_coord=(k, k))
        
    print("\n¡Visualización completa! La matriz A se transformó en la identidad I.")
    print("Solución leída en la columna 'b':", Ab[:, -1])


if __name__ == "__main__":
    demo_visualizacion_didactica()
