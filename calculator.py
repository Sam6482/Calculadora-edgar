import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy as sp
import sys

class SurfaceAnalyzer:
    """
    Clase para manejar el análisis, cálculo y visualización de funciones de dos variables.
    """
    
    def __init__(self, func_str, x_range, y_range, points=100):
        """
        Inicializa el analizador.
        
        :param func_str: Cadena con la función, ej: "sin(x)*cos(y)"
        :param x_range: Tupla (min, max) para el eje X
        :param y_range: Tupla (min, max) para el eje Y
        :param points: Número de puntos para la malla (resolución)
        """
        self.func_str = func_str
        self.x_range = x_range
        self.y_range = y_range
        self.points = points
        
        # Variables simbólicas
        self.x_sym, self.y_sym = sp.symbols('x y')
        self.func_lambda = None
        self.X, self.Y, self.Z = None, None, None

    def parse_function(self):
        """
        Convierte la cadena de texto en una función vectorizada de Numpy.
        Maneja errores de sintaxis matemática.
        """
        try:
            # Convertir string a expresión simbólica de Sympy
            expr = sp.sympify(self.func_str)
            
            # Validar que no haya variables desconocidas
            free_symbols = expr.free_symbols
            if not free_symbols.issubset({self.x_sym, self.y_sym}):
                raise ValueError(f"La función contiene variables no permitidas. Use solo 'x' e 'y'. Encontrado: {free_symbols}")

            # Convertir a función lambda compatible con numpy
            # modules='numpy' asegura que funciones como sin, exp usen la versión de numpy
            self.func_lambda = sp.lambdify((self.x_sym, self.y_sym), expr, modules='numpy')
            
            print(f"[INFO] Función interpretada correctamente: f(x,y) = {expr}")
            
        except Exception as e:
            print(f"[ERROR] No se pudo interpretar la función: {e}")
            sys.exit(1)

    def generate_mesh(self):
        """
        Genera la malla de coordenadas y evalúa la función Z = f(X, Y).
        """
        try:
            x = np.linspace(self.x_range[0], self.x_range[1], self.points)
            y = np.linspace(self.y_range[0], self.y_range[1], self.points)
            
            self.X, self.Y = np.meshgrid(x, y)
            
            # Evaluar Z. Se maneja el caso de funciones constantes o de una sola variable
            # transmitiendo la forma correcta si es necesario.
            self.Z = self.func_lambda(self.X, self.Y)
            
            # Si la función devuelve un escalar (ej: f(x,y) = 5), llenar el array
            if np.isscalar(self.Z):
                self.Z = np.full_like(self.X, self.Z)
            # Si devuelve forma incorrecta (ej: f(x,y) = x), asegurar broadcast
            elif self.Z.shape != self.X.shape:
                 self.Z = np.broadcast_to(self.Z, self.X.shape)
                 
        except Exception as e:
            print(f"[ERROR] Error matemático al evaluar la función (posible división por cero o dominio inválido): {e}")
            sys.exit(1)

    def calculate_volume_trapezoidal(self):
        """
        Calcula el volumen bajo la superficie usando la regla del trapecio doble con Numpy.
        Integral doble aproximada: ∫∫ f(x,y) dA
        """
        # Paso 1: Integrar a lo largo del eje X (axis=1 en meshgrid standard) para cada Y
        # np.trapz(y, x=coordenadas)
        # x_range define el dx implícito basado en la malla
        
        x_axis = np.linspace(self.x_range[0], self.x_range[1], self.points)
        y_axis = np.linspace(self.y_range[0], self.y_range[1], self.points)

        # Primera integral respecto a x (obtenemos área transversal para cada y)
        # axis=1 corresponde a las columnas (variación en x)
        integral_x = np.trapz(self.Z, x=x_axis, axis=1)
        
        # Segunda integral respecto a y (sumamos las áreas para obtener volumen)
        volume = np.trapz(integral_x, x=y_axis)
        
        return volume

    def plot_surface(self, volume_val):
        """
        Genera y muestra el gráfico 3D.
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Graficar superficie
        surf = ax.plot_surface(self.X, self.Y, self.Z, cmap=cm.viridis, 
                               linewidth=0, antialiased=False, alpha=0.8)

        # Elementos estéticos
        fig.colorbar(surf, shrink=0.5, aspect=5, label='f(x, y)')
        ax.set_title(f'Superficie: z = {self.func_str}\nVolumen calculado: {volume_val:.4f}', fontsize=14)
        ax.set_xlabel('Eje X')
        ax.set_ylabel('Eje Y')
        ax.set_zlabel('Eje Z')

        # Ajustar vista inicial
        ax.view_init(elev=30, azim=-60)
        
        print("[INFO] Generando gráfico...")
        plt.show()

def get_float_input(prompt):
    """Validador simple de entrada numérica."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Por favor, ingrese un número válido.")

def main():
    print("--- Calculadora de Superficies y Volúmenes 3D ---")
    print("Sintaxis soportada: +, -, *, /, **, sin, cos, exp, sqrt, log, etc.")
    print("Ejemplo de función: sin(sqrt(x**2 + y**2))")
    print("-" * 50)

    # 1. Entrada de datos
    func_input = input("Ingrese la función f(x, y): ").strip()
    
    print("\nDefinición del dominio [a, b] x [c, d]:")
    x_min = get_float_input("Límite inferior X (a): ")
    x_max = get_float_input("Límite superior X (b): ")
    y_min = get_float_input("Límite inferior Y (c): ")
    y_max = get_float_input("Límite superior Y (d): ")

    # Validaciones básicas de dominio
    if x_min >= x_max or y_min >= y_max:
        print("[ERROR] Los límites inferiores deben ser menores que los superiores.")
        return

    # 2. Instanciación y Procesamiento
    analyzer = SurfaceAnalyzer(func_input, (x_min, x_max), (y_min, y_max), points=100)
    
    # 3. Ejecución de pasos
    analyzer.parse_function()
    analyzer.generate_mesh()
    
    # 4. Cálculo numérico
    vol = analyzer.calculate_volume_trapezoidal()
    print(f"\n[RESULTADO] El volumen aproximado bajo la superficie es: {vol:.6f}")
    
    # 5. Visualización
    analyzer.plot_surface(vol)

if __name__ == "__main__":
    main()