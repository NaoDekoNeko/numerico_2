import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from IPython.display import display, Math, Markdown

class CubicSpline:
    """
    Clase para interpolar puntos utilizando splines cúbicos.
    """
    def __init__(self, x=None, y=None, boundary_type='natural'):
        """
        Inicializar la clase de interpolación con splines cúbicos.
        
        Parameters:
        -----------
        x : array_like
            Valores x de los puntos a interpolar
        y : array_like
            Valores y de los puntos a interpolar
        boundary_type : str
            Tipo de condición de frontera ('natural', 'clamped', 'not-a-knot')
        """
        self.x = x
        self.y = y
        self.boundary_type = boundary_type
        self.coefficients = None
        self.spline_func = None
        
    def set_points(self, x, y):
        """
        Definir los puntos para la interpolación.
        
        Parameters:
        -----------
        x : array_like
            Valores x de los puntos a interpolar
        y : array_like
            Valores y de los puntos a interpolar
        """
        self.x = np.array(x)
        self.y = np.array(y)
        
        # Ordenar puntos por x si no están ordenados
        if not np.all(np.diff(self.x) > 0):
            idx = np.argsort(self.x)
            self.x = self.x[idx]
            self.y = self.y[idx]
        
        # Reiniciar los coeficientes ya que los puntos han cambiado
        self.coefficients = None
        self.spline_func = None
        
    def calculate_coefficients(self, show_steps=False):
        """
        Calcular los coeficientes de los splines cúbicos.
        
        Parameters:
        -----------
        show_steps : bool
            Si mostrar los pasos del cálculo
        """
        if self.x is None or self.y is None:
            raise ValueError("Debe definir los puntos x e y antes de calcular los coeficientes")
            
        n = len(self.x) - 1  # Número de intervalos
        
        # Diferencias y pasos
        h = np.diff(self.x)
        
        # Calcular las diferencias divididas
        delta = np.diff(self.y) / h
        
        # Construir el sistema tridiagonal para las segundas derivadas
        A = np.zeros((n+1, n+1))
        b = np.zeros(n+1)
        
        # Condiciones internas (continuidad de la segunda derivada)
        for i in range(1, n):
            A[i, i-1] = h[i-1]
            A[i, i] = 2 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            b[i] = 6 * (delta[i] - delta[i-1])
            
        # Condiciones de frontera
        if self.boundary_type == 'natural':
            # Segundas derivadas en los extremos son cero
            A[0, 0] = 1.0
            A[n, n] = 1.0
            b[0] = 0.0
            b[n] = 0.0
        elif self.boundary_type == 'clamped':
            # Se especifican las primeras derivadas en los extremos
            # Por defecto las tomamos como 0
            A[0, 0] = 2 * h[0]
            A[0, 1] = h[0]
            A[n, n-1] = h[n-1]
            A[n, n] = 2 * h[n-1]
            b[0] = 6 * (delta[0] - 0)  # f'(x₀) = 0
            b[n] = 6 * (0 - delta[n-1])  # f'(xₙ) = 0
        elif self.boundary_type == 'not-a-knot':
            # Tercera derivada es continua en x₁ y xₙ₋₁
            A[0, 0] = h[1]
            A[0, 1] = -(h[0] + h[1])
            A[0, 2] = h[0]
            A[n, n-2] = h[n-1]
            A[n, n-1] = -(h[n-2] + h[n-1])
            A[n, n] = h[n-2]
        
        # Resolver el sistema para obtener las segundas derivadas
        c = np.linalg.solve(A, b)
        
        # Calcular los coeficientes a, b, d
        a = self.y[:-1]
        b = delta - h * (2 * c[:-1] + c[1:]) / 6
        d = np.diff(c) / (6 * h)
        
        # Almacenar los coeficientes (a, b, c, d) para cada intervalo
        self.coefficients = {
            'a': a,
            'b': b,
            'c': c[:-1] / 2,  # Dividido por 2 para ajustar a la forma estándar
            'd': d
        }
        
        # Crear una función usando scipy para evaluación rápida
        self.spline_func = interpolate.CubicSpline(self.x, self.y, bc_type=self.boundary_type)
        
        if show_steps:
            self._display_calculation_steps(h, delta, A, b, c)
            
        return self.coefficients
    
    def _display_calculation_steps(self, h, delta, A, b, c):
        """
        Mostrar los pasos de cálculo para fines educativos.
        """
        display(Markdown("### Cálculo de coeficientes para splines cúbicos"))
        
        # Mostrar puntos
        display(Markdown("#### Puntos de interpolación"))
        points_data = {'x': self.x, 'y': self.y}
        display(pd.DataFrame(points_data))
        
        # Mostrar diferencias
        display(Markdown("#### Diferencias y pasos"))
        diff_data = {'i': range(len(h)), 'h_i': h, 'Δy_i': np.diff(self.y), 'δ_i': delta}
        display(pd.DataFrame(diff_data))
        
        # Mostrar sistema tridiagonal
        display(Markdown("#### Sistema tridiagonal para segundas derivadas"))
        display(Markdown("Matriz A:"))
        display(A)
        display(Markdown("Vector b:"))
        display(b)
        
        # Mostrar solución (segundas derivadas)
        display(Markdown("#### Segundas derivadas en los nodos (c_i)"))
        display(c)
        
        # Mostrar coeficientes finales
        display(Markdown("#### Coeficientes de los splines cúbicos"))
        coef_data = {
            'Intervalo': [f"[{self.x[i]:.4f}, {self.x[i+1]:.4f}]" for i in range(len(self.x)-1)],
            'a_i': self.coefficients['a'],
            'b_i': self.coefficients['b'],
            'c_i': self.coefficients['c'] * 2,  # Multiplicado por 2 para mostrar la segunda derivada original
            'd_i': self.coefficients['d']
        }
        display(pd.DataFrame(coef_data))
        
        # Explicación de la fórmula
        display(Markdown("""
        #### Fórmula del spline cúbico en cada intervalo
        Para $x \\in [x_i, x_{i+1}]$, el spline $S_i(x)$ es:
        
        $S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3$
        
        donde:
        - $a_i = f(x_i)$
        - $b_i = \\frac{f(x_{i+1}) - f(x_i)}{h_i} - \\frac{h_i}{6}(2c_i + c_{i+1})$
        - $c_i = \\frac{f''(x_i)}{2}$
        - $d_i = \\frac{f''(x_{i+1}) - f''(x_i)}{6h_i}$
        """))
    
    def evaluate(self, x_eval):
        """
        Evaluar el spline cúbico en puntos específicos.
        
        Parameters:
        -----------
        x_eval : array_like
            Puntos donde evaluar el spline
            
        Returns:
        --------
        array_like: Valores interpolados
        """
        if self.spline_func is None:
            self.calculate_coefficients()
            
        return self.spline_func(x_eval)
    
    def plot_interpolation(self, num_points=1000, show_points=True, show_derivatives=False):
        """
        Graficar la interpolación por splines cúbicos.
        
        Parameters:
        -----------
        num_points : int
            Número de puntos para evaluar el spline
        show_points : bool
            Si mostrar los puntos originales
        show_derivatives : bool
            Si mostrar las derivadas primera y segunda
            
        Returns:
        --------
        matplotlib.figure.Figure: Figura de matplotlib
        """
        if self.spline_func is None:
            self.calculate_coefficients()
            
        # Crear puntos para evaluación
        x_range = np.linspace(min(self.x), max(self.x), num_points)
        y_interp = self.evaluate(x_range)
        
        if show_derivatives:
            # Crear figura con subplots para función y derivadas
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            
            # Función interpolada
            ax1.plot(x_range, y_interp, 'b-', label='Spline cúbico')
            if show_points:
                ax1.plot(self.x, self.y, 'ro', label='Puntos originales')
            ax1.set_title('Interpolación con spline cúbico')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Primera derivada
            dy_dx = self.spline_func.derivative(1)(x_range)
            ax2.plot(x_range, dy_dx, 'g-', label='Primera derivada')
            ax2.set_title('Primera derivada del spline')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Segunda derivada
            d2y_dx2 = self.spline_func.derivative(2)(x_range)
            ax3.plot(x_range, d2y_dx2, 'r-', label='Segunda derivada')
            ax3.set_title('Segunda derivada del spline')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
        else:
            # Crear figura simple con la función interpolada
            fig, ax = plt.subplots(figsize=(12, 8))
            
            ax.plot(x_range, y_interp, 'b-', label='Spline cúbico')
            if show_points:
                ax.plot(self.x, self.y, 'ro', label='Puntos originales')
                
            # Marcar los intervalos
            for xi in self.x:
                ax.axvline(xi, color='gray', linestyle='--', alpha=0.3)
                
            ax.set_title('Interpolación con spline cúbico')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        return fig
    
    def calculate_error(self, f_true=None, x_eval=None, num_points=1000):
        """
        Calcular error entre el spline y una función verdadera.
        
        Parameters:
        -----------
        f_true : callable
            Función verdadera a comparar
        x_eval : array_like
            Puntos donde calcular el error (por defecto, malla fina)
        num_points : int
            Número de puntos a evaluar si x_eval no está especificado
            
        Returns:
        --------
        DataFrame con información de error
        """
        if self.spline_func is None:
            self.calculate_coefficients()
            
        if f_true is None:
            raise ValueError("Debe proporcionar una función verdadera para calcular el error")
            
        if x_eval is None:
            x_eval = np.linspace(min(self.x), max(self.x), num_points)
            
        # Evaluar función verdadera y spline
        y_true = np.array([f_true(xi) for xi in x_eval])
        y_spline = self.evaluate(x_eval)
        
        # Calcular errores
        abs_error = np.abs(y_true - y_spline)
        max_error = np.max(abs_error)
        max_error_x = x_eval[np.argmax(abs_error)]
        mean_error = np.mean(abs_error)
        rms_error = np.sqrt(np.mean(abs_error**2))
        
        # Calcular errores por intervalo
        interval_errors = []
        for i in range(len(self.x) - 1):
            mask = (x_eval >= self.x[i]) & (x_eval <= self.x[i+1])
            if np.any(mask):
                interval_max_error = np.max(abs_error[mask])
                interval_max_x = x_eval[mask][np.argmax(abs_error[mask])]
                interval_mean_error = np.mean(abs_error[mask])
                interval_errors.append({
                    'Intervalo': f"[{self.x[i]:.4f}, {self.x[i+1]:.4f}]",
                    'Error máximo': interval_max_error,
                    'Posición del error máximo': interval_max_x,
                    'Error medio': interval_mean_error
                })
        
        # Crear DataFrame con resultados globales
        global_results = pd.DataFrame({
            'Métrica': ['Error máximo', 'Posición del error máximo', 'Error medio', 'Error RMS'],
            'Valor': [max_error, max_error_x, mean_error, rms_error]
        })
        
        # Crear DataFrame con resultados por intervalo
        interval_results = pd.DataFrame(interval_errors)
        
        return {
            'global': global_results,
            'interval': interval_results,
            'x_eval': x_eval,
            'abs_error': abs_error
        }
    
    def plot_error(self, error_data):
        """
        Graficar el error de interpolación.
        
        Parameters:
        -----------
        error_data : dict
            Datos de error generados por calculate_error()
            
        Returns:
        --------
        matplotlib.figure.Figure: Figura de matplotlib
        """
        x_eval = error_data['x_eval']
        abs_error = error_data['abs_error']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Gráfico del error absoluto
        ax1.plot(x_eval, abs_error, 'r-')
        ax1.set_title('Error absoluto')
        ax1.set_xlabel('x')
        ax1.set_ylabel('|f(x) - S(x)|')
        ax1.grid(True, alpha=0.3)
        
        # Marcar los nodos de interpolación
        for xi in self.x:
            ax1.axvline(xi, color='gray', linestyle='--', alpha=0.3)
            
        # Marcar punto de error máximo
        max_error_idx = np.argmax(abs_error)
        max_error_x = x_eval[max_error_idx]
        max_error = abs_error[max_error_idx]
        ax1.plot(max_error_x, max_error, 'bo', markersize=8)
        ax1.annotate(f'Error máx: {max_error:.2e} en x={max_error_x:.4f}', 
                     xy=(max_error_x, max_error),
                     xytext=(max_error_x, max_error*1.1),
                     arrowprops=dict(arrowstyle="->"))
        
        # Gráfico del error por intervalo
        interval_data = error_data['interval']
        intervals = interval_data['Intervalo']
        interval_errors = interval_data['Error máximo']
        
        ax2.bar(range(len(intervals)), interval_errors, tick_label=intervals)
        ax2.set_title('Error máximo por intervalo')
        ax2.set_xlabel('Intervalo')
        ax2.set_ylabel('Error máximo')
        ax2.set_xticks(range(len(intervals)))
        ax2.set_xticklabels(intervals, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def compare_boundary_conditions(self, f_true, boundary_types=['natural', 'clamped', 'not-a-knot']):
        """
        Comparar diferentes condiciones de frontera.
        
        Parameters:
        -----------
        f_true : callable
            Función verdadera a comparar
        boundary_types : list
            Lista de tipos de condiciones de frontera a comparar
            
        Returns:
        --------
        matplotlib.figure.Figure: Figura de matplotlib
        """
        # Guardar el tipo de frontera original
        original_type = self.boundary_type
        
        x_eval = np.linspace(min(self.x), max(self.x), 1000)
        y_true = np.array([f_true(xi) for xi in x_eval])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Colores para cada tipo
        colors = {'natural': 'blue', 'clamped': 'green', 'not-a-knot': 'purple'}
        
        error_results = {}
        
        # Evaluar cada tipo de frontera
        for bt in boundary_types:
            self.boundary_type = bt
            self.calculate_coefficients()
            y_spline = self.evaluate(x_eval)
            abs_error = np.abs(y_true - y_spline)
            
            # Guardar resultados
            error_results[bt] = {
                'max_error': np.max(abs_error),
                'mean_error': np.mean(abs_error),
                'rms_error': np.sqrt(np.mean(abs_error**2))
            }
            
            # Graficar interpolación
            ax1.plot(x_eval, y_spline, '-', color=colors[bt], label=f'Spline ({bt})')
            
            # Graficar error
            ax2.plot(x_eval, abs_error, '-', color=colors[bt], label=f'Error ({bt})')
        
        # Graficar función verdadera
        ax1.plot(x_eval, y_true, 'k--', label='Función verdadera')
        ax1.plot(self.x, self.y, 'ro', label='Puntos interpolados')
        
        # Configurar gráficos
        ax1.set_title('Comparación de condiciones de frontera - Interpolación')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Comparación de condiciones de frontera - Error absoluto')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Restaurar tipo de frontera original
        self.boundary_type = original_type
        
        # Crear tabla de errores
        error_df = pd.DataFrame({
            'Tipo de frontera': list(error_results.keys()),
            'Error máximo': [error_results[bt]['max_error'] for bt in error_results],
            'Error medio': [error_results[bt]['mean_error'] for bt in error_results],
            'Error RMS': [error_results[bt]['rms_error'] for bt in error_results]
        })
        
        plt.tight_layout()
        return fig, error_df

# Ejemplo de uso y prueba de la clase
if __name__ == "__main__":
    x = [1950, 1960, 1970, 1980, 1990, 2000]
    y = [123.5, 131.2, 150.7, 141.3, 203.2, 240.5]

    # Crear la instancia de spline
    spline = CubicSpline()

    # Configurar los puntos
    spline.set_points(x, y)

    # Calcular los coeficientes
    coeffs = spline.calculate_coefficients()

    # Ahora puedes evaluar el spline en cualquier punto
    x_eval = 1965
    y_interpolado = spline.evaluate(x_eval)
    print(f"Valor interpolado en x={x_eval}: {y_interpolado}")

    # Si quieres ver los polinomios por tramos:
    print("\nPolinomios por tramos:")
    for i in range(len(x)-1):
        a = coeffs['a'][i]
        b = coeffs['b'][i]
        c = coeffs['c'][i]
        d = coeffs['d'][i]
        x_i = x[i]
        print(f"En [{x[i]}, {x[i+1]}]: S(x) = {a} + {b}(x-{x_i}) + {c}(x-{x_i})² + {d}(x-{x_i})³")

    # Visualizar el spline
    fig = spline.plot_interpolation()