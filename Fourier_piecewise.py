class FourierSeries:
    """
    Clase para calcular y visualizar series de Fourier para funciones
    periódicas por partes.
    """
    def __init__(self, period=2*np.pi, num_terms=100):
        """
        Inicializar la clase de series de Fourier.
        
        Parameters:
        -----------
        period : float
            Período de la función (por defecto 2π)
        num_terms : int
            Número de términos para calcular
        """
        self.period = period
        self.L = period / 2
        self.num_terms = num_terms
        self.a_coeffs = None  # coeficientes coseno
        self.b_coeffs = None  # coeficientes seno
        
    def define_function(self, f_callable):
        """
        Define la función para la cual calcular serie de Fourier.
        
        Parameters:
        -----------
        f_callable : callable
            Función definida por partes a analizar
        """
        self.f_callable = f_callable
        
    def calculate_coefficients(self, show_steps=True, num_coef=3):
        """
        Calcular los coeficientes a_k y b_k de la serie de Fourier.
        
        Parameters:
        -----------
        show_steps : bool
            Si mostrar los pasos del cálculo de las integrales
        num_coef : int
            Número de coeficientes a mostrar en los pasos (default 3)
        """
        self.a_coeffs = []
        self.b_coeffs = []
        
        # Calcular a_0
        def integrand_a0(x):
            return self.f_callable(x)
        
        a0, _ = integrate.quad(integrand_a0, -self.L, self.L)
        a0 = a0 / self.L
        self.a_coeffs.append(a0)
        
        if show_steps:
            display(Markdown("### Cálculo de coeficientes de Fourier"))
            display(Markdown(f"#### Cálculo de $a_0$"))
            display(Math(r"a_0 = \frac{1}{L} \int_{-L}^{L} f(x) \, dx"))
            display(Math(r"a_0 = \frac{1}{" + r"2\pi" + r"} \int_{-" + r"2\pi" + r"}^{" + r"2\pi" + r"} f(x) \, dx"))
            
            # Para nuestra función específica
            display(Markdown("Para nuestra función específica:"))
            display(Math(r"a_0 = \frac{1}{" + r"2\pi" + r"} \left[ \int_{-\pi/2}^{\pi/2} \cos(x) \, dx + \int_{-\pi}^{-\pi/2} 0 \, dx + \int_{\pi/2}^{\pi} 0 \, dx \right]"))
            display(Math(r"a_0 = \frac{1}{" + r"2\pi" + r"} \left[ \left. \sin(x) \right|_{-\pi/2}^{\pi/2} + 0 + 0 \right]"))
            display(Math(r"a_0 = \frac{1}{" + r"2\pi" + r"} \left[ \sin\left(\frac{\pi}{2}\right) - \sin\left(-\frac{\pi}{2}\right) \right]"))
            display(Math(r"a_0 = \frac{1}{" + r"2\pi" + r"} \left[ 1 - (-1) \right] = \frac{2}{" + r"2\pi" + r"} = " + str(a0)))
        
        # Calcular a_k y b_k para k > 0
        for k in range(1, self.num_terms + 1):
            # Calcular a_k: (1/L) * ∫ f(x)*cos(k*π*x/L) dx
            def integrand_ak(x, k=k):
                return self.f_callable(x) * np.cos(k * np.pi * x / self.L)
            
            ak, _ = integrate.quad(integrand_ak, -self.L, self.L)
            ak = ak / self.L
            self.a_coeffs.append(ak)
            
            if show_steps and k <= num_coef:
                display(Markdown(f"#### Cálculo de $a_{k}$"))
                display(Math(r"a_" + str(k) + r" = \frac{1}{L} \int_{-L}^{L} f(x) \cos\left(\frac{" + str(k) + r"\pi x}{L}\right) \, dx"))
                display(Math(r"a_" + str(k) + r" = \frac{1}{" + r"2\pi" + r"} \int_{-" + r"2\pi" + r"}^{" + r"2\pi" + r"} f(x) \cos\left(\frac{" + str(k) + r"x}{" + r"2\pi" + r"}\right) \, dx"))
                
                # Para nuestra función específica
                display(Markdown("Para nuestra función específica:"))
                display(Math(r"a_" + str(k) + r" = \frac{1}{" + r"2\pi" + r"} \int_{-\pi/2}^{\pi/2} \cos(x) \cos\left(" + str(k) + r"x\right) \, dx"))
                display(Math(r"a_" + str(k) + r" = " + str(ak)))
            
            # Calcular b_k: (1/L) * ∫ f(x)*sin(k*π*x/L) dx
            def integrand_bk(x, k=k):
                return self.f_callable(x) * np.sin(k * np.pi * x / self.L)
            
            bk, _ = integrate.quad(integrand_bk, -self.L, self.L)
            bk = bk / self.L
            self.b_coeffs.append(bk)
            
            if show_steps and k <= num_coef:
                display(Markdown(f"#### Cálculo de $b_{k}$"))
                display(Math(r"b_" + str(k) + r" = \frac{1}{L} \int_{-L}^{L} f(x) \sin\left(\frac{" + str(k) + r"\pi x}{L}\right) \, dx"))
                display(Math(r"b_" + str(k) + r" = \frac{1}{" + r"2\pi" + r"} \int_{-" + r"2\pi" + r"}^{" + r"2\pi" + r"} f(x) \sin\left(" + str(k) + r"x\right) \, dx"))
                
                # Para nuestra función específica
                display(Markdown("Para nuestra función específica:"))
                display(Math(r"b_" + str(k) + r" = \frac{1}{" + r"2\pi" + r"} \int_{-\pi/2}^{\pi/2} \cos(x) \sin\left(" + str(k) + r"x\right) \, dx"))
                display(Math(r"b_" + str(k) + r" = " + str(bk)))
    def get_approximation_function(self, terms=None):
        """
        Obtener una función que aproxima la serie de Fourier.
        
        Parameters:
        -----------
        terms : int
            Número de términos a incluir en la aproximación
            
        Returns:
        --------
        callable: Función de aproximación
        """
        if self.a_coeffs is None or self.b_coeffs is None:
            self.calculate_coefficients(show_steps=False)
            
        if terms is None:
            terms = self.num_terms
        else:
            terms = min(terms, self.num_terms)
            
        def approximation(x):
            # Asegurarse que x está en [-L, L]
            x = ((x + self.L) % (2*self.L)) - self.L
            
            # Inicializar con el término a_0/2
            result = self.a_coeffs[0] / 2
            
            # Sumar los términos a_k*cos(k*π*x/L) + b_k*sin(k*π*x/L)
            for k in range(1, terms + 1):
                result += self.a_coeffs[k] * np.cos(k * np.pi * x / self.L)
                result += self.b_coeffs[k-1] * np.sin(k * np.pi * x / self.L)
                
            return result
        
        return approximation
    
    def display_coefficients_table(self, num_to_show=10):
        """
        Mostrar tabla de coeficientes de Fourier.
        """
        if self.a_coeffs is None or self.b_coeffs is None:
            self.calculate_coefficients(show_steps=False)
            
        n_to_show = min(num_to_show, self.num_terms)
        
        data = {
            'n': [0] + list(range(1, n_to_show + 1)),
            'a_n': self.a_coeffs[:n_to_show + 1],
            'b_n': [0] + self.b_coeffs[:n_to_show]  # b_0 no existe
        }
        
        df = pd.DataFrame(data)
        display(df)
        return df
    
    def plot_function_and_approximations(self, x_range=None, num_points=1000, term_list=None):
        """
        Graficar la función original y sus aproximaciones de Fourier.
        
        Parameters:
        -----------
        x_range : tuple
            Rango (min_x, max_x) para graficar
        num_points : int
            Número de puntos para evaluar
        term_list : list
            Lista de números de términos para aproximaciones
        """
        if x_range is None:
            x_range = (-self.L*1.5, self.L*1.5)
            
        if term_list is None:
            term_list = [1, 5, 10, 20, 50]
            
        x = np.linspace(x_range[0], x_range[1], num_points)
        y_original = np.array([self.f_callable(xi) for xi in x])
        
        plt.figure(figsize=(12, 8))
        
        # Graficar función original
        plt.plot(x, y_original, 'k-', label='f(x) original', linewidth=2)
        
        # Graficar aproximaciones
        for terms in term_list:
            if terms <= self.num_terms:  # Verificar que el número de términos no exceda self.num_terms
                approx_func = self.get_approximation_function(terms)
                y_approx = np.array([approx_func(xi) for xi in x])
                plt.plot(x, y_approx, label=f'Fourier ({terms} términos)')
            else:
                print(f"Advertencia: El número de términos {terms} excede el máximo permitido ({self.num_terms}).")
        
        # Marcar límites del período
        plt.axvline(-self.L, color='gray', linestyle='--', alpha=0.7)
        plt.axvline(self.L, color='gray', linestyle='--', alpha=0.7)
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title("Serie de Fourier para f(x)")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        
        return plt.gcf()
    
    def calculate_error(self, x_points=None, term_list=None, tolerance=1e-10):
        """
        Calcular error entre aproximación y función real.
        
        Parameters:
        -----------
        x_points : list or array
            Puntos donde calcular el error
        term_list : list
            Lista de números de términos para evaluación
        tolerance : float
            Tolerancia para considerar un error como máximo (relativo al máximo absoluto)
            
        Returns:
        --------
        DataFrame con información de error y puntos donde ocurre el error máximo
        """
        if x_points is None:
            # Aumentar densidad de puntos cerca de las discontinuidades
            x1 = np.linspace(-self.L, -np.pi/2 - 0.01, 100)
            x2 = np.linspace(-np.pi/2 - 0.01, -np.pi/2 + 0.01, 200)  # Más puntos cerca de -π/2
            x3 = np.linspace(-np.pi/2 + 0.01, np.pi/2 - 0.01, 100)
            x4 = np.linspace(np.pi/2 - 0.01, np.pi/2 + 0.01, 200)    # Más puntos cerca de π/2
            x5 = np.linspace(np.pi/2 + 0.01, self.L, 100)
            x_points = np.concatenate((x1, x2, x3, x4, x5))
            
        if term_list is None:
            term_list = [1, 5, 10, 20, 50]
            
        # Calcular valores reales
        actual_values = np.array([self.f_callable(x) for x in x_points])
        
        # Para cada número de términos
        results = []
        
        for terms in term_list:
            if terms <= self.num_terms:
                approx_func = self.get_approximation_function(terms)
                approx_values = np.array([approx_func(x) for x in x_points])
                abs_error = np.abs(actual_values - approx_values)
                mse = np.mean(abs_error**2)
                max_error = np.max(abs_error)
                
                # Encontrar todos los puntos donde el error está cerca del máximo
                max_indices = np.where(abs_error >= max_error - tolerance)[0]
                max_points = x_points[max_indices]
                
                # Crear cadenas de notación π para cada punto
                max_point_labels = [self._format_point_as_pi(p) for p in max_points]
                
                # Para mostrar como una lista separada por comas
                if len(max_points) > 1:
                    max_point_str = ", ".join([f"{p:.6f}" for p in max_points])
                    max_point_label_str = ", ".join(max_point_labels)
                else:
                    max_point_str = f"{max_points[0]:.6f}"
                    max_point_label_str = max_point_labels[0]
                
                results.append({
                    'Términos': terms,
                    'Error cuadrático medio': mse,
                    'Error máximo': max_error,
                    'Puntos de error máximo': max_point_str,
                    'Puntos (notación π)': max_point_label_str
                })
        
        # Crear DataFrame
        df = pd.DataFrame(results)
        return df

    def _format_point_as_pi(self, point, close_tolerance = 0.01 ):
        """
        Convierte un punto numérico en notación π para mejor visualización.
        
        Parameters:
        -----------
        point : float
            Valor numérico a convertir
        close_tolerance : float

            
        Returns:
        --------
        str : Representación simbólica simplificada en términos de π
        """
        
        # Valores exactos comunes (usamos tolerancia más amplia para redondear)
        if abs(point - np.pi/2) < close_tolerance:
            return "π/2"
        elif abs(point + np.pi/2) < close_tolerance:
            return "-π/2"
        elif abs(point - np.pi) < close_tolerance:
            return "π"
        elif abs(point + np.pi) < close_tolerance:
            return "-π"
        elif abs(point) < close_tolerance:
            return "0"
        elif abs(point - np.pi/4) < close_tolerance:
            return "π/4"
        elif abs(point + np.pi/4) < close_tolerance:
            return "-π/4"
        
        # Para otros valores, mostrar en términos de π redondeado a 2 decimales
        pi_fraction = point / np.pi
        return f"{pi_fraction:.2f}π"

    def visualize_convergence(self, x_point=0, max_terms=None):
        """
        Visualizar cómo converge la serie de Fourier en un punto específico.
        
        Parameters:
        -----------
        x_point : float
            Punto donde verificar la convergencia
        max_terms : int
            Número máximo de términos a incluir
        """
        if max_terms is None:
            max_terms = self.num_terms
            
        # Calcular valor real
        actual = self.f_callable(x_point)
        
        # Calcular aproximaciones con términos crecientes
        terms_range = list(range(1, max_terms + 1))
        approximations = []
        errors = []
        
        for n in terms_range:
            approx_func = self.get_approximation_function(n)
            approx = approx_func(x_point)
            approximations.append(approx)
            errors.append(abs(actual - approx))
        
        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de valores de aproximación
        ax1.plot(terms_range, approximations, 'bo-')
        ax1.axhline(actual, color='r', linestyle='--', label=f'Valor exacto: {actual:.6f}')
        ax1.set_title(f'Convergencia en x = {x_point}')
        ax1.set_xlabel('Número de términos')
        ax1.set_ylabel('Valor de la aproximación')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Gráfico de errores
        ax2.semilogy(terms_range, errors, 'ro-')
        ax2.set_title(f'Error de aproximación en x = {x_point}')
        ax2.set_xlabel('Número de términos')
        ax2.set_ylabel('Error (escala logarítmica)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def visualize_max_error_points(self, term_list=None):
        """
        Visualizar los puntos donde ocurre el error máximo para diferentes términos.
        
        Parameters:
        -----------
        term_list : list
            Lista de términos a evaluar
            
        Returns:
        --------
        Figura matplotlib
        """
        if term_list is None:
            term_list = [1, 5, 10, 20, 50]
            
        # Calcular puntos de error máximo con mayor precisión
        # Usar más puntos cerca de las discontinuidades
        x1 = np.linspace(-self.L, -np.pi/2 - 0.01, 100)
        x2 = np.linspace(-np.pi/2 - 0.01, -np.pi/2 + 0.01, 200)  # Más puntos cerca de -π/2
        x3 = np.linspace(-np.pi/2 + 0.01, np.pi/2 - 0.01, 100)
        x4 = np.linspace(np.pi/2 - 0.01, np.pi/2 + 0.01, 200)    # Más puntos cerca de π/2
        x5 = np.linspace(np.pi/2 + 0.01, self.L, 100)
        x_points = np.concatenate((x1, x2, x3, x4, x5))
        
        error_df = self.calculate_error(x_points, term_list, tolerance=1e-10)
        
        # Configurar gráfico
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Graficar función original
        x = np.linspace(-self.L*1.5, self.L*1.5, 1000)
        y_original = np.array([self.f_callable(xi) for xi in x])
        ax.plot(x, y_original, 'k-', label='f(x) original', linewidth=2)
        
        # Graficar aproximaciones y marcar puntos de error máximo
        colors = plt.cm.tab10(np.linspace(0, 1, len(term_list)))
        
        for i, terms in enumerate(term_list):
            if terms <= self.num_terms:
                approx_func = self.get_approximation_function(terms)
                y_approx = np.array([approx_func(xi) for xi in x])
                ax.plot(x, y_approx, color=colors[i], label=f'Fourier ({terms} términos)')
                
                # Extraer todos los puntos máximos del string (formato "x1, x2, x3...")
                max_error_points_str = error_df.loc[error_df['Términos'] == terms, 'Puntos de error máximo'].values[0]
                max_error_points = [float(p) for p in max_error_points_str.split(", ")]
                
                # Marcar cada punto de error máximo
                for point in max_error_points:
                    max_error_value = approx_func(point)
                    ax.plot(point, max_error_value, 'o', color=colors[i], markersize=8)
        
        # Marcar límites del período
        ax.axvline(-self.L, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(self.L, color='gray', linestyle='--', alpha=0.7)
        
        # Añadir anotación para los puntos de discontinuidad
        ax.axvline(-np.pi/2, color='red', linestyle=':', alpha=0.7, 
                   label='Puntos de discontinuidad')
        ax.axvline(np.pi/2, color='red', linestyle=':', alpha=0.7)
        
        # Añadir etiquetas en los ejes para los puntos de discontinuidad
        ax.text(-np.pi/2, -0.1, r'$-\pi/2$', ha='center', color='red')
        ax.text(np.pi/2, -0.1, r'$\pi/2$', ha='center', color='red')
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=min(4, len(term_list)))
        ax.set_title("Serie de Fourier y puntos de error máximo")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        
        return fig

    def display_fourier_series(self, terms=None):
        """
        Mostrar la representación simbólica de la serie de Fourier.
        
        Parameters:
        -----------
        terms : int
            Número de términos a incluir en la representación (default: todos los términos disponibles)
            
        Returns:
        --------
        sympy.Expr: Representación simbólica de la serie de Fourier
        """
        if self.a_coeffs is None or self.b_coeffs is None:
            self.calculate_coefficients(show_steps=False)
        
        if terms is None:
            terms = self.num_terms
        else:
            terms = min(terms, self.num_terms)
        
        # Definir variable simbólica
        x = sp.symbols('x')
        
        # Construir la serie de Fourier
        fourier_series = self.a_coeffs[0] / 2  # Término constante a_0 / 2
        for k in range(1, terms + 1):
            fourier_series += self.a_coeffs[k] * sp.cos(k * sp.pi * x / self.L)
            fourier_series += self.b_coeffs[k - 1] * sp.sin(k * sp.pi * x / self.L)
        
        # Mostrar la serie
        display(fourier_series)
        return fourier_series
    
# Definimos nuestra función f(x)
def f(x):
    if abs(x) < np.pi/2:
        return np.cos(x)
    else:
        return 0

# Creamos una instancia de la serie de Fourier
fourier = FourierSeries(period=2*np.pi, num_terms=100)
fourier.define_function(f)

# Calculamos los coeficientes mostrando los pasos
fourier.calculate_coefficients(show_steps=True)

# Mostramos tabla de coeficientes
display(Markdown("### Tabla de coeficientes de Fourier"))
fourier.display_coefficients_table(num_to_show=15)

fourier.display_fourier_series(terms=15)

# Graficamos la función y sus aproximaciones
display(Markdown("### Gráfica de la función y sus aproximaciones de Fourier"))
fourier.plot_function_and_approximations()
plt.show()

# Comparamos diferentes valores de L (período) y su impacto
display(Markdown("### Efecto de cambiar el período L"))
for L in [np.pi, 2*np.pi, 4*np.pi]:
    fourier_L = FourierSeries(period=2*L, num_terms=100)
    fourier_L.define_function(f)
    fourier_L.calculate_coefficients(show_steps=False, num_coef=15)
    display(Markdown(f"#### Período = {sp.latex(sp.Rational(L / np.pi))}π"))
    fourier_L.plot_function_and_approximations(x_range=(-2*L, 2*L))
    plt.show()

# Analizamos los puntos de error máximo
display(Markdown("### Análisis detallado de errores máximos"))
error_df_detailed = fourier.calculate_error()
display(error_df_detailed)

# Visualizamos dónde ocurren los errores máximos
display(Markdown("### Visualización de puntos de error máximo"))
fourier.visualize_max_error_points()
plt.show()

# Analizamos específicamente la región cercana a las discontinuidades
display(Markdown("### Análisis cerca de discontinuidades"))

# Análisis cerca de π/2
display(Markdown("#### Análisis cerca de discontinuidad en π/2"))
x_cerca_disc_pos = np.linspace(np.pi/2 - 0.1, np.pi/2 + 0.1, 400)
error_discont_pos = fourier.calculate_error(x_cerca_disc_pos, [10, 20, 40, 50, 100])
display(error_discont_pos)

# Análisis cerca de -π/2
display(Markdown("#### Análisis cerca de discontinuidad en -π/2"))
x_cerca_disc_neg = np.linspace(-np.pi/2 - 0.1, -np.pi/2 + 0.1, 400)
error_discont_neg = fourier.calculate_error(x_cerca_disc_neg, [10, 20, 40, 50, 100])
display(error_discont_neg)

print("Observamos que los errores máximos ocurren muy cerca de las discontinuidades en x = ±π/2.")
print("Este es un ejemplo clásico del fenómeno de Gibbs, que se caracteriza por un sobrepasamiento")
print("(overshoot) en las discontinuidades cuando se aproxima una función con series de Fourier.")
print("\nA medida que aumentamos el número de términos:")
print("1. El error máximo no desaparece, se mantiene aproximadamente en un 9% del salto")
print("2. El punto donde ocurre el error máximo se acerca cada vez más a la discontinuidad")
print("3. La oscilación se concentra más cerca de la discontinuidad, haciéndose más estrecha")