import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
import sympy as sp
import pandas as pd

class TaylorSeries:
    """
    A class to calculate and visualize Taylor series approximations for any function.
    """
    def __init__(self, function_expr, variable=None, expansion_point=0):
        """
        Initialize with a symbolic function expression.
        
        Parameters:
        -----------
        function_expr : str or sympy expression
            The function to approximate with Taylor series
        variable : sympy.Symbol, optional
            The variable symbol to use (defaults to 'x' if None)
        expansion_point : float, optional
            Point around which to expand the Taylor series (default 0)
        """
        if variable is None:
            self.x_sym = sp.Symbol('x')
        else:
            self.x_sym = variable
        
        # Convert string to symbolic expression if needed
        if isinstance(function_expr, str):
            self.f_sym = sp.sympify(function_expr)
        else:
            self.f_sym = function_expr
            
        self.x0 = expansion_point
        self.coefficients = []
        self.derivatives = []
        self.formula = None
    
    def calculate_coefficients(self, max_terms=8):
        """
        Calculate Taylor series coefficients up to the specified number of terms.
        """
        self.coefficients = []
        self.derivatives = []
        
        for i in range(max_terms):
            if i == 0:
                derivative = self.f_sym
            else:
                derivative = self.f_sym.diff(self.x_sym, i)
                
            self.derivatives.append(derivative)
            
            # Calculate coefficient: f^(i)(x0) / i!
            coeff = derivative.subs(self.x_sym, self.x0) / factorial(i)
            self.coefficients.append(float(coeff))
        
        # Create the Taylor polynomial formula
        self.formula = sum(self.coefficients[i] * (self.x_sym - self.x0)**i 
                           for i in range(len(self.coefficients)))
        
        return self.coefficients
    
    def get_approximation_function(self):
        """
        Returns a callable function for the Taylor approximation.
        """
        coeffs = self.coefficients
        x0 = self.x0
        
        def taylor_approx(x):
            result = 0
            for i, coeff in enumerate(coeffs):
                result += coeff * (x - x0)**i
            return result
        
        return taylor_approx
    
    def numeric_function(self, x):
        """
        Evaluate the original function numerically.
        """
        f_lambda = sp.lambdify(self.x_sym, self.f_sym, 'numpy')
        return f_lambda(x)
    
    def get_formula(self, terms=None):
        """
        Returns the symbolic formula for the Taylor series.
        """
        if terms is None:
            terms = len(self.coefficients)
        else:
            terms = min(terms, len(self.coefficients))
            
        return sum(self.coefficients[i] * (self.x_sym - self.x0)**i 
                  for i in range(terms))
    
    def get_latex_formula(self, terms=None):
        """
        Returns LaTeX representation of the Taylor series.
        """
        if terms is None:
            terms = len(self.coefficients)
        else:
            terms = min(terms, len(self.coefficients))
            
        formula = self.get_formula(terms)
        return sp.latex(formula)
    
    def display_coefficients_table(self):
        """
        Display a table of coefficients and derivatives.
        """
        data = {
            'n': list(range(len(self.coefficients))),
            'Derivative f^(n)(x)': [sp.latex(d) for d in self.derivatives],
            'Value at x0': [float(d.subs(self.x_sym, self.x0)) for d in self.derivatives],
            'Coefficient a_n': self.coefficients,
            'Term': [f"{c}(x-{self.x0})^{i}" for i, c in enumerate(self.coefficients)]
        }
        
        return pd.DataFrame(data)
    
    def plot_approximations(self, x_range=(-3, 3), num_points=400, term_list=None, ylim=None):
        """
        Plot the original function and its Taylor approximations.
        
        Parameters:
        -----------
        x_range : tuple
            (min_x, max_x) range for plotting
        num_points : int
            Number of points to evaluate
        term_list : list
            List of approximation orders to plot
        ylim : tuple
            (min_y, max_y) for the plot
        """
        if term_list is None:
            term_list = [2, 4, 6, min(8, len(self.coefficients))]
        
        x = np.linspace(x_range[0], x_range[1], num_points)
        
        try:
            y_actual = self.numeric_function(x)
        except:
            # Handle potential errors in function evaluation
            y_actual = np.zeros_like(x)
            for i, x_val in enumerate(x):
                try:
                    y_actual[i] = float(self.f_sym.subs(self.x_sym, x_val))
                except:
                    y_actual[i] = np.nan
        
        plt.figure(figsize=(12, 8))
        
        # Plot approximations
        for n in term_list:
            if n <= len(self.coefficients):
                approx_func = lambda x: sum(self.coefficients[i] * (x - self.x0)**i for i in range(n))
                y_taylor = np.array([approx_func(x_val) for x_val in x])
                plt.plot(x, y_taylor, label=f"Taylor ({n} términos)")
        
        # Plot original function
        plt.plot(x, y_actual, 'k-', label='f(x) original', linewidth=2)
        
        # Add expansion point
        plt.scatter([self.x0], [self.numeric_function(self.x0)], 
                   color='red', s=50, zorder=5, label=f'Punto de expansión (x₀={self.x0})')
        
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(f"Serie de Taylor de f(x) = {sp.latex(self.f_sym)}")
        plt.grid(True)
        
        if ylim:
            plt.ylim(ylim)
            
        return plt.gcf()
    
    def calculate_error(self, x_points, term_list=None):
        """
        Calculate error between approximation and actual function at specified points.
        """
        if term_list is None:
            term_list = [2, 4, 6, min(8, len(self.coefficients))]
        
        results = {}
        for x_val in x_points:
            exact = float(self.f_sym.subs(self.x_sym, x_val))
            errors = {}
            for n in term_list:
                if n <= len(self.coefficients):
                    approx = sum(self.coefficients[i] * (x_val - self.x0)**i for i in range(n))
                    errors[n] = abs(exact - approx)
            results[x_val] = errors
            
        # Create DataFrame
        df = pd.DataFrame(columns=['x'] + [f'Error ({n} términos)' for n in term_list])
        for i, x_val in enumerate(x_points):
            row = {'x': x_val}
            for n in term_list:
                if n in results[x_val]:
                    row[f'Error ({n} términos)'] = results[x_val][n]
            df.loc[i] = row
            
        return df

# Example usage for Taylor Series
if __name__ == "__main__":
    # Define the rational function
    expr = "(x**3 - x**2 - 4*x)/(2*x**2 + 4*x + 4)"
    
    # Create Taylor series object
    taylor = TaylorSeries(expr, expansion_point=0)
    
    # Calculate coefficients
    taylor.calculate_coefficients(max_terms=8)
    
    # Print original function
    print("Función original:")
    print(f"f(x) = {sp.latex(taylor.f_sym)}")
    
    # Display table of coefficients
    print("\nTabla de coeficientes y derivadas:")
    df_coeffs = taylor.display_coefficients_table()
    print(df_coeffs)
    
    # Display Taylor series formulas for different orders
    print("\nFórmulas de la serie de Taylor:")
    n_terms_list = [2, 4, 6, 8]
    for n in n_terms_list:
        formula = taylor.get_latex_formula(n)
        print(f"\nPolinomio de Taylor con {n} términos:")
        print(f"P_{n}(x) = {formula}")
    
    # Plot the approximations
    fig = taylor.plot_approximations(ylim=(-5, 5))
    plt.show()
    
    # Calculate error at specific points
    test_points = [-2, -1, 0, 1, 2]
    print("\nError de aproximación en puntos específicos:")
    error_df = taylor.calculate_error(test_points)
    print(error_df)