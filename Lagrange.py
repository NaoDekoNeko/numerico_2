import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd

class LagrangeInterpolation:
    """
    A class to perform Lagrange's interpolation method for polynomial fitting.
    """
    def __init__(self, x_points=None, y_points=None):
        """
        Initialize with optional data points.
        
        Parameters:
        -----------
        x_points : array-like, optional
            x coordinates of data points
        y_points : array-like, optional
            y coordinates of data points
        """
        self.x = None
        self.y = None
        self.L_polynomials = None
        self.polynomial = None
        
        if x_points is not None and y_points is not None:
            self.set_points(x_points, y_points)
    
    def set_points(self, x_points, y_points):
        """
        Set the data points to interpolate.
        
        Parameters:
        -----------
        x_points : array-like
            x coordinates of data points
        y_points : array-like
            y coordinates of data points
        """
        if len(x_points) != len(y_points):
            raise ValueError("x_points and y_points must have the same length")
            
        self.x = np.array(x_points, dtype=float)
        self.y = np.array(y_points, dtype=float)
        self.L_polynomials = None  # Reset Lagrange basis polynomials
        self.polynomial = None     # Reset interpolation polynomial
    
    def compute_lagrange_basis(self, i, x_val):
        """
        Compute the i-th Lagrange basis polynomial at a given x value.
        
        Parameters:
        -----------
        i : int
            Index of the Lagrange basis polynomial
        x_val : float or array-like
            Point(s) at which to evaluate the basis polynomial
        
        Returns:
        --------
        float or numpy.ndarray
            Value(s) of the i-th Lagrange basis polynomial at x_val
        """
        n = len(self.x)
        result = 1.0
        
        for j in range(n):
            if j != i:
                result = result * (x_val - self.x[j]) / (self.x[i] - self.x[j])
                
        return result
    
    def evaluate(self, x_eval):
        """
        Evaluate the Lagrange interpolation polynomial at given points.
        
        Parameters:
        -----------
        x_eval : float or array-like
            Points at which to evaluate the polynomial
        
        Returns:
        --------
        float or numpy.ndarray
            Value(s) of the polynomial at x_eval
        """
        # Convert input to numpy array if it's a scalar
        scalar_input = np.isscalar(x_eval)
        x_array = np.array([x_eval]) if scalar_input else np.array(x_eval)
        
        result = np.zeros_like(x_array, dtype=float)
        n = len(self.x)
        
        for i in range(len(x_array)):
            sum_val = 0.0
            for j in range(n):
                sum_val += self.y[j] * self.compute_lagrange_basis(j, x_array[i])
            result[i] = sum_val
            
        return result[0] if scalar_input else result
    
    def get_lagrange_basis_symbolic(self):
        """
        Get the symbolic expressions for all Lagrange basis polynomials.
        
        Returns:
        --------
        list of sympy expressions
            List of Lagrange basis polynomials in symbolic form
        """
        x = sp.Symbol('x')
        n = len(self.x)
        L_polynomials = []
        
        for i in range(n):
            L_i = 1
            for j in range(n):
                if j != i:
                    L_i *= (x - self.x[j]) / (self.x[i] - self.x[j])
            L_polynomials.append(L_i)
            
        self.L_polynomials = L_polynomials
        return L_polynomials
    
    def get_polynomial_symbolic(self):
        """
        Get the symbolic expression for the full Lagrange interpolation polynomial.
        
        Returns:
        --------
        sympy expression
            Symbolic form of the Lagrange polynomial
        """
        if self.L_polynomials is None:
            self.get_lagrange_basis_symbolic()
            
        polynomial = 0
        for i, L_i in enumerate(self.L_polynomials):
            polynomial += self.y[i] * L_i
            
        self.polynomial = polynomial
        return polynomial
    
    def get_polynomial_terms(self):
        """
        Get the terms of the Lagrange form polynomial.
        
        Returns:
        --------
        list of strings
            Terms of the Lagrange polynomial in readable form
        """
        if self.L_polynomials is None:
            self.get_lagrange_basis_symbolic()
            
        terms = []
        
        for i, basis in enumerate(self.L_polynomials):
            term = f"{self.y[i]} * L_{i}(x)"
            terms.append(term)
            
        return terms
    
    def get_polynomial_string(self):
        """
        Get the Lagrange interpolation polynomial as a readable string.
        
        Returns:
        --------
        str
            String representation of the Lagrange polynomial
        """
        terms = self.get_polynomial_terms()
        
        # Join terms with appropriate signs
        polynomial = terms[0]
        for term in terms[1:]:
            polynomial += f" + {term}"
                
        return polynomial
    
    def get_standard_form_symbolic(self):
        """
        Convert the Lagrange form to standard form using sympy.
        
        Returns:
        --------
        sympy expression
            Standard form of the polynomial
        """
        if self.polynomial is None:
            self.get_polynomial_symbolic()
            
        # Expand to standard form
        expanded_poly = sp.expand(self.polynomial)
        
        return expanded_poly
    
    def get_standard_form_string(self):
        """
        Get the standard form of the polynomial as a string.
        
        Returns:
        --------
        str
            String representation of the polynomial in standard form
        """
        expanded_poly = self.get_standard_form_symbolic()
        return str(expanded_poly)
    
    def get_standard_form_latex(self):
        """
        Get the standard form of the polynomial in LaTeX format.
        
        Returns:
        --------
        str
            LaTeX representation of the polynomial in standard form
        """
        expanded_poly = self.get_standard_form_symbolic()
        return sp.latex(expanded_poly)
    
    def plot_interpolation(self, x_range=None, num_points=1000):
        """
        Plot the data points and the interpolation polynomial.
        
        Parameters:
        -----------
        x_range : tuple, optional
            Range (x_min, x_max) for plotting the polynomial
        num_points : int, optional
            Number of points to use for plotting the smooth curve
        
        Returns:
        --------
        matplotlib figure
        """
        if x_range is None:
            # Extend range slightly beyond data points
            x_min, x_max = min(self.x), max(self.x)
            margin = (x_max - x_min) * 0.1
            x_range = (x_min - margin, x_max + margin)
            
        # Generate points for smooth curve
        x_curve = np.linspace(x_range[0], x_range[1], num_points)
        y_curve = self.evaluate(x_curve)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the data points
        ax.scatter(self.x, self.y, color='red', s=50, label='Puntos de datos')
        
        # Plot the interpolation curve
        ax.plot(x_curve, y_curve, 'b-', label='Polinomio de interpolación')
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Interpolación de Lagrange')
        ax.legend()
        
        return fig
    
    def display_lagrange_basis(self, x_range=None, num_points=1000):
        """
        Plot each Lagrange basis polynomial.
        
        Parameters:
        -----------
        x_range : tuple, optional
            Range (x_min, x_max) for plotting
        num_points : int, optional
            Number of points to use for plotting the smooth curves
        
        Returns:
        --------
        matplotlib figure
        """
        if x_range is None:
            # Extend range slightly beyond data points
            x_min, x_max = min(self.x), max(self.x)
            margin = (x_max - x_min) * 0.1
            x_range = (x_min - margin, x_max + margin)
            
        # Generate points for smooth curves
        x_curve = np.linspace(x_range[0], x_range[1], num_points)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each Lagrange basis polynomial
        for i in range(len(self.x)):
            y_basis = np.array([self.compute_lagrange_basis(i, x) for x in x_curve])
            ax.plot(x_curve, y_basis, label=f'L_{i}(x)')
            
        # Plot the data points
        ax.scatter(self.x, np.ones_like(self.x), color='red', s=50)
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('L_i(x)')
        ax.set_title('Polinomios base de Lagrange')
        ax.legend()
        
        return fig
    
    def get_error_formula(self, x_eval):
        """
        Get a symbolic representation of the error formula for Lagrange interpolation.
        
        Parameters:
        -----------
        x_eval : float
            Point at which to evaluate the error formula
        
        Returns:
        --------
        str
            Symbolic representation of the error formula
        """
        n = len(self.x)
        
        # Create the product term (x - x_0)(x - x_1)...(x - x_{n-1})
        product = f"(ξ)"
        for i in range(n):
            if self.x[i] >= 0:
                product += f"(x - {self.x[i]})"
            else:
                product += f"(x + {-self.x[i]})"
        
        # Create the error formula f^(n)(ξ)/n! * product
        error_formula = f"f^({n})(ξ)/{n}! * {product}"
        
        return error_formula
    
    def get_error_bounds(self, x_eval, max_derivative):
        """
        Estimate error bounds for the interpolation at given points.
        
        Parameters:
        -----------
        x_eval : float or array-like
            Points at which to estimate error bounds
        max_derivative : float
            Maximum absolute value of the n-th derivative in the interval
        
        Returns:
        --------
        float or numpy.ndarray
            Estimated upper bounds of the error
        """
        n = len(self.x)
        
        # Convert input to numpy array if it's a scalar
        scalar_input = np.isscalar(x_eval)
        x_array = np.array([x_eval]) if scalar_input else np.array(x_eval)
        
        error_bounds = np.zeros_like(x_array, dtype=float)
        
        for i in range(len(x_array)):
            # Calculate product term (x - x_0)(x - x_1)...(x - x_{n-1})
            product = 1.0
            for j in range(n):
                product *= (x_array[i] - self.x[j])
                
            # Calculate factorial n!
            factorial_n = np.math.factorial(n)
            
            # Calculate error bound
            error_bounds[i] = abs(product) * max_derivative / factorial_n
            
        return error_bounds[0] if scalar_input else error_bounds