import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd

class NewtonInterpolation:
    """
    A class to perform Newton's interpolation method for polynomial fitting.
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
        self.coefficients = None
        self.divided_differences = None
        
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
        self.coefficients = None  # Reset coefficients
        self.divided_differences = None  # Reset divided differences
    
    def compute_divided_differences(self):
        """
        Compute the divided differences table for Newton interpolation.
        
        Returns:
        --------
        numpy.ndarray
            Table of divided differences
        """
        n = len(self.x)
        # Initialize divided differences table
        # Each row represents a level of differences
        table = np.zeros((n, n))
        
        # First column is y values
        table[:, 0] = self.y
        
        # Calculate divided differences
        for j in range(1, n):
            for i in range(n - j):
                table[i, j] = (table[i+1, j-1] - table[i, j-1]) / (self.x[i+j] - self.x[i])
        
        self.divided_differences = table
        
        # Extract coefficients (first row of divided differences table)
        self.coefficients = table[0, :]
        
        return table
    
    def evaluate(self, x_eval):
        """
        Evaluate the Newton interpolation polynomial at given points.
        
        Parameters:
        -----------
        x_eval : float or array-like
            Points at which to evaluate the polynomial
        
        Returns:
        --------
        float or numpy.ndarray
            Value(s) of the polynomial at x_eval
        """
        if self.coefficients is None:
            self.compute_divided_differences()
            
        # Convert input to numpy array if it's a scalar
        scalar_input = np.isscalar(x_eval)
        x_array = np.array([x_eval]) if scalar_input else np.array(x_eval)
        
        n = len(self.x)
        result = np.zeros_like(x_array, dtype=float)
        
        for i in range(len(x_array)):
            # Start with the constant term (first coefficient)
            temp = self.coefficients[n-1]
            
            # Build up the polynomial using Horner's method
            for j in range(n-2, -1, -1):
                temp = temp * (x_array[i] - self.x[j]) + self.coefficients[j]
                
            result[i] = temp
            
        return result[0] if scalar_input else result
    
    def get_polynomial_terms(self):
        """
        Get the terms of the Newton form polynomial.
        
        Returns:
        --------
        list of strings
            Terms of the Newton polynomial in readable form
        """
        if self.coefficients is None:
            self.compute_divided_differences()
            
        terms = []
        
        # First term is just the constant coefficient
        terms.append(f"{self.coefficients[0]:.6g}")
        
        # Build up remaining terms
        for i in range(1, len(self.coefficients)):
            if self.coefficients[i] == 0:
                continue  # Skip zero coefficients
                
            term = f"{self.coefficients[i]:.6g}"
            
            # Multiply by (x - x_j) factors
            for j in range(i):
                x_j = self.x[j]
                if x_j >= 0:
                    term += f"(x - {x_j})"
                else:
                    term += f"(x + {-x_j})"
                
            terms.append(term)
            
        return terms
    
    def get_polynomial_string(self):
        """
        Get the Newton interpolation polynomial as a readable string.
        
        Returns:
        --------
        str
            String representation of the Newton polynomial
        """
        terms = self.get_polynomial_terms()
        
        # Join terms with appropriate signs
        polynomial = terms[0]
        for term in terms[1:]:
            if term.startswith('-'):
                polynomial += f" {term}"
            else:
                polynomial += f" + {term}"
                
        return polynomial
    
    def get_standard_form_symbolic(self):
        """
        Convert the Newton form to standard form using sympy.
        
        Returns:
        --------
        sympy expression
            Standard form of the polynomial
        """
        # Create symbolic variable
        x = sp.Symbol('x')
        
        # Build Newton form symbolically
        polynomial = self.coefficients[0]
        product = 1
        
        for i in range(1, len(self.coefficients)):
            product *= (x - self.x[i-1])
            polynomial += self.coefficients[i] * product
            
        # Expand to standard form
        expanded_poly = sp.expand(polynomial)
        
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
        if self.coefficients is None:
            self.compute_divided_differences()
            
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
        ax.set_title('Interpolación de Newton')
        ax.legend()
        
        return fig
    
    def display_divided_differences_table(self):
        """
        Display the divided differences table in a formatted way.
        
        Returns:
        --------
        pandas DataFrame
            Table of divided differences
        """
        if self.divided_differences is None:
            self.compute_divided_differences()
            
        n = len(self.x)
        
        # Create a DataFrame for better display
        columns = ['y'] + [f'Δ^{i}' for i in range(1, n)]
        index = [f'x_{i} = {self.x[i]}' for i in range(n)]
        
        # Only include the relevant part of the divided differences table
        table_data = np.zeros((n, n))
        for i in range(n):
            table_data[i, 0:n-i] = self.divided_differences[i, 0:n-i]
            
        df = pd.DataFrame(table_data, index=index, columns=columns)
        
        return df
    
    def get_error_bounds(self, x_eval):
        """
        Estimate error bounds for the interpolation at given points.
        
        Note: This assumes the function being interpolated has continuous derivatives
        up to the n-th order, where n is the degree of the polynomial.
        
        Parameters:
        -----------
        x_eval : float or array-like
            Points at which to estimate error bounds
        
        Returns:
        --------
        float or numpy.ndarray
            Estimated upper bounds of the error
        """
        if self.coefficients is None:
            self.compute_divided_differences()
            
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
                
            # Estimate error using the last divided difference as approximation
            # of the n-th derivative divided by n!
            error_bounds[i] = abs(product * self.divided_differences[0, n-1])
            
        return error_bounds[0] if scalar_input else error_bounds