import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

class FastFourierTransform:
    """
    A class to calculate and visualize Fourier analysis using FFT for efficient computation.
    """
    def __init__(self, function=None, x_data=None, y_data=None, num_points=1024, domain=(-10, 10)):
        """
        Initialize with either a function or discrete data points.
        
        Parameters:
        -----------
        function : callable, optional
            Function to analyze with FFT
        x_data : array-like, optional
            x-coordinates of data points (if providing discrete data)
        y_data : array-like, optional
            y-coordinates of data points (if providing discrete data)
        num_points : int, optional
            Number of points to sample (if providing a function)
        domain : tuple, optional
            Domain (x_min, x_max) to sample function
        """
        self.domain = domain
        self.num_points = num_points
        
        # Process input data - either from function or discrete points
        if function is not None:
            # Sample the function
            self.x = np.linspace(domain[0], domain[1], num_points)
            self.y = np.array([function(xi) for xi in self.x])
        elif x_data is not None and y_data is not None:
            # Use provided data points
            self.x = np.array(x_data)
            self.y = np.array(y_data)
            self.num_points = len(self.x)
            self.domain = (min(self.x), max(self.x))
        else:
            raise ValueError("Either function or x_data/y_data must be provided")
            
        # Store attributes for frequency analysis
        self.frequencies = None
        self.amplitudes = None
        self.phases = None
        self.fft_result = None
        
    def compute_fft(self):
        """
        Compute the Fast Fourier Transform of the data.
        
        Returns:
        --------
        tuple: (frequencies, amplitudes, phases)
        """
        # Compute FFT
        fft_complex = np.fft.fft(self.y)
        
        # Normalize by number of points
        fft_complex = fft_complex / self.num_points
        
        # Compute frequency values
        sample_spacing = (self.domain[1] - self.domain[0]) / (self.num_points - 1)
        self.frequencies = np.fft.fftfreq(self.num_points, d=sample_spacing)
        
        # Calculate amplitudes and phases (only positive frequencies)
        self.amplitudes = np.abs(fft_complex)
        self.phases = np.angle(fft_complex)
        
        # Store full FFT result
        self.fft_result = fft_complex
        
        return self.frequencies, self.amplitudes, self.phases
    
    def get_dominant_frequencies(self, n=10):
        """
        Get the n most dominant frequencies in the signal.
        
        Parameters:
        -----------
        n : int
            Number of dominant frequencies to return
        
        Returns:
        --------
        DataFrame with frequency, amplitude, phase information
        """
        if self.frequencies is None:
            self.compute_fft()
            
        # Consider only positive frequencies (up to Nyquist frequency)
        pos_idx = np.arange(1, self.num_points // 2)
        
        # Get indices of top n amplitudes
        top_indices = np.argsort(self.amplitudes[pos_idx])[-n:][::-1]
        top_indices = pos_idx[top_indices]
        
        # Create DataFrame
        data = {
            'Frequency': self.frequencies[top_indices],
            'Amplitude': self.amplitudes[top_indices],
            'Phase (rad)': self.phases[top_indices]
        }
        
        return pd.DataFrame(data)
    
    def plot_spectrum(self, log_scale=False):
        """
        Plot the frequency spectrum.
        
        Parameters:
        -----------
        log_scale : bool
            Whether to use logarithmic scale for amplitude
        
        Returns:
        --------
        matplotlib figure
        """
        if self.frequencies is None:
            self.compute_fft()
        
        # Plot only up to Nyquist frequency (positive frequencies)
        freq_pos = self.frequencies[:self.num_points//2]
        amp_pos = self.amplitudes[:self.num_points//2]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot frequency spectrum
        if log_scale:
            # Add small value to avoid log(0)
            ax.semilogy(freq_pos, amp_pos + 1e-10)
            ax.set_ylabel('Amplitud (escala logarítmica)')
        else:
            ax.plot(freq_pos, amp_pos)
            ax.set_ylabel('Amplitud')
            
        ax.set_xlabel('Frecuencia')
        ax.set_title('Espectro de Frecuencias')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def reconstruct_signal(self, num_components=None):
        """
        Reconstruct the signal using inverse FFT, optionally with limited components.
        
        Parameters:
        -----------
        num_components : int or None
            Number of frequency components to use (None = all components)
            
        Returns:
        --------
        reconstructed signal values
        """
        if self.fft_result is None:
            self.compute_fft()
            
        # If using all components, just use the original FFT result
        if num_components is None or num_components >= self.num_points//2:
            reconstructed_fft = self.fft_result
        else:
            # Start with zeros
            reconstructed_fft = np.zeros_like(self.fft_result, dtype=complex)
            
            # Keep DC component (frequency=0)
            reconstructed_fft[0] = self.fft_result[0]
            
            # Find indices of dominant frequencies (excluding DC)
            pos_idx = np.arange(1, self.num_points // 2)
            top_indices = np.argsort(self.amplitudes[pos_idx])[-num_components:]
            top_indices = pos_idx[top_indices]
            
            # Set these components in both positive and negative frequencies (complex conjugates)
            reconstructed_fft[top_indices] = self.fft_result[top_indices]
            reconstructed_fft[self.num_points - top_indices] = self.fft_result[self.num_points - top_indices]
            
        # Perform inverse FFT and scale
        reconstructed_signal = np.real(np.fft.ifft(reconstructed_fft * self.num_points))
        
        return reconstructed_signal
    
    def plot_reconstructions(self, component_list=None):
        """
        Plot original signal and reconstructions with different numbers of components.
        
        Parameters:
        -----------
        component_list : list
            List of number of components to use for reconstructions
            
        Returns:
        --------
        matplotlib figure
        """
        if component_list is None:
            component_list = [1, 5, 10, 20]
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot original signal
        ax.plot(self.x, self.y, 'k', label='Señal original', linewidth=2)
        
        # Plot reconstructions
        for n in component_list:
            reconstructed = self.reconstruct_signal(n)
            ax.plot(self.x, reconstructed, label=f'Reconstrucción ({n} componentes)')
            
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Señal Original y Reconstrucciones con FFT')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
    
    def compare_with_fourier_series(self, fourier_series, component_list=None):
        """
        Compare FFT reconstruction with traditional Fourier series.
        
        Parameters:
        -----------
        fourier_series : FourierSeries object
            Traditional Fourier series to compare with
        component_list : list
            List of number of components to use for comparison
            
        Returns:
        --------
        matplotlib figure
        """
        if component_list is None:
            component_list = [5, 10]
            
        fig, axs = plt.subplots(len(component_list), 1, figsize=(12, 4*len(component_list)))
        if len(component_list) == 1:
            axs = [axs]
            
        for i, n in enumerate(component_list):
            # Get reconstructions
            fft_recon = self.reconstruct_signal(n)
            
            # Plot original and FFT reconstruction
            axs[i].plot(self.x, self.y, 'k', label='Original', linewidth=2)
            axs[i].plot(self.x, fft_recon, 'r--', label=f'FFT ({n} componentes)')
            
            # Plot Fourier series if available
            if fourier_series is not None and fourier_series.fourier_terms >= n:
                fourier_approx = fourier_series.get_approximation_function(terms=n)
                y_fourier = [fourier_approx(xi) for xi in self.x]
                axs[i].plot(self.x, y_fourier, 'g-.', label=f'Serie Fourier ({n} términos)')
                
            axs[i].set_title(f'Comparación con {n} componentes')
            axs[i].grid(True, alpha=0.3)
            axs[i].legend()
            
        fig.tight_layout()
        return fig
    
    def calculate_error(self, component_list=None):
        """
        Calculate reconstruction error for different numbers of components.
        
        Parameters:
        -----------
        component_list : list
            List of number of components to use
            
        Returns:
        --------
        DataFrame with errors
        """
        if component_list is None:
            component_list = [1, 2, 5, 10, 20, 50, 100]
            
        errors = []
        
        for n in component_list:
            reconstructed = self.reconstruct_signal(n)
            mse = np.mean((self.y - reconstructed)**2)
            max_error = np.max(np.abs(self.y - reconstructed))
            errors.append((n, mse, max_error))
            
        df = pd.DataFrame(errors, columns=['Componentes', 'Error cuadrático medio', 'Error máximo'])
        return df
    def display_coefficients_table(self):
        """
        Display a table of Fourier coefficients.
        """
        data = {
            'n': list(range(len(self.a_coeffs))),
            'a_n': self.a_coeffs,
            'b_n': [0] + self.b_coeffs  # Add 0 for b_0 which doesn't exist
        }
        
        df = pd.DataFrame(data)
        display(df)
        return df
    
    def plot_approximations(self, x_range=None, num_points=1000, term_list=None, ylim=None):
        """
        Plot the original function and its Fourier approximations.
        
        Parameters:
        -----------
        x_range : tuple
            (min_x, max_x) range for plotting (default is the function interval)
        num_points : int
            Number of points to evaluate
        term_list : list
            List of approximation orders to plot
        ylim : tuple
            (min_y, max_y) for the plot
        """
        if x_range is None:
            # Extend the interval a bit for better visualization
            a, b = self.interval
            margin = (b - a) * 0.1
            x_range = (a - margin, b + margin)
            
        if term_list is None:
            term_list = [1, 3, 5, min(10, self.fourier_terms)]
            
        x = np.linspace(x_range[0], x_range[1], num_points)
        
        # Evaluate the original function
        try:
            y_actual = np.array([self.f_callable(xi) for xi in x])
        except:
            y_actual = np.zeros_like(x)
            for i, xi in enumerate(x):
                try:
                    y_actual[i] = float(self.f_sym.subs(self.x_sym, xi))
                except:
                    y_actual[i] = np.nan
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot the original function
        plt.plot(x, y_actual, 'k-', label='f(x) original', linewidth=2)
        
        # Plot each approximation
        for terms in term_list:
            if terms <= self.fourier_terms:
                approx_func = self.get_approximation_function(terms)
                y_approx = np.array([approx_func(xi) for xi in x])
                plt.plot(x, y_approx, label=f'Fourier ({terms} términos)')
        
        # Mark the interval boundaries
        plt.axvline(self.interval[0], color='gray', linestyle='--', alpha=0.7)
        plt.axvline(self.interval[1], color='gray', linestyle='--', alpha=0.7)
        
        # Add grid, legend, labels
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title(f"Serie de Fourier para f(x)")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        
        if ylim:
            plt.ylim(ylim)
            
        return plt.gcf()
    
    def calculate_error(self, x_points, term_list=None):
        """
        Calculate error between approximation and actual function at specified points.
        
        Parameters:
        -----------
        x_points : list or array
            Points at which to calculate the error
        term_list : list
            List of term counts to evaluate
        """
        if term_list is None:
            term_list = [1, 3, 5, min(10, self.fourier_terms)]
            
        errors = {}
        
        # Calculate actual function values
        actual_values = {}
        for x in x_points:
            try:
                actual_values[x] = self.f_callable(x)
            except:
                try:
                    actual_values[x] = float(self.f_sym.subs(self.x_sym, x))
                except:
                    actual_values[x] = np.nan
        
        # Calculate errors for each term count
        for terms in term_list:
            if terms <= self.fourier_terms:
                approx_func = self.get_approximation_function(terms)
                errors[terms] = []
                
                for x in x_points:
                    if np.isnan(actual_values[x]):
                        errors[terms].append(np.nan)
                    else:
                        approx = approx_func(x)
                        error = abs(actual_values[x] - approx)
                        errors[terms].append(error)
        
        # Create DataFrame
        df = pd.DataFrame({'x': x_points})
        for terms in term_list:
            if terms in errors:
                df[f'Error ({terms} términos)'] = errors[terms]
                
        return df
    
    def visualize_convergence(self, x_point, max_terms=20):
        """
        Visualize how the Fourier approximation converges at a specific point.
        
        Parameters:
        -----------
        x_point : float
            Point at which to check convergence
        max_terms : int
            Maximum number of terms to include
        """
        # Calculate actual value
        try:
            actual = self.f_callable(x_point)
        except:
            actual = float(self.f_sym.subs(self.x_sym, x_point))
            
        # Calculate approximations with increasing terms
        terms = list(range(1, max_terms + 1))
        approximations = []
        errors = []
        
        for n in terms:
            if n <= self.fourier_terms:
                approx_func = self.get_approximation_function(n)
                approx = approx_func(x_point)
                approximations.append(approx)
                errors.append(abs(actual - approx))
            else:
                # Calculate more coefficients if needed
                old_terms = self.fourier_terms
                self.calculate_coefficients(n)
                approx_func = self.get_approximation_function(n)
                approx = approx_func(x_point)
                approximations.append(approx)
                errors.append(abs(actual - approx))
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot approximation values
        ax1.plot(terms, approximations, 'bo-')
        ax1.axhline(actual, color='r', linestyle='--', label=f'Valor exacto: {actual:.6f}')
        ax1.set_title(f'Convergencia en x = {x_point}')
        ax1.set_xlabel('Número de términos')
        ax1.set_ylabel('Valor de la aproximación')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot error values
        ax2.semilogy(terms, errors, 'ro-')
        ax2.set_title(f'Error de aproximación en x = {x_point}')
        ax2.set_xlabel('Número de términos')
        ax2.set_ylabel('Error (escala logarítmica)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig