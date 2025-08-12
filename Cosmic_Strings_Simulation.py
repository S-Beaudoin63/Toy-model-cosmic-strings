#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 11:05:15 2025

@author: simonbeaudoin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress
from typing import Tuple, List, Any
from dataclasses import dataclass, field



@dataclass
class SimulationParameters:
    """Configuration parameters of the simulation"""
    grid_size: int = 100
    spatial_resolution: float = 1.0
    time_step: float = 0.05
    total_steps: int = 500
    animation_interval: int = 4
    vacuum_expectation: float = 1.0
    self_interaction_strength: float = 1.0
    damping_coefficient: float = 0.0
    
    
    
@dataclass
class TimeEvolutionData:
    """Stores time evolution data for easy access"""
    times: List[float] = field(default_factory=list)
    total_energy: List[float] = field(default_factory=list)
    kinetic_energy: List[float] = field(default_factory=list)
    gradient_energy: List[float] = field(default_factory=list)
    potential_energy: List[float] = field(default_factory=list)
    axion_energy: List[float] = field(default_factory=list)
    string_length: List[float] = field(default_factory=list)
    total_winding: List[float] = field(default_factory=list)
    mean_field_magnitude: List[float] = field(default_factory=list)



class AxionFieldSimulation:
    """
    This is the main class for axion field dynamics simulation.
    Handles the numerical integration of the Klein-Gordon equation
    for a complex scalar field with Mexican hat potential, along with
    visualization and analysis of the dynamics
    """

    def __init__(self, params: SimulationParameters):
        """Initializes the simulation with given parameters"""
        self.params = params
        self.time_data = TimeEvolutionData()
        self._initialize_fields()       # Initializes field arrays
        self._setup_figures()           # Setup visualization
        
    def _initialize_fields(self) -> None:
        """Initializes the complex field and its time derivatives"""
        N = self.params.grid_size
        self.phi_real = np.random.normal(0, 0.05, (N, N))
        self.phi_imag = np.random.normal(0, 0.05, (N, N))
        self.phi_real_dot = np.zeros((N, N))
        self.phi_imag_dot = np.zeros((N, N))
        
        
    def _setup_figures(self) -> None:
        """Matplotlib figures and axes"""
        # Figure 1: Phase maps, energies, and power spectra
        self.fig1 = plt.figure(figsize=(16, 12))
        gs1 = self.fig1.add_gridspec(2, 3, hspace=0.2, wspace=0.7)
        self.ax_phase = self.fig1.add_subplot(gs1[0, 0])            # Creates axes for first figure
        self.ax_energy = self.fig1.add_subplot(gs1[0, 1])
        self.ax_axion_energy = self.fig1.add_subplot(gs1[0, 2])
        self.ax_spectrum = self.fig1.add_subplot(gs1[1, 0])
        self.ax_radial = self.fig1.add_subplot(gs1[1, 1])
        self.ax_axion_radial = self.fig1.add_subplot(gs1[1, 2])
        # Figure 2: Velocity field and time evolution plots
        self.fig2 = plt.figure(figsize=(16, 12))
        gs2 = self.fig2.add_gridspec(2, 3, hspace=0.2, wspace=0.7)
        self.ax_velocity = self.fig2.add_subplot(gs2[0, 0])         # Creates axes for second figure
        self.ax_energy_time = self.fig2.add_subplot(gs2[0, 1])
        self.ax_string_time = self.fig2.add_subplot(gs2[1, 0])
        self.ax_scaling = self.fig2.add_subplot(gs2[1, 1])
        self._setup_plot_elements()
        
    def _setup_plot_elements(self) -> None:
        """Purely plots setup"""
        N = self.params.grid_size
        fa = self.params.vacuum_expectation
        # Phase plot with string overlay
        divider1 = make_axes_locatable(self.ax_phase)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        self.phase_image = self.ax_phase.imshow(np.zeros((N, N)), cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
        self.ax_phase.set_title("Phase with String Overlay")
        self.ax_phase.axis('off')
        self.fig1.colorbar(self.phase_image, cax=cax1, label=r"$\theta \in [-\pi, \pi]$")
        # String markers
        self.pos_scatter = self.ax_phase.plot([], [], 'o', markersize=8, label='+1 winding', color='yellow')[0]
        self.neg_scatter = self.ax_phase.plot([], [], 'o', markersize=8, label='-1 winding', color='green')[0]
        self.ax_phase.legend(loc='upper right', fontsize=8)
        self._setup_energy_plots(fa)            # Energy density plots
        self._setup_spectrum_plots(N)           # Power spectrum plots
        self._setup_time_evolution_plots()      # Time evolution plots
        # Figure titles
        self.fig1.suptitle("Axion Field Dynamics - Spatial Analysis", fontsize=16)
        self.fig2.suptitle("Axion Field Dynamics - Temporal Analysis", fontsize=16)
        
    def _setup_energy_plots(self, fa: float) -> None:
        """Setup energy density visualization plots"""
        N = self.params.grid_size
        # Total energy density
        divider2 = make_axes_locatable(self.ax_energy)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        self.energy_img = self.ax_energy.imshow(np.zeros((N, N)), cmap='inferno', origin='lower', vmin=0, vmax=fa)
        self.ax_energy.set_title("Total Energy Density")
        self.ax_energy.axis('off')
        self.fig1.colorbar(self.energy_img, cax=cax2, label="Energy Density")
        # Axion energy density
        divider3 = make_axes_locatable(self.ax_axion_energy)
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        self.axion_energy_img = self.ax_axion_energy.imshow(np.zeros((N, N)), cmap='inferno', origin='lower', vmin=0, vmax=fa)
        self.ax_axion_energy.set_title("Axion Energy Density")
        self.ax_axion_energy.axis('off')
        self.fig1.colorbar(self.axion_energy_img, cax=cax3, label="Energy Density")
        
    def _setup_spectrum_plots(self, N: int) -> None:
        """Setup power spectrum visualization plots"""
        # Fourier power spectrum
        divider4 = make_axes_locatable(self.ax_spectrum)
        cax4 = divider4.append_axes("right", size="5%", pad=0.05)
        self.spectrum_img = self.ax_spectrum.imshow(np.zeros((N, N)), cmap='magma', origin='lower', extent=[-N/2, N/2, -N/2, N/2])
        self.ax_spectrum.set_title(r"Fourier Power Spectrum $|\tilde{\phi}|^2$")
        self.ax_spectrum.set_xlabel(r"$k_x$")
        self.ax_spectrum.set_ylabel(r"$k_y$")
        self.fig1.colorbar(self.spectrum_img, cax=cax4, label=r"$\log_{10}(|\tilde{\phi}|^2)$")
        # Radial spectrum plots
        self.spectrum_line, = self.ax_radial.plot([], [], lw=2, color='blue', alpha=0.8)
        self.spectral_index_txt = self.ax_radial.text(0.98, 0.85, '', transform=self.ax_radial.transAxes, fontsize=12,
                                                      verticalalignment='top', horizontalalignment='right',
                                                      bbox=dict(facecolor='white', alpha=0.7))
        self.ax_radial.set_title(r"Radial Power Spectrum $P(k)$")
        self.ax_radial.set_xlabel(r"$\log_{10}(k)$")
        self.ax_radial.set_ylabel(r"$\log_{10}(P(k))$")
        self.ax_radial.set_xlim(np.log10(1), np.log10(N//2))
        self.ax_radial.set_ylim(-10, 10)
        # Axion spectrum
        self.axion_spectrum_line, = self.ax_axion_radial.plot([], [], lw=2, color='red', alpha=0.8)
        self.axion_spectral_index_txt = self.ax_axion_radial.text(0.98, 0.85, '', transform=self.ax_axion_radial.transAxes, fontsize=12,
                                                                  verticalalignment='top', horizontalalignment='right',
                                                                  bbox=dict(facecolor='white', alpha=0.7))
        self.ax_axion_radial.set_title(r"Axion Power Spectrum $P_\theta(k)$")
        self.ax_axion_radial.set_xlabel(r"$\log_{10}(k)$")
        self.ax_axion_radial.set_ylabel(r"$\log_{10}(P_{\theta}(k))$")
        self.ax_axion_radial.set_xlim(np.log10(1), np.log10(N//2))
        self.ax_axion_radial.set_ylim(-10, 10)
        
    def _setup_time_evolution_plots(self) -> None:
        """Setup time evolution visualization plots"""
        # Velocity field
        self.ax_velocity.set_title("Phase Velocity Field")
        self.ax_velocity.set_xlim(0, self.params.grid_size)
        self.ax_velocity.set_ylim(0, self.params.grid_size)
        
        # Energy evolution
        self.energy_lines = {
            'total': self.ax_energy_time.plot([], [], 'black', lw=2, alpha=0.8, label='Total')[0],
            'kinetic': self.ax_energy_time.plot([], [], 'red', lw=1, alpha=0.8, label='Kinetic')[0],
            'gradient': self.ax_energy_time.plot([], [], 'green', lw=1, alpha=0.8, label='Gradient')[0],
            'potential': self.ax_energy_time.plot([], [], 'blue', lw=1, alpha=0.8, label='Potential')[0],
        }
        self.ax_energy_time.set_title("Energy Evolution")
        self.ax_energy_time.set_xlabel("Time")
        self.ax_energy_time.set_ylabel("Energy Density")
        self.ax_energy_time.legend(fontsize=8)
        self.ax_energy_time.set_yscale('log')
        
        # String evolution
        self.string_length_line, = self.ax_string_time.plot([], [], color='red', alpha=0.8)
        self.ax_string_time.set_title("String Length Density")
        self.ax_string_time.set_xlabel("Time")
        self.ax_string_time.set_ylabel("String Length/Area")
        
        # Scaling analysis
        self.scaling_line, = self.ax_scaling.plot([], [], color='blue', alpha=0.8)
        self.ax_scaling.set_title("String Network Scaling")
        self.ax_scaling.set_xlabel("log(Time)")
        self.ax_scaling.set_ylabel("log(String Length)")
        
        
    def compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Computes discrete Laplacian using finite differences with periodic boundary conditions
        Args: field: 2D array scalar field
        Returns: 2D array Laplacian
        """
        dx = self.params.spatial_resolution
        return (-4 * field +
                np.roll(field, +1, axis=0) + np.roll(field, -1, axis=0) +
                np.roll(field, +1, axis=1) + np.roll(field, -1, axis=1)
                ) / dx**2
    
    def compute_gradient_squared(self, field: np.ndarray) -> np.ndarray:
        """
        Computes squared gradient magnitude using central differences
        Args: field: 2D array scalar field
        Returns: 2D array gradient squared magnitude
        """
        dx = self.params.spatial_resolution
        df_dx = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2 * dx)
        df_dy = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2 * dx)
        return df_dx**2 + df_dy**2
    
    def compute_gradient(self, field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes gradient vector field using central differences
        Args: field: 2D array scalar field
        Returns: (gradient_x, gradient_y)
        """
        dx = self.params.spatial_resolution
        df_dx = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2 * dx)
        df_dy = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2 * dx)
        return df_dx, df_dy
    
    def compute_energy_components(self, phi_r: np.ndarray, phi_i: np.ndarray, 
                                  phi_r_dot: np.ndarray, phi_i_dot: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes energy density components separately
        Args: phi_r, phi_i: Real and imaginary parts of the field
              phi_r_dot, phi_i_dot: Time derivatives of field components
        Returns: (kinetic, gradient, potential) energy densities
        """
        fa = self.params.vacuum_expectation
        lambda_ = self.params.self_interaction_strength
        kinetic = 0.5 * (phi_r_dot**2 + phi_i_dot**2)
        gradient = 0.5 * (self.compute_gradient_squared(phi_r) + 
                         self.compute_gradient_squared(phi_i))
        r2 = phi_r**2 + phi_i**2
        potential = lambda_ * (r2 - fa**2 / 2)**2
        return kinetic, gradient, potential
    
    def compute_axion_energy_density(self, phi_r: np.ndarray, phi_i: np.ndarray, 
                                     phi_r_dot: np.ndarray, phi_i_dot: np.ndarray) -> np.ndarray:
        """
        Computes axion energy density
        Args: phi_r, phi_i: Real and imaginary parts of the field
              phi_r_dot, phi_i_dot: Time derivatives of field components 
        Returns: 2D array axion energy density
        """
        fa = self.params.vacuum_expectation
        theta = np.arctan2(phi_i, phi_r)
        theta_dot = (phi_r * phi_i_dot - phi_i * phi_r_dot) / (phi_r**2 + phi_i**2 + 1e-12)
        grad_sq_theta = self.compute_gradient_squared(theta)
        kinetic = 0.5 * fa**2 * theta_dot**2
        gradient = 0.5 * fa**2 * grad_sq_theta
        return kinetic + gradient
    
    def compute_phase_velocity_field(self, phi_r: np.ndarray, phi_i: np.ndarray, 
                                     phi_r_dot: np.ndarray, phi_i_dot: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes velocity field
        Args: phi_r, phi_i: Real and imaginary parts of the field
              phi_r_dot, phi_i_dot: Time derivatives of field components 
        Returns: (v_x, v_y, theta_dot)
        """
        theta = np.arctan2(phi_i, phi_r)
        theta_dot = (phi_r * phi_i_dot - phi_i * phi_r_dot) / (phi_r**2 + phi_i**2 + 1e-12)
        grad_theta_x, grad_theta_y = self.compute_gradient(theta)
        v_x = -grad_theta_y        # Velocity field perpendicular to phase gradient
        v_y = grad_theta_x
        return v_x, v_y, theta_dot
    
    
    
    @staticmethod
    def unwrap_angle(dtheta: np.ndarray) -> np.ndarray:
        """
        Unwraps angle differences into [-π, π] range
        Args: dtheta: Array of angle differences
        Returns: Unwrapped angles
        """
        return (dtheta + np.pi) % (2 * np.pi) - np.pi
    
    def detect_axion_strings(self, phi_r: np.ndarray, phi_i: np.ndarray) -> np.ndarray:
        """
        Detects axion strings by computing winding number around plaquettes
        Args: phi_r, phi_i: Real and imaginary parts of the field
        Returns: 2D array of winding numbers (float values)
        """
        theta = np.arctan2(phi_i, phi_r)
        dtheta1 = self.unwrap_angle(np.roll(theta, -1, axis=1) - theta)
        dtheta2 = self.unwrap_angle(np.roll(theta, (-1, -1), axis=(0, 1)) - np.roll(theta, -1, axis=1))
        dtheta3 = self.unwrap_angle(np.roll(theta, -1, axis=0) - np.roll(theta, (-1, -1), axis=(0, 1)))
        dtheta4 = self.unwrap_angle(theta - np.roll(theta, -1, axis=0))
        total_winding = (dtheta1[:-1, :-1] + dtheta2[:-1, :-1] + dtheta3[:-1, :-1] + dtheta4[:-1, :-1])
        return total_winding / (2 * np.pi)
    
    def compute_string_length_density(self, phi_r: np.ndarray, phi_i: np.ndarray) -> float:
        """
        Computes total string length per unit area
        Args: phi_r, phi_i: Real and imaginary parts of the field
        Returns: String length density
        """
        winding_float = self.detect_axion_strings(phi_r, phi_i)
        dx = self.params.spatial_resolution
        N = self.params.grid_size
        string_length = np.sum(np.abs(winding_float)) * dx
        return string_length / (N * N * dx**2)
    
    def compute_power_spectrum(self, phi_r: np.ndarray, phi_i: np.ndarray) -> np.ndarray:
        """
        Computes Fourier power spectrum of the complex field
        Args: phi_r, phi_i: Real and imaginary parts of the field
        Returns: 2D power spectrum array
        """
        phi_complex = phi_r + 1j * phi_i
        fft_phi = np.fft.fftshift(np.fft.fft2(phi_complex))
        return np.abs(fft_phi)**2 / phi_complex.size
    
    def compute_axion_power_spectrum(self, phi_r: np.ndarray, phi_i: np.ndarray) -> np.ndarray:
        """
        Computes power spectrum of the axion phase field
        Args: phi_r, phi_i: Real and imaginary parts of the field
        Returns: 2D power spectrum array of the phase
        """
        theta = np.arctan2(phi_i, phi_r)
        fft_theta = np.fft.fftshift(np.fft.fft2(theta))
        return np.abs(fft_theta)**2 / theta.size
    
    
    
    @staticmethod
    def compute_radial_profile(data: np.ndarray) -> np.ndarray:
        """
        Computes radially averaged profile of 2D data
        Args: data: 2D array to be radially averaged
        Returns: 1D radially averaged profile
        """
        N = data.shape[0]
        y, x = np.indices((N, N))
        center = (N // 2, N // 2)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        radial_mean = np.bincount(r.ravel(), weights=data.ravel()) / np.maximum(1, np.bincount(r.ravel()))
        return radial_mean
    
    def update_field(self) -> None:
        """
        Updates field using Klein-Gordon equation with possible damping
        """
        dt = self.params.time_step
        lambda_ = self.params.self_interaction_strength
        fa = self.params.vacuum_expectation
        damping = self.params.damping_coefficient
        r2 = self.phi_real**2 + self.phi_imag**2                    # Computes potential derivatives
        dV_real = 2 * lambda_ * (r2 - fa**2 / 2) * self.phi_real
        dV_imag = 2 * lambda_ * (r2 - fa**2 / 2) * self.phi_imag
        # Updates time derivatives (Klein-Gordon equation)
        self.phi_real_dot += dt * (self.compute_laplacian(self.phi_real) - dV_real - damping * self.phi_real_dot)
        self.phi_imag_dot += dt * (self.compute_laplacian(self.phi_imag) - dV_imag - damping * self.phi_imag_dot)
        # Updates field values
        self.phi_real += dt * self.phi_real_dot
        self.phi_imag += dt * self.phi_imag_dot
    
    def store_time_evolution_data(self, step: int) -> None:
        """
        Stores current field state data for time evolution analysis
        """
        dt = self.params.time_step
        kinetic, gradient, potential = self.compute_energy_components(self.phi_real, self.phi_imag, self.phi_real_dot, self.phi_imag_dot)
        axion_energy = self.compute_axion_energy_density(self.phi_real, self.phi_imag, self.phi_real_dot, self.phi_imag_dot)
        string_length = self.compute_string_length_density(self.phi_real, self.phi_imag)
        winding_map = self.detect_axion_strings(self.phi_real, self.phi_imag)
        self.time_data.times.append(step * dt)
        self.time_data.total_energy.append(np.mean(kinetic + gradient + potential))
        self.time_data.kinetic_energy.append(np.mean(kinetic))
        self.time_data.gradient_energy.append(np.mean(gradient))
        self.time_data.potential_energy.append(np.mean(potential))
        self.time_data.axion_energy.append(np.mean(axion_energy))
        self.time_data.string_length.append(string_length)
        self.time_data.total_winding.append(np.sum(np.abs(winding_map)))
        self.time_data.mean_field_magnitude.append(np.mean(np.sqrt(self.phi_real**2 + self.phi_imag**2)))
    
    def update_visualization(self, frame: int) -> Tuple[Any, ...]:
        """
        Updates all visualization elements for animation
        """
        current_step = frame * self.params.animation_interval
        for _ in range(self.params.animation_interval):
            self.update_field()
        self.store_time_evolution_data(current_step)
        self._update_phase_visualization(frame, current_step)
        self._update_energy_visualizations()
        self._update_velocity_field(frame)
        self._update_spectrum_visualizations()
        self._update_time_evolution_plots()
        return (self.phase_image, self.pos_scatter, self.neg_scatter, 
                self.energy_img, self.axion_energy_img, self.spectrum_img,
                self.spectrum_line, self.axion_spectrum_line)
    
    def _update_phase_visualization(self, frame: int, current_step: int) -> None:
        """Updates phase map and string overlays"""
        theta = np.arctan2(self.phi_imag, self.phi_real)
        self.phase_image.set_data(theta)
        winding_float = self.detect_axion_strings(self.phi_real, self.phi_imag)
        winding_rounded = np.round(winding_float).astype(int)
        pos_y, pos_x = np.where(winding_rounded == 1)
        neg_y, neg_x = np.where(winding_rounded == -1)
        self.pos_scatter.set_data(pos_x + 0.5, pos_y + 0.5)
        self.neg_scatter.set_data(neg_x + 0.5, neg_y + 0.5)
        net_winding = np.sum(winding_float)
        total_abs = np.sum(np.abs(winding_float))
        max_wind = np.max(np.abs(winding_float))
        self.ax_phase.set_title(f"Phase with String Overlay\n"
                                f"Step {current_step} | Net: {net_winding:.2f}, "
                                f"|w|: {total_abs:.2f}, Max |w|: {max_wind:.2f}")
    
    def _update_energy_visualizations(self) -> None:
        """Updates energy density visualizations"""
        kinetic, gradient, potential = self.compute_energy_components(self.phi_real, self.phi_imag, self.phi_real_dot, self.phi_imag_dot)
        total_energy = kinetic + gradient + potential
        self.energy_img.set_data(total_energy)
        axion_energy = self.compute_axion_energy_density(self.phi_real, self.phi_imag, self.phi_real_dot, self.phi_imag_dot)
        self.axion_energy_img.set_data(axion_energy)
    
    def _update_velocity_field(self, frame: int) -> None:
        """Updates velocity field visualization"""
        if frame % 2 == 0:  # Updates less frequently for performance
            v_x, v_y, _ = self.compute_phase_velocity_field(self.phi_real, self.phi_imag, self.phi_real_dot, self.phi_imag_dot)
            N = self.params.grid_size
            skip = 8
            X, Y = np.meshgrid(np.arange(0, N, skip), np.arange(0, N, skip))
            U = v_x[::skip, ::skip]
            V = v_y[::skip, ::skip]
            magnitude = np.sqrt(U**2 + V**2) * 0.5
            max_mag = np.percentile(magnitude, 95)
            clip_mag = np.maximum(magnitude, max_mag)
            U_normalized = U / (clip_mag + 1e-12)
            V_normalized = V / (clip_mag + 1e-12)
            self.ax_velocity.clear()
            self.ax_velocity.quiver(X, Y, U_normalized, V_normalized, scale=25, alpha=0.7)
            self.ax_velocity.set_title("Phase Velocity Field")
            self.ax_velocity.set_xlim(0, N)
            self.ax_velocity.set_ylim(0, N)
    
    def _update_spectrum_visualizations(self) -> None:
        """Updates power spectrum visualizations"""
        power = self.compute_power_spectrum(self.phi_real, self.phi_imag)
        self.spectrum_img.set_data(np.log10(power + 1e-12))
        radial_phi = self.compute_radial_profile(np.log10(power + 1e-12))
        axion_power = self.compute_axion_power_spectrum(self.phi_real, self.phi_imag)
        radial_theta = self.compute_radial_profile(np.log10(axion_power + 1e-12))
        k_vals = np.arange(len(radial_phi))
        log_k = np.log10(k_vals[1:] + 1e-12)
        log_radial_phi = radial_phi[1:]
        log_radial_theta = radial_theta[1:]
        self.spectrum_line.set_data(log_k, log_radial_phi)
        self.axion_spectrum_line.set_data(log_k, log_radial_theta)
        N = self.params.grid_size
        fit_mask = (k_vals[1:] > 4) & (k_vals[1:] < N//2)
        if np.any(fit_mask) and len(log_k[fit_mask]) > 3:
            slope, _, _, _, _ = linregress(log_k[fit_mask], log_radial_phi[fit_mask])
            self.spectral_index_txt.set_text(f'n = {-slope:.2f}')
            slope_axion, _, _, _, _ = linregress(log_k[fit_mask], log_radial_theta[fit_mask])
            self.axion_spectral_index_txt.set_text(f'n = {-slope_axion:.2f}')
    
    def _update_time_evolution_plots(self) -> None:
        """Updates time evolution plots"""
        if len(self.time_data.times) <= 1:
            return
        times = np.array(self.time_data.times)
        self.energy_lines['total'].set_data(times, self.time_data.total_energy)
        self.energy_lines['kinetic'].set_data(times, self.time_data.kinetic_energy)
        self.energy_lines['gradient'].set_data(times, self.time_data.gradient_energy)
        self.energy_lines['potential'].set_data(times, self.time_data.potential_energy)
        # Updates axis limits for energy plot
        if times[-1] > times[0]:
            self.ax_energy_time.set_xlim(times[0], times[-1])
        if len(times) > 5:
            y_min = min(
                min(self.time_data.kinetic_energy),
                min(self.time_data.gradient_energy),
                min(self.time_data.potential_energy)
            )
            y_max = max(self.time_data.total_energy)
            if y_max > y_min:
                self.ax_energy_time.set_ylim(y_min * 0.5, y_max * 2)
        # Updates string evolution plots (only for times > 2)
        if times[-1] > 2:
            mask = times > 2
            times_filtered = times[mask]
            if len(times_filtered) > 1:
                string_length_filtered = np.array(self.time_data.string_length)[mask]
                self.string_length_line.set_data(times_filtered, string_length_filtered)
                self.ax_string_time.set_xlim(times_filtered[0], times_filtered[-1])
                if max(string_length_filtered) > 0:
                    self.ax_string_time.set_ylim(0, max(string_length_filtered) * 1.1)
                if len(times_filtered) > 6:
                    log_times = np.log10(times_filtered[5:] + 1e-12)
                    log_strings = np.log10(string_length_filtered[5:] + 1e-12)
                    self.scaling_line.set_data(log_times, log_strings)
                    if len(log_times) > 0 and log_times[-1] > log_times[0]:
                        self.ax_scaling.set_xlim(log_times[0], log_times[-1])
                    if len(log_strings) > 0 and max(log_strings) > min(log_strings):
                        self.ax_scaling.set_ylim(min(log_strings) - 0.5, max(log_strings) + 0.5)
    
    
    def run_simulation(self) -> None:
        """
        Runs the complete simulation with real-time visualization
        """
        total_frames = self.params.total_steps // self.params.animation_interval
        self.ani1 = animation.FuncAnimation(
            self.fig1, self.update_visualization, frames=total_frames,
            interval=150, blit=False, repeat=False) 
        self.ani2 = animation.FuncAnimation(
            self.fig2, lambda frame: (), frames=total_frames,
            interval=150, blit=False, repeat=False)
        plt.show()
    


def main():
    params = SimulationParameters(grid_size=100,
                                  spatial_resolution=1.0,
                                  time_step=0.05,
                                  total_steps=500,
                                  animation_interval=4,
                                  vacuum_expectation=1.0,
                                  self_interaction_strength=1.0,
                                  damping_coefficient=0.0)
    simulation = AxionFieldSimulation(params)
    print("Starting axion field dynamics simulation...")
    print(f"Grid size: {params.grid_size}x{params.grid_size}")
    print(f"Total simulation time: {params.total_steps * params.time_step:.2f}")
    print(f"Animation frames: {params.total_steps // params.animation_interval}")
    simulation.run_simulation()
    
if __name__ == "__main__":
    main()
