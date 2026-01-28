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
from scipy.signal import find_peaks
from scipy.ndimage import label

N = 100                # Grid size
dx = 1.0               # Spatial resolution
dt = 0.05              # Time step
total_steps = 1000     # Total number of steps
anim_interval = 4      # Animation update interval
fa = 1.0               # Vacuum expectation value
lam = 1.0              # Self-interaction strength
gamma = 0.0            # Damping coefficient


class AxionSim:
    """
    Main simulation class for axion field dynamics.
    Handles field evolution, diagnostics, and visualization.
    """
    
    def __init__(self):
        """Initialize fields and setup visualization"""
        # Initialize field arrays
        self.phi_r = np.random.normal(0, 0.05, (N, N))
        self.phi_i = np.random.normal(0, 0.05, (N, N))
        self.phi_r_dot = np.zeros((N, N))
        self.phi_i_dot = np.zeros((N, N))
        self.phi_r_prev = self.phi_r.copy()
        self.phi_i_prev = self.phi_i.copy()
        
        # Initialize time series storage
        self.t_list = []
        self.E_tot = []
        self.E_kin = []
        self.E_grad = []
        self.E_pot = []
        self.E_ax = []
        self.L_string = []
        self.winding_tot = []
        self.phi_mean = []
        self.k_peak = []
        self.flux = []
        self.v_string = []
        self.d_string = []
        self.xi = []
        self.n_segments = []
        self.vort_rms = []
        
        self.setup_plots()
    
    def setup_plots(self):
        """Setup matplotlib figures and axes"""
        # Fig 1
        self.fig1 = plt.figure(figsize=(18, 14))
        gs1 = self.fig1.add_gridspec(3, 3, hspace=0.3, wspace=0.4)
        
        self.ax1 = self.fig1.add_subplot(gs1[0, 0])
        self.ax2 = self.fig1.add_subplot(gs1[0, 1])
        self.ax3 = self.fig1.add_subplot(gs1[0, 2])
        self.ax4 = self.fig1.add_subplot(gs1[1, 0])
        self.ax5 = self.fig1.add_subplot(gs1[1, 1])
        self.ax6 = self.fig1.add_subplot(gs1[1, 2])
        self.ax7 = self.fig1.add_subplot(gs1[2, 0])
        self.ax8 = self.fig1.add_subplot(gs1[2, 1])
        self.ax9 = self.fig1.add_subplot(gs1[2, 2])
        
        # Fig 2
        self.fig2 = plt.figure(figsize=(18, 14))
        gs2 = self.fig2.add_gridspec(3, 3, hspace=0.3, wspace=0.4)
        
        self.ax10 = self.fig2.add_subplot(gs2[0, 0])
        self.ax11 = self.fig2.add_subplot(gs2[0, 1])
        self.ax12 = self.fig2.add_subplot(gs2[1, 0])
        self.ax13 = self.fig2.add_subplot(gs2[1, 1])
        self.ax14 = self.fig2.add_subplot(gs2[0, 2])
        self.ax15 = self.fig2.add_subplot(gs2[1, 2])
        self.ax16 = self.fig2.add_subplot(gs2[2, 0])
        self.ax17 = self.fig2.add_subplot(gs2[2, 1])
        self.ax18 = self.fig2.add_subplot(gs2[2, 2])
        
        # Phase plot
        div1 = make_axes_locatable(self.ax1)
        cax1 = div1.append_axes("right", size="5%", pad=0.05)
        self.im1 = self.ax1.imshow(np.zeros((N, N)), cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
        self.ax1.set_title("Phase with String Overlay")
        self.ax1.axis('off')
        self.fig1.colorbar(self.im1, cax=cax1, label=r"$\theta \in [-\pi, \pi]$")
        self.scat_p, = self.ax1.plot([], [], 'o', ms=8, label='+1 winding', color='yellow')
        self.scat_n, = self.ax1.plot([], [], 'o', ms=8, label='-1 winding', color='green')
        self.ax1.legend(loc='upper right', fontsize=8)
        
        # Energy density
        div2 = make_axes_locatable(self.ax2)
        cax2 = div2.append_axes("right", size="5%", pad=0.05)
        self.im2 = self.ax2.imshow(np.zeros((N, N)), cmap='inferno', origin='lower', vmin=0, vmax=fa)
        self.ax2.set_title("Total Energy Density")
        self.ax2.axis('off')
        self.fig1.colorbar(self.im2, cax=cax2, label="Energy Density")
        
        # Axion energy
        div3 = make_axes_locatable(self.ax3)
        cax3 = div3.append_axes("right", size="5%", pad=0.05)
        self.im3 = self.ax3.imshow(np.zeros((N, N)), cmap='inferno', origin='lower', vmin=0, vmax=fa)
        self.ax3.set_title("Axion Energy Density")
        self.ax3.axis('off')
        self.fig1.colorbar(self.im3, cax=cax3, label="Energy Density")
        
        # Power spectrum
        div4 = make_axes_locatable(self.ax4)
        cax4 = div4.append_axes("right", size="5%", pad=0.05)
        self.im4 = self.ax4.imshow(np.zeros((N, N)), cmap='magma', origin='lower', extent=[-N/2, N/2, -N/2, N/2])
        self.ax4.set_title(r"Fourier Power Spectrum $|\tilde{\phi}|^2$")
        self.ax4.set_xlabel(r"$k_x$")
        self.ax4.set_ylabel(r"$k_y$")
        self.fig1.colorbar(self.im4, cax=cax4, label=r"$\log_{10}(|\tilde{\phi}|^2)$")
        
        # Radial spectrum
        self.line5, = self.ax5.plot([], [], lw=2, color='blue', alpha=0.8)
        self.peak5, = self.ax5.plot([], [], 'ro', ms=8, label='Peaks')
        self.txt5 = self.ax5.text(0.98, 0.85, '', transform=self.ax5.transAxes, fontsize=10,
                                  va='top', ha='right', bbox=dict(facecolor='white', alpha=0.7))
        self.ax5.set_title(r"Radial Power Spectrum $P(k)$")
        self.ax5.set_xlabel(r"$\log_{10}(k)$")
        self.ax5.set_ylabel(r"$\log_{10}(P(k))$")
        self.ax5.set_xlim(np.log10(1), np.log10(N//2))
        self.ax5.set_ylim(-10, 10)
        self.ax5.legend(fontsize=8)
        
        # Axion spectrum
        self.line6, = self.ax6.plot([], [], lw=2, color='red', alpha=0.8)
        self.txt6 = self.ax6.text(0.98, 0.85, '', transform=self.ax6.transAxes, fontsize=10,
                                  va='top', ha='right', bbox=dict(facecolor='white', alpha=0.7))
        self.ax6.set_title(r"Axion Power Spectrum $P_\theta(k)$")
        self.ax6.set_xlabel(r"$\log_{10}(k)$")
        self.ax6.set_ylabel(r"$\log_{10}(P_{\theta}(k))$")
        self.ax6.set_xlim(np.log10(1), np.log10(N//2))
        self.ax6.set_ylim(-10, 10)
        
        # Correlation
        self.line7, = self.ax7.plot([], [], lw=2, color='purple', alpha=0.8)
        self.ax7.axhline(y=0, color='k', ls='--', alpha=0.3)
        self.ax7.set_title("Two-Point Correlation Function")
        self.ax7.set_xlabel("Distance")
        self.ax7.set_ylabel(r"$C(r)$")
        self.ax7.set_xlim(0, N // 2)
        self.ax7.set_ylim(-0.5, 1)
        
        # Histogram
        self.ax8.set_title("Field Magnitude Distribution")
        self.ax8.set_xlabel(r"$|\phi| / f_a$")
        self.ax8.set_ylabel("Probability Density")
        
        # Vorticity
        div9 = make_axes_locatable(self.ax9)
        cax9 = div9.append_axes("right", size="5%", pad=0.05)
        self.im9 = self.ax9.imshow(np.zeros((N, N)), cmap='RdBu_r', origin='lower')
        self.ax9.set_title("Vorticity Field")
        self.ax9.axis('off')
        self.fig1.colorbar(self.im9, cax=cax9, label="Vorticity")
        
        # Velocity field
        self.ax10.set_title("Phase Velocity Field")
        self.ax10.set_xlim(0, N)
        self.ax10.set_ylim(0, N)
        
        # Energy time
        self.line11a, = self.ax11.plot([], [], 'k', lw=2, alpha=0.8, label='Total')
        self.line11b, = self.ax11.plot([], [], 'r', lw=1, alpha=0.8, label='Kinetic')
        self.line11c, = self.ax11.plot([], [], 'g', lw=1, alpha=0.8, label='Gradient')
        self.line11d, = self.ax11.plot([], [], 'b', lw=1, alpha=0.8, label='Potential')
        self.ax11.set_title("Energy Evolution")
        self.ax11.set_xlabel("Time")
        self.ax11.set_ylabel("Energy Density")
        self.ax11.legend(fontsize=8)
        self.ax11.set_yscale('log')
        
        # String length
        self.line12, = self.ax12.plot([], [], 'r', alpha=0.8)
        self.ax12.set_title("String Length Density")
        self.ax12.set_xlabel("Time")
        self.ax12.set_ylabel("String Length/Area")
        
        # Scaling
        self.line13, = self.ax13.plot([], [], 'b', alpha=0.8)
        self.txt13 = self.ax13.text(0.05, 0.95, '', transform=self.ax13.transAxes,
                                    fontsize=10, va='top', bbox=dict(facecolor='white', alpha=0.7))
        self.ax13.set_title("String Network Scaling")
        self.ax13.set_xlabel("log(Time)")
        self.ax13.set_ylabel("log(String Length)")
        
        # String velocity
        self.line14, = self.ax14.plot([], [], color='orange', alpha=0.8, lw=2)
        self.ax14.set_title("Mean String Velocity")
        self.ax14.set_xlabel("Time")
        self.ax14.set_ylabel(r"$\langle v_{string} \rangle$")
        
        # String spacing
        self.line15, = self.ax15.plot([], [], color='cyan', alpha=0.8, lw=2)
        self.ax15.set_title("Inter-String Spacing")
        self.ax15.set_xlabel("Time")
        self.ax15.set_ylabel(r"$\langle d \rangle$")
        
        # Peak k
        self.line16, = self.ax16.plot([], [], color='magenta', alpha=0.8, lw=2)
        self.ax16.set_title("Spectral Peak Evolution")
        self.ax16.set_xlabel("Time")
        self.ax16.set_ylabel(r"$k_{peak}$")
        
        # Flux
        self.line17, = self.ax17.plot([], [], color='brown', alpha=0.8, lw=2)
        self.ax17.set_title("Energy Flux")
        self.ax17.set_xlabel("Time")
        self.ax17.set_ylabel(r"$\Phi_E$")
        
        # Correlation length
        self.line18, = self.ax18.plot([], [], color='navy', alpha=0.8, lw=2)
        self.ax18.set_title("Correlation Length")
        self.ax18.set_xlabel("Time")
        self.ax18.set_ylabel(r"$\xi$")
        
        self.fig1.suptitle("Axion Field Dynamics - Spatial Analysis", fontsize=16)
        self.fig2.suptitle("Axion Field Dynamics - Temporal Analysis", fontsize=16)
    

    
    def laplacian(self, f):
        """Compute discrete Laplacian with periodic BC"""
        return (-4*f + np.roll(f,1,0) + np.roll(f,-1,0) + 
                np.roll(f,1,1) + np.roll(f,-1,1)) / dx**2
    
    def grad_sq(self, f):
        """Compute squared gradient magnitude"""
        dfx = (np.roll(f,-1,1) - np.roll(f,1,1)) / (2*dx)
        dfy = (np.roll(f,-1,0) - np.roll(f,1,0)) / (2*dx)
        return dfx**2 + dfy**2
    
    def grad(self, f):
        """Compute gradient components"""
        dfx = (np.roll(f,-1,1) - np.roll(f,1,1)) / (2*dx)
        dfy = (np.roll(f,-1,0) - np.roll(f,1,0)) / (2*dx)
        return dfx, dfy
    
    def step(self):
        """Evolve field by one time step using Klein-Gordon equation"""
        self.phi_r_prev = self.phi_r.copy()
        self.phi_i_prev = self.phi_i.copy()
        
        r2 = self.phi_r**2 + self.phi_i**2
        dVr = 2*lam*(r2 - fa**2/2)*self.phi_r
        dVi = 2*lam*(r2 - fa**2/2)*self.phi_i
        
        self.phi_r_dot += dt * (self.laplacian(self.phi_r) - dVr - gamma*self.phi_r_dot)
        self.phi_i_dot += dt * (self.laplacian(self.phi_i) - dVi - gamma*self.phi_i_dot)
        
        self.phi_r += dt * self.phi_r_dot
        self.phi_i += dt * self.phi_i_dot
    

    
    def energy_components(self):
        """Compute kinetic, gradient, and potential energy densities"""
        kin = 0.5 * (self.phi_r_dot**2 + self.phi_i_dot**2)
        grad = 0.5 * (self.grad_sq(self.phi_r) + self.grad_sq(self.phi_i))
        r2 = self.phi_r**2 + self.phi_i**2
        pot = lam * (r2 - fa**2/2)**2
        return kin, grad, pot
    
    def axion_energy(self):
        """Compute axion field energy density"""
        theta = np.arctan2(self.phi_i, self.phi_r)
        theta_dot = (self.phi_r*self.phi_i_dot - self.phi_i*self.phi_r_dot) / (self.phi_r**2 + self.phi_i**2 + 1e-12)
        kin = 0.5 * fa**2 * theta_dot**2
        grad = 0.5 * fa**2 * self.grad_sq(theta)
        return kin + grad
    
    def energy_flux(self):
        """Estimate energy flux"""
        kin_c, grad_c, pot_c = self.energy_components()
        kin_p, grad_p, pot_p = self.energy_components()
        dE = np.mean((kin_c + grad_c + pot_c) - (kin_p + grad_p + pot_p))
        dt_eff = dt * anim_interval
        return np.abs(dE / dt_eff) if dt_eff > 0 else 0.0
    

    
    def velocity_field(self):
        """Compute phase velocity field"""
        theta = np.arctan2(self.phi_i, self.phi_r)
        theta_dot = (self.phi_r*self.phi_i_dot - self.phi_i*self.phi_r_dot) / (self.phi_r**2 + self.phi_i**2 + 1e-12)
        gx, gy = self.grad(theta)
        return -gy, gx, theta_dot
    
    def vorticity(self, vx, vy):
        """Compute vorticity from velocity field"""
        dvy_dx = (np.roll(vy,-1,1) - np.roll(vy,1,1)) / (2*dx)
        dvx_dy = (np.roll(vx,-1,0) - np.roll(vx,1,0)) / (2*dx)
        return dvy_dx - dvx_dy
    

    
    def correlation_func(self):
        """Compute two-point correlation function"""
        phi_mag = np.sqrt(self.phi_r**2 + self.phi_i**2)
        phi_norm = phi_mag - np.mean(phi_mag)
        fft = np.fft.fft2(phi_norm)
        power = np.abs(fft)**2
        corr = np.fft.ifft2(power).real
        corr = np.fft.fftshift(corr)
        corr /= corr[N//2, N//2]
        return self.radial_avg(corr)
    
    def corr_length(self, corr):
        """Extract correlation length"""
        target = 1.0/np.e
        idx = np.where(corr < target)[0]
        return float(idx[0]) if len(idx) > 0 else float(len(corr)-1)
    
    
    
    def power_spec(self):
        """Compute Fourier power spectrum of complex field"""
        phi_c = self.phi_r + 1j*self.phi_i
        fft = np.fft.fftshift(np.fft.fft2(phi_c))
        return np.abs(fft)**2 / phi_c.size
    
    def axion_spec(self):
        """Compute power spectrum of axion phase field"""
        theta = np.arctan2(self.phi_i, self.phi_r)
        fft = np.fft.fftshift(np.fft.fft2(theta))
        return np.abs(fft)**2 / theta.size
    
    def radial_avg(self, data):
        """Compute radially averaged profile"""
        y, x = np.indices(data.shape)
        c = (data.shape[0]//2, data.shape[1]//2)
        r = np.sqrt((x-c[1])**2 + (y-c[0])**2).astype(int)
        return np.bincount(r.ravel(), weights=data.ravel()) / np.maximum(1, np.bincount(r.ravel()))
    
    def find_peaks(self, spec, k):
        """Detect peaks in radial spectrum"""
        peaks, _ = find_peaks(spec, prominence=0.5, distance=3)
        return k[peaks].tolist() if len(peaks) > 0 else []
    
    
    
    # STRING ANALYSIS MACHINERY
    
    def unwrap(self, dth):
        """Unwrap phase differences to [-π, π]"""
        return (dth + np.pi) % (2*np.pi) - np.pi
    
    def detect_strings(self):
        """Detect topological defects via winding number"""
        theta = np.arctan2(self.phi_i, self.phi_r)
        dth1 = self.unwrap(np.roll(theta,-1,1) - theta)
        dth2 = self.unwrap(np.roll(theta,(-1,-1),(0,1)) - np.roll(theta,-1,1))
        dth3 = self.unwrap(np.roll(theta,-1,0) - np.roll(theta,(-1,-1),(0,1)))
        dth4 = self.unwrap(theta - np.roll(theta,-1,0))
        w = (dth1[:-1,:-1] + dth2[:-1,:-1] + dth3[:-1,:-1] + dth4[:-1,:-1])
        return w / (2*np.pi)
    
    def string_length(self):
        """Compute total string length per unit area"""
        w = self.detect_strings()
        L = np.sum(np.abs(w)) * dx
        return L / (N*N*dx**2)
    
    def string_velocity(self):
        """Estimate mean velocity of string network"""
        wind = self.detect_strings()
        mask = np.abs(wind) > 0.5
        if np.sum(mask) == 0:
            return 0.0
        vx, vy, _ = self.velocity_field()
        vx = vx[:-1, :-1]
        vy = vy[:-1, :-1]
        vmag = np.sqrt(vx**2 + vy**2)
        return float(np.mean(vmag[mask]))
    
    def string_spacing(self):
        """Compute mean inter-string distance"""
        wind = self.detect_strings()
        wind_int = np.round(wind).astype(int)
        sy, sx = np.where(np.abs(wind_int) > 0)
        if len(sx) < 2:
            return 0.0
        
        pos = np.column_stack([sx, sy])
        if len(pos) > 100:
            idx = np.random.choice(len(pos), 100, replace=False)
            pos = pos[idx]
        
        dists = []
        for p in pos:
            diff = pos - p
            # Handle periodic boundaries
            diff = np.where(diff > N/2, diff - N, diff)
            diff = np.where(diff < -N/2, diff + N, diff)
            d = np.sqrt(np.sum(diff**2, axis=1))
            d = d[d > 0]
            if len(d) > 0:
                dists.append(np.min(d))
        
        return float(np.mean(dists)) if len(dists) > 0 else 0.0
    
    def count_segments(self):
        """Count number of disconnected string segments"""
        wind = self.detect_strings()
        wind_int = np.round(wind).astype(int)
        mask = np.abs(wind_int) > 0
        _, n = label(mask)
        return n
    

    
    def store_data(self, step):
        """Store current field state for time evolution analysis"""
        kin, grad, pot = self.energy_components()
        E_ax = self.axion_energy()
        L_str = self.string_length()
        wind = self.detect_strings()
        
        self.t_list.append(step * dt)
        self.E_tot.append(np.mean(kin + grad + pot))
        self.E_kin.append(np.mean(kin))
        self.E_grad.append(np.mean(grad))
        self.E_pot.append(np.mean(pot))
        self.E_ax.append(np.mean(E_ax))
        self.L_string.append(L_str)
        self.winding_tot.append(np.sum(np.abs(wind)))
        self.phi_mean.append(np.mean(np.sqrt(self.phi_r**2 + self.phi_i**2)))
        
        # Spectral diagnostics
        P = self.power_spec()
        rad = self.radial_avg(P)
        k = np.arange(len(rad))
        peaks = self.find_peaks(rad, k)
        self.k_peak.append(peaks[0] if len(peaks) > 0 else 0.0)
        
        # String network diagnostics
        self.flux.append(self.energy_flux())
        self.v_string.append(self.string_velocity())
        self.d_string.append(self.string_spacing())
        
        # Correlation diagnostics
        corr = self.correlation_func()
        self.xi.append(self.corr_length(corr))
        self.n_segments.append(self.count_segments())
        
        # Vorticity diagnostics
        vx, vy, _ = self.velocity_field()
        vort = self.vorticity(vx, vy)
        self.vort_rms.append(np.sqrt(np.mean(vort**2)))
    

    
    def update(self, frame):
        """Update visualization for animation frame"""
        step = frame * anim_interval
        for _ in range(anim_interval):
            self.step()
        self.store_data(step)
        
        # Update phase
        theta = np.arctan2(self.phi_i, self.phi_r)
        self.im1.set_data(theta)
        wind = self.detect_strings()
        wind_int = np.round(wind).astype(int)
        py, px = np.where(wind_int == 1)
        ny, nx = np.where(wind_int == -1)
        self.scat_p.set_data(px+0.5, py+0.5)
        self.scat_n.set_data(nx+0.5, ny+0.5)
        n_seg = self.n_segments[-1] if self.n_segments else 0
        net_w = np.sum(wind)
        tot_w = np.sum(np.abs(wind))
        max_w = np.max(np.abs(wind))
        self.ax1.set_title(f"Phase with String Overlay\n"
                          f"Step {step} | Net: {net_w:.2f}, "
                          f"|w|: {tot_w:.2f}, Segments: {n_seg}")
        
        # Update energy
        kin, grad, pot = self.energy_components()
        self.im2.set_data(kin + grad + pot)
        self.im3.set_data(self.axion_energy())
        
        # Update spectrum
        P = self.power_spec()
        self.im4.set_data(np.log10(P + 1e-12))
        
        rad_phi = self.radial_avg(np.log10(P + 1e-12))
        P_ax = self.axion_spec()
        rad_ax = self.radial_avg(np.log10(P_ax + 1e-12))
        
        k = np.arange(len(rad_phi))
        logk = np.log10(k[1:] + 1e-12)
        self.line5.set_data(logk, rad_phi[1:])
        self.line6.set_data(logk, rad_ax[1:])
        
        peaks = self.find_peaks(rad_phi[1:], k[1:])
        if len(peaks) > 0:
            pk_logk = np.log10(np.array(peaks) + 1e-12)
            pk_val = np.interp(pk_logk, logk, rad_phi[1:])
            self.peak5.set_data(pk_logk, pk_val)
        else:
            self.peak5.set_data([], [])
        
        mask = (k[1:] > 4) & (k[1:] < N//2)
        if np.any(mask) and len(logk[mask]) > 3:
            s, _, _, _, _ = linregress(logk[mask], rad_phi[1:][mask])
            self.txt5.set_text(f'n = {-s:.2f}')
            s2, _, _, _, _ = linregress(logk[mask], rad_ax[1:][mask])
            self.txt6.set_text(f'n = {-s2:.2f}')
        
        # Update velocity
        if frame % 2 == 0:
            vx, vy, _ = self.velocity_field()
            skip = 8
            X, Y = np.meshgrid(np.arange(0, N, skip), np.arange(0, N, skip))
            U = vx[::skip, ::skip]
            V = vy[::skip, ::skip]
            mag = np.sqrt(U**2 + V**2) * 0.5
            max_m = np.percentile(mag, 95)
            clip_m = np.maximum(mag, max_m)
            U = U / (clip_m + 1e-12)
            V = V / (clip_m + 1e-12)
            self.ax10.clear()
            self.ax10.quiver(X, Y, U, V, scale=25, alpha=0.7)
            self.ax10.set_title("Phase Velocity Field")
            self.ax10.set_xlim(0, N)
            self.ax10.set_ylim(0, N)
        
        # Update spatial diagnostics
        if frame % 3 == 0:
            corr = self.correlation_func()
            r = np.arange(len(corr))
            self.line7.set_data(r, corr)
            
            # Histogram
            phi_mag = np.sqrt(self.phi_r**2 + self.phi_i**2)
            phi_n = phi_mag.flatten() / fa
            self.ax8.clear()
            c, b, _ = self.ax8.hist(phi_n, bins=60, density=True, alpha=0.7, color='blue', ec='black', lw=0.5)
            vac = 1.0/np.sqrt(2)
            self.ax8.axvline(x=vac, color='red', ls='--', label=r'$|\phi|=f_a/\sqrt{2}$ (vacuum)', lw=2)
            m = np.mean(phi_n)
            s = np.std(phi_n)
            bw = b[1] - b[0]
            integ = np.sum(c) * bw
            self.ax8.text(0.98, 0.98, f'μ={m:.3f}, σ={s:.3f}\n∫P(x)dx={integ:.3f}',
                         transform=self.ax8.transAxes, va='top', ha='right',
                         bbox=dict(fc='white', alpha=0.8), fontsize=9)
            self.ax8.set_title("Field Magnitude Distribution")
            self.ax8.set_xlabel(r"$|\phi| / f_a$")
            self.ax8.set_ylabel("Probability Density")
            x0 = max(0, min(vac-0.3, m-3*s))
            x1 = max(vac+0.3, m+3*s)
            self.ax8.set_xlim(x0, x1)
            if len(c) > 0:
                self.ax8.set_ylim(0, np.max(c)*1.15)
            self.ax8.legend(fontsize=8, loc='upper left')
            self.ax8.grid(alpha=0.3, ls=':')
            
            # Vorticity
            vx, vy, _ = self.velocity_field()
            vort = self.vorticity(vx, vy)
            v_max = np.percentile(np.abs(vort), 99)
            self.im9.set_data(vort)
            self.im9.set_clim(-v_max, v_max)
        
        # Time plots
        if len(self.t_list) > 1:
            t = np.array(self.t_list)
            self.line11a.set_data(t, self.E_tot)
            self.line11b.set_data(t, self.E_kin)
            self.line11c.set_data(t, self.E_grad)
            self.line11d.set_data(t, self.E_pot)
            
            if t[-1] > t[0]:
                self.ax11.set_xlim(t[0], t[-1])
            if len(t) > 5:
                y0 = min(min(self.E_kin), min(self.E_grad), min(self.E_pot))
                y1 = max(self.E_tot)
                if y1 > y0:
                    self.ax11.set_ylim(y0*0.5, y1*2)
            
            if t[-1] > 2:
                m = t > 2
                tf = t[m]
                if len(tf) > 1:
                    Lf = np.array(self.L_string)[m]
                    self.line12.set_data(tf, Lf)
                    self.ax12.set_xlim(tf[0], tf[-1])
                    if max(Lf) > 0:
                        self.ax12.set_ylim(0, max(Lf)*1.1)
                    
                    if len(tf) > 6:
                        lt = np.log10(tf[5:] + 1e-12)
                        lL = np.log10(Lf[5:] + 1e-12)
                        self.line13.set_data(lt, lL)
                        
                        if len(lt) > 10:
                            sl, _, _, _, _ = linregress(lt[-20:], lL[-20:])
                            self.txt13.set_text(f'L ~ t^{sl:.2f}')
                        
                        if len(lt) > 0 and lt[-1] > lt[0]:
                            self.ax13.set_xlim(lt[0], lt[-1])
                        if len(lL) > 0 and max(lL) > min(lL):
                            self.ax13.set_ylim(min(lL)-0.5, max(lL)+0.5)
            
            has_range = len(t) > 1 and t[-1] > t[0]
            
            if len(self.v_string) > 0:
                self.line14.set_data(t, self.v_string)
                if has_range:
                    self.ax14.set_xlim(t[0], t[-1])
                mv = max(self.v_string)
                if mv > 0:
                    self.ax14.set_ylim(0, mv*1.1)
            
            if len(self.d_string) > 0:
                d = np.array(self.d_string)
                vm = d > 0
                if np.any(vm):
                    self.line15.set_data(t[vm], d[vm])
                    if has_range:
                        self.ax15.set_xlim(t[0], t[-1])
                    self.ax15.set_ylim(0, max(d[vm])*1.1)
            
            if len(self.k_peak) > 0:
                kp = np.array(self.k_peak)
                vm = kp > 0
                if np.any(vm):
                    self.line16.set_data(t[vm], kp[vm])
                    if has_range:
                        self.ax16.set_xlim(t[0], t[-1])
                    self.ax16.set_ylim(0, max(kp[vm])*1.1)
            
            if len(self.flux) > 0:
                self.line17.set_data(t, self.flux)
                if has_range:
                    self.ax17.set_xlim(t[0], t[-1])
                mf = max(self.flux)
                if mf > 0:
                    self.ax17.set_ylim(0, mf*1.1)
            
            if len(self.xi) > 0:
                self.line18.set_data(t, self.xi)
                if has_range:
                    self.ax18.set_xlim(t[0], t[-1])
                mx = max(self.xi)
                if mx > 0:
                    self.ax18.set_ylim(0, mx*1.1)
        
        return (self.im1, self.scat_p, self.scat_n, self.im2, self.im3, 
                self.im4, self.line5, self.line6)
    
    
    
    def run(self):
        nframes = total_steps // anim_interval
        self.anim1 = animation.FuncAnimation(
            self.fig1, self.update, frames=nframes,
            interval=150, blit=False, repeat=False)
        self.anim2 = animation.FuncAnimation(
            self.fig2, lambda f: (), frames=nframes,
            interval=150, blit=False, repeat=False)
        plt.show()



def main():
    sim = AxionSim()
    sim.run()



if __name__ == "__main__":
    main()
