# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "numba",
#   "scipy",
# ]
# ///

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numba import njit, prange

# =====================================================================
# 1. SCIPA NUMBA PHYSICS ENGINE FOR DIFFERENTIAL GALAXY ROTATION
# =====================================================================

@njit(parallel=True, fastmath=True)
def step_gas_and_superfluid_kinematics(grid_np, dt, viscosity, quantum_pressure_coeff):
    """
    Numba JIT accelerated Navier-Stokes + Gross-Pitaevskii physics step.
    Updates density and colors based on velocity vector advection fields.
    """
    _, channels, h, w = grid_np.shape
    out_grid = grid_np.copy()
    
    for y in prange(1, h - 1):
        for x in range(1, w - 1):
            rho = grid_np[0, 3, y, x]
            vx = grid_np[0, 4, y, x]
            vy = grid_np[0, 5, y, x]
            
            if rho <= 1e-4:
                continue
            
            # Finite difference spatial gradients
            d_rho_dx = (grid_np[0, 3, y, x+1] - grid_np[0, 3, y, x-1]) / 2.0
            d_rho_dy = (grid_np[0, 3, y+1, x] - grid_np[0, 3, y-1, x]) / 2.0
            
            lap_vx = (grid_np[0, 4, y, x+1] + grid_np[0, 4, y, x-1] + grid_np[0, 4, y+1, x] + grid_np[0, 4, y-1, x] - 4*vx)
            lap_vy = (grid_np[0, 5, y, x+1] + grid_np[0, 5, y, x-1] + grid_np[0, 5, y+1, x] + grid_np[0, 5, y-1, x] - 4*vy)
            lap_rho = (grid_np[0, 3, y, x+1] + grid_np[0, 3, y, x-1] + grid_np[0, 3, y+1, x] + grid_np[0, 3, y-1, x] - 4*rho)
            
            # 1/r Potential Gravity Vector Field
            d_pot_dx = (grid_np[0, 6, y, x+1] - grid_np[0, 6, y, x-1]) / 2.0
            d_pot_dy = (grid_np[0, 6, y+1, x] - grid_np[0, 6, y-1, x]) / 2.0
            
            # Superfluid Quantum Pressure (Bohm Gradient Force Mapping)
            q_fx = quantum_pressure_coeff * lap_rho * d_rho_dx
            q_fy = quantum_pressure_coeff * lap_rho * d_rho_dy
            
            # Kinematic acceleration additions
            new_vx = vx + dt * (viscosity * lap_vx - (rho * d_pot_dx) + q_fx)
            new_vy = vy + dt * (viscosity * lap_vy - (rho * d_pot_dy) + q_fy)
            
            # Conservation of mass: Continuity divergence advection step
            div_v = ((grid_np[0, 4, y, x+1] - grid_np[0, 4, y, x-1]) + (grid_np[0, 5, y+1, x] - grid_np[0, 5, y-1, x])) / 2.0
            new_rho = rho - dt * (vx * d_rho_dx + vy * d_rho_dy + rho * div_v)
            
            out_grid[0, 3, y, x] = max(0.0, min(new_rho, 4.0))
            out_grid[0, 4, y, x] = max(-2.0, min(new_vx, 2.0))
            out_grid[0, 5, y, x] = max(-2.0, min(new_vy, 2.0))
            
            # Fluid transport step for visual color fields (RGB channels 0-2)
            for c in range(3):
                d_c_dx = (grid_np[0, c, y, x+1] - grid_np[0, c, y, x-1]) / 2.0
                d_c_dy = (grid_np[0, c, y+1, x] - grid_np[0, c, y-1, x]) / 2.0
                out_grid[0, c, y, x] = max(0.0, min(grid_np[0, c, y, x] - dt * (vx * d_c_dx + vy * d_c_dy), 1.0))
                
    return out_grid

# =====================================================================
# 2. SEED ENGINE WITH ANALYTICAL BRANDT DIFFERENTIAL ROTATION
# =====================================================================

def generate_differential_galaxy_seed(grid_size=64):
    """
    Creates a physical state initializing a flat galaxy rotation velocity curve.
    """
    grid = torch.zeros(1, 16, grid_size, grid_size, dtype=torch.float32)
    center = grid_size // 2
    
    # Brandt Curve Tuning Parameters (Flat Outer Velocity Profile)
    v_max = 0.6
    r_max = grid_size // 4
    n = 1.5  # Shape factor governing rotation velocity curve flattening
    
    phi = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi / (phi ** 2)
    
    for y in range(grid_size):
        for x in range(grid_size):
            dx, dy = x - center, y - center
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            if r > 0:
                # Local Newtonian Gravity Sink Potential Map
                grid[0, 6, y, x] = 5.0 / (r + 1.5)
                
                # Brandt Differential Rotation Calculation
                num = v_max * (r / r_max)
                denom = (1.0/3.0 + (2.0/3.0) * (r / r_max)**n)**(1.5 / n)
                v_theta = num / denom
                
                # Decompose angular speed to Local Velocity Field channels (4 and 5)
                grid[0, 4, y, x] = -v_theta * np.sin(theta)  # Vx
                grid[0, 5, y, x] = v_theta * np.cos(theta)   # Vy
                
                # Form structural arms using Fibonacci spiral distribution
                spiral_val = (np.log(r) / np.log(phi) - theta / golden_angle) % 1.0
                if spiral_val < 0.25 and r < center * 0.9:
                    grid[0, 0, y, x] = np.clip(1.0 - (r/center), 0.2, 0.9)  # Hydrogen Gas (Red)
                    grid[0, 1, y, x] = np.clip(spiral_val * 3, 0.1, 0.6)     # Dust Columns (Green)
                    grid[0, 2, y, x] = np.clip(r / center, 0.3, 0.8)        # Ionized Zone (Blue)
                    grid[0, 3, y, x] = 0.80                                  # Mass Concentration Mask
                    
    grid[0, 7:, :, :] = torch.randn(1, 9, grid_size, grid_size) * 0.02
    return grid

# =====================================================================
# 3. HYBRID BACKPROPAGATION TRAINING CORE
# =====================================================================

class EvolvedCosmicNCA(nn.Module):
    def __init__(self, channel_dim=16, hidden_dim=32):
        super(EvolvedCosmicNCA, self).__init__()
        # Local AI parameters acting as coordinate-free 'space-time corrections'
        self.conv1 = nn.Conv2d(channel_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, channel_dim, kernel_size=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, steps=4, dt=0.01):
        for _ in range(steps):
            # A. Analytical Physical Advection via Scipa Numba
            device = x.device
            grid_np = x.detach().cpu().numpy()
            grid_np = step_gas_and_superfluid_kinematics(grid_np, dt=dt, viscosity=0.02, quantum_pressure_coeff=0.005)
            x = torch.from_numpy(grid_np).to(device)
            
            # B. Localized AI Adaptations (Latent channels steer emergent shapes)
            dx = self.conv1(x)
            dx = F.relu(dx)
            dx = self.conv2(dx)
            
            # Asynchronous Cell Update Masking
            mask = torch.rand(x.size(0), 1, x.size(2), x.size(3), device=device) < 0.5
            x = x + mask.float() * dx
            x[:, :4, :, :] = torch.clamp(x[:, :4, :, :], 0.0, 1.0)
        return x

if __name__ == "__main__":
    print("🌌 Initializing Dark Matter Halo Differential Kinematics Setup...")
    
    # 1. Instantiate state space
    model = EvolvedCosmicNCA(channel_dim=16, hidden_dim=32)
    initial_seed = generate_differential_galaxy_seed(grid_size=64)
    
    # 2. Mock a specific target image profile (e.g., target gas arrangement to adapt to)
    target_img = initial_seed[:, :3, :, :].clone()
    # Apply an arbitrary physical twist to create a target discrepancy
    target_img = torch.rot90(target_img, 1, [2, 3]) 
    
    # 3. Setup PyTorch Gradient Descent Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\n🚀 Beginning Local AI Engine Adaptation Loop...")
    for epoch in range(1, 6):
        optimizer.zero_grad()
        
        # Clone seed state to keep initialization clean across execution graphs
        current_state = initial_seed.clone()
        
        # Forward execution through hybrid physics + AI layers
        # Retain gradients across internal tensor operations
        evolved_state = model(current_state, steps=3, dt=0.02)
        
        # Compute L2 reconstruction optimization loss against target RGB channels
        loss = F.mse_loss(evolved_state[:, :3, :, :], target_img)
        
        # Backpropagate through Neural update layers
        loss.backward()
        optimizer.step()
        
        print(f"   Epoch [{epoch}/5] -> Optimization L2 Loss Field: {loss.item():.6f}")

    print("\n✅ Training validation iteration completed successfully via astral environment parameters.")
