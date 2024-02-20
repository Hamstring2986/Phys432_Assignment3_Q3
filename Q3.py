"""
This code simmulates the flow of viscous lava down an incline plane of 45 degrees. 
See the submission pdf for the derivation of the equation of motion.
The output is an animation simulating the velocity field of lava supperimposed on the steady state solution.
The axis is perpendicular to the direction of propagation, again see submission pdf.

@author: Marc-Antoine Leclerc
@collab: None
February 26th, 2024
"""
######################################################################################################
#                               Librairies, Function and Constants
######################################################################################################
import matplotlib.pyplot as plt
import numpy as np

def steady_state_sol(x):
    """
    Computes the steady state solution. This was calculated in the lectures
    """
    prefactor = -g/D *np.sin(alpha)
    return prefactor*(1/2*x**2-Ngrid*x)

Ngrid = 50 # mm. Grid size, also equal to H.
Nsteps = 1000 # Number of timesteps
dt = 0.00001 # Timestep
dx = 1 # Grid step
g = 9800 # mm/s, gravitationnal acceleration
alpha = np.pi/4 # Angle of the incline plane in radiant. This is a value I chose.
D = 10**6 # mm^2/s. This is the diffusion coefficient or in other words, the kinematic viscosity "nu".
       # From the lecture note on lava viscosity, D = 10^4 cm^2/s
beta = D*dt/(dx**2) # Beta parameter for diffusion
y_lim = 10 # Limit of y-axis for visualization

######################################################################################################
#                                      Initial Conditions
######################################################################################################

# Create grid
x = np.arange(0, Ngrid, dx, dtype=float) 

# Create velocity field defined over the grid
# Initialize it to 0
u = np.zeros(len(x), dtype = float)

# Create the steady state solution
u_steady = np.array([steady_state_sol(xs) for xs in x], dtype=float)

# Setting up diffusion matrix (without boundary conditions)
A = np.eye(Ngrid, dtype = float)*(1.0 + 2.0*beta) + np.eye(Ngrid, k=1, dtype = float)*(-beta) \
    + np.eye(Ngrid, k=-1, dtype = float)*(-beta)

# Setting up no-slip boundary conditions
A[0][0] = 1.0
A[0][1] = 0.0

# Setting up no-stress
A[-1][-1] = 1.0 + beta

# Set up the plot
plt.ion()
fig, ax = plt.subplots(1,1)
ax.set_title("Velocity Field of Lava")
ax.set_xlabel("x-axis (mm)")
ax.set_ylabel("Velocity (mm/s)")

# Plotting the steady state solution in background
ax.plot(x,u_steady,"k-")

# Plotting object that will be updated
plot, = ax.plot(x,u,"r-")

# Axes limits
ax.set_xlim([0,Ngrid])
ax.set_ylim([0,y_lim])

# Draw plot
fig.canvas.draw()

######################################################################################################
#                                           Evolution
######################################################################################################

count = 0
while count < Nsteps:

    # Calculating the advection part
    u[1:] = u[1:] + dt*g*np.sin(alpha) 

    # Solving for the next time step using implicit method
    u = np.linalg.solve(A,u)

    #Update the plot
    plot.set_ydata(u)

    # Draw plot
    fig.canvas.draw()
    plt.pause(0.01)

    count += 1