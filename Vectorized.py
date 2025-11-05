import numpy as np
from dataclasses import dataclass
from scipy.signal import fftconvolve
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import linalg as spla
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from matplotlib.animation import FFMpegWriter
#from IPython.display import HTML

#===============================================================================
#Helper Functions
#===============================================================================

#Parameters---------------------------------------------------------------------
	
@dataclass
class SimParams:
	h: float    = 0.005     #Spatial grid spacing
    dt:float    = 0.005     #Timestep
    rtol:float  = 0.1       #Tolerance to say an agent has 'reached' a point
    R: float    = 1         #Radius of initial semicircle path

    # Field
    kp0: float = 3000.0     #numerator for per-agent kp=kp0/num_steps
    kpl: float = 30.0       #Generation constant
    kn: float  = 0.07       #Decay constant
    D: float   = 0.0005     #Diffusion costant
    z: float   = 0.00       #Saturation Term
    sigma: float = 0.02     #Width of laid trail

    # Agent
    v_a: float  = 0.70      #Average speed of agents
    s_mu: float = 0.05      #sigma_mu: Standard deviation of agent speeds
    l: float    = 0.01      #Length between antennae and pher laying organs
    defl:float  = 0.8*np.pi #Angular velocity of turning to avoid collisions
    rdist:float = 0.07      #Minimum distance at which collision avoidance behaviour begins

    rest:int    = 200       #Avg no. of timesteps to rest for
    s_rest:float= 50        #Variance in no. of rest steps

    #Intrinsic Policy
    G: float    = 0.05      #Directional Gain
    n: float    = 0.00      #Intrinsic policy noise

    #OU Process
    alpha: float = 0.3      #Strength of mean reverting behaviour
    diff: float  = 0.00    #OU diffusion factor
    p_switch: float = 0.005  #Switching probability (Intrinsic to OU) per step

    def __post_init__(self):
        self.colony = np.array([self.R, 0.0])   #Colony Location
        self.food   = np.array([-self.R, 0.0])  #Target Location
        self.Nx = 3*int(self.R/self.h)          #Number of gridpoints
        self.Ny = 2*int(self.R/self.h)
        self.x = np.linspace(-self.Nx//2, self.Nx//2, self.Nx, endpoint=False)
        self.y = np.linspace(-1*self.Ny//4, (-1*self.Ny//4)+self.Ny, self.Ny, endpoint=False)
        #grid coordinates (note: X,Y shape (Nx,Ny))
        self.X, self.Y = np.meshgrid(self.x*self.h, self.y*self.h, indexing='ij')

def sample_gradient(field, sensors):
    """
    Sample discrete gradient of the pheromone field at each sensor's nearest grid cell.
    """
    # Convert world coordinates -> grid indices
    i = ((agents["pos"][:,0] - params.X.min()) / params.h).astype(int)
    j = ((agents["pos"][:,1] - params.Y.min()) / params.h).astype(int)

    # Keep only valid indices
    valid = (i > 0) & (i < params.Nx-1) & (j > 0) & (j < params.Ny-1)
    gx = np.zeros_like(i, dtype=float)
    gy = np.zeros_like(j, dtype=float)

    # Compute central differences only for valid ones
    gx[valid] = (c[i[valid]+1, j[valid]] - c[i[valid]-1, j[valid]]) / (2 * params.h)
    gy[valid] = (c[i[valid], j[valid]+1] - c[i[valid], j[valid]-1]) / (2 * params.h)

    return np.stack([gx, gy], axis=1)

#===============================================================================
#Initialization
#===============================================================================

#Field  ------------------------------------------------------------------------
#Operator to Perform Implicit Euler Integration: A*c(n+1) = c(n) + generation_term
def matvec(v):
    Nx, Ny = params.Nx, params.Ny
    dt, D, kn, h = params.dt, params.D, params.kn, params.h

    V = v.reshape((Nx, Ny))
    L = np.zeros_like(V)
    #Laplacian using central differences
    L[1:-1,1:-1] = (V[2:,1:-1] + V[:-2,1:-1] + V[1:-1,2:] + V[1:-1,:-2] - 4*V[1:-1,1:-1]) / (h**2)
    return ((1 + kn*dt) * v - (dt * D) * L.ravel())

#Initial semicircle path
def set_initial_semicircle():
    Nx, Ny = params.Nx, params.Ny
    R, sigma = params.R, params.sigma

    c = np.zeros((Nx, Ny), dtype=float)
    r = np.sqrt(params.X**2 + params.Y**2)
    mask = (params.Y >= 0) & (np.abs(r - R) <= params.h)
    c[mask] = 3.0
    kern = make_gaussian_kernel(sigma=1.5*params.sigma, h=params.h)
    c = fftconvolve(c, kern, mode='same')
    return c

def make_gaussian_kernel(sigma, h):
    size = int(6 * sigma / h)
    if size % 2 == 0: size += 1
    x = np.linspace(-3*sigma, 3*sigma, size)
    y = np.linspace(-3*sigma, 3*sigma, size)
    Xg, Yg = np.meshgrid(x, y)
    k = np.exp(-(Xg**2 + Yg**2)/(2*sigma**2))
    k /= k.sum()
    return k

#Agents ------------------------------------------------------------------------
def initialize_agents(w=12, f=8):
    total = w + f

    #Arrays
    pos =   np.zeros((total,2), dtype=float)
    thet =  np.zeros(total, dtype=float)
    mu =    np.random.normal(scale=params.s_mu, size=total)
    laying= np.zeros(total, dtype=bool)
    isForager = np.zeros(total, dtype=bool)
    state = np.zeros(total, dtype=bool)  #0=resting, 1=moving
    steps = np.zeros(total, dtype=int)
    tot_rest = (np.random.randn(total)*params.s_rest + params.rest).astype(int)
    rest_steps = np.zeros(total, dtype=int)
    kp_agent = np.full(total, params.kpl, dtype=float)
    policy = np.zeros(total, dtype=bool) #0=Intrinsic, 1=OU
    tgtphi = np.zeros(total, dtype=float)   #stored heading direction in agent memory
    gradient = np.zeros((total,2), dtype=float)

    R = params.R
    idx = 0

    #workers:
    for i in range(w//2):
        pos[idx] = [R*np.cos(2*i*np.pi/w), R*np.sin(2*i*np.pi/w)]
        thet[idx] = (2*i*np.pi/w + np.pi/2)
        state[idx] = 1
        idx += 1
    for j in range(w//2):
        i = j + 0.8
        pos[idx] = [R*np.cos(2*i*np.pi/w), R*np.sin(2*i*np.pi/w)]
        thet[idx] = (2*i*np.pi/w - np.pi/2)
        state[idx] = 1
        laying[idx] = True
        idx += 1

    #foragers:
    for i in range(f):
        pos[idx] = params.colony.copy()
        thet[idx] = np.pi/2
        isForager[idx] = True
        state[idx] = 0  #resting to start
        rest_steps[idx] = tot_rest[idx]
        idx += 1

    #phat = np.column_stack([np.cos(thet), np.sin(thet)])
    agents = {
        "pos": pos,
        "thet": thet,
        #"phat": phat,
        "mu": mu,
        "laying": laying,
        "isForager": isForager,
        "state": state,
        "steps": steps,
        "rest_steps": rest_steps,
        "tot_rest": tot_rest,
        "kp_agent": kp_agent,
        "policy": policy,
        "tgtphi": tgtphi,
        "gradient": gradient
    }
    return agents

#===============================================================================
#Sim Steps
#===============================================================================

def sim_step():
    #agent updates
    step_agents(agents, c)

    #repulsion
    #apply_repulsion(agents)

    #deposit pheromones
    rho = deposit_agents()

    #Implicit diffusion+decay step
    rhs = cvec
    rhs[:] = c.ravel() + rho.ravel() * params.dt
    sol, info = spla.cg(A, rhs, x0=cvec, maxiter=500, atol=1e-6)
    if info != 0:
        #fallback: simple explicit update
        cvec[:] = rhs / (1 + params.kn * params.dt)
    cvec[:] = sol
    c[:] = cvec.reshape((params.Nx, params.Ny))

def deposit_agents():
    mask = np.zeros((params.Nx, params.Ny), dtype=float)
    laying_idx = np.nonzero(agents["laying"])[0]
    if laying_idx.size == 0:
        return np.zeros_like(mask)
    pos = agents["pos"][laying_idx]# - params.l * agents["phat"][laying_idx]
    kps = agents["kp_agent"][laying_idx]
    #map to integer grid indices
    ix = np.floor((pos[:,0] - params.X.min()) / params.h).astype(int)
    iy = np.floor((pos[:,1] - params.Y.min()) / params.h).astype(int)

    np.add.at(mask, (ix, iy), kps)
    deposit = fftconvolve(mask, kernel, mode='same')
    #saturation
    deposit = deposit / (1.0 + params.z * deposit)
    return deposit

def apply_repulsion(agents):
    """Apply repulsion to agents not at colony"""
    pos = agents["pos"]

    #consider only agents not at colony (active)
    active_mask = ~((np.sum((pos - params.colony)**2, axis=1)) < (params.rtol**2)) #1 if not at colony
    act_idx = np.nonzero(active_mask)[0]
    positions = pos[act_idx]
    tree = cKDTree(positions)
    pairs = np.array(list(tree.query_pairs(r=params.rdist)))  #array of agents closer than rdist shape (M,2) or empty
    if pairs.size == 0:
        return
    a = pairs[:,0]; b = pairs[:,1]
    ia = act_idx[a]; ib = act_idx[b]  #indices

    #accumulate deflection amount per agent
    deflect = np.zeros(pos.shape[0], dtype=float)
    #each pair contributes defl*dt to both agents
    np.add.at(deflect, ia, params.defl * params.dt)
    np.add.at(deflect, ib, params.defl * params.dt)

    # apply deflection sign: rotate by +deflect for both (keeps them moving apart)
    agents["thet"] += deflect
    #agents["phat"] = np.column_stack([np.cos(agents["thet"]), np.sin(agents["thet"])])
    return

def step_agents(agents, field):
    """
    Step update for all agents, Handles:
      - intrinsic & OU policies
      - out-of-bounds reset to colony
      - state transitions, resting, forager switching
    """

    N = agents["pos"].shape[0]
    pos = agents["pos"]
    thet = agents["thet"]
    #phat = agents["phat"]
    state = agents["state"]        #0: Resting, 1: Moving
    laying = agents["laying"]
    isFor = agents["isForager"]
    policy = agents["policy"]      #0: Intrinsic, 1: OU

    #Reset agents that are out of bounds to Colony
    mask_reset = (
        (agents["pos"][:, 0] < params.X.min() + params.h) |
        (agents["pos"][:, 0] >= params.X.max() - params.h) |
        (agents["pos"][:, 1] < params.Y.min() + params.h) |
        (agents["pos"][:, 1] >= params.Y.max() - params.h)
    )
    agents["pos"][mask_reset] = params.colony.copy()  #reset to colony

    #Compute gradients at sensors (antennae)
    sensors = pos + params.l * np.column_stack([np.cos(agents["thet"]), np.sin(agents["thet"])])
    grads = sample_gradient(field, sensors)
    agents["gradient"] = grads


    #grads = sample_gradient(field, sensors)

    #Mask groups
    laying = (laying == True)
    moving = (state == 1)
    resting = (state == 0)
    int_mask = (policy == 0) & moving
    ou_mask  = (policy == 1) & moving

    #Intrinsic policy update
    intrinsic_update(agents, np.nonzero(int_mask)[0], grads[int_mask])

    #OU policy update
    if np.any(ou_mask):
        ou_update(agents, np.nonzero(ou_mask)[0])

    #Foragers: probabilistic switching from intrinsic policy to OU
    mask_f_int = isFor & (policy == 0) & moving #Foragers that are moving with Int Policy
    if np.any(mask_f_int):
        switch = np.random.rand(np.count_nonzero(mask_f_int)) < params.p_switch
        idx = np.nonzero(mask_f_int)[0]
        if np.any(switch):
            idx_switch = idx[switch]
            vec = params.food.reshape(1,2) - agents["pos"][idx_switch] #New heading direction
            agents["thet"][idx_switch] = np.arctan2(vec[:,1], vec[:,0]) % 2*np.pi
            agents["tgtphi"][idx_switch] = np.arctan2(vec[:,1], vec[:,0]) % 2*np.pi
            agents["policy"][idx_switch] = 1     #switch to OU
            agents["steps"][idx_switch] = 0      #reset step count
            print(np.arctan2(vec[:,1], vec[:,0]))

    #Reached food
    mask_food = moving & (np.linalg.norm(pos - params.food, axis=1) < params.rtol) & ~laying  #Agents that have reached food
    if np.any(mask_food):
        pos[mask_food] = params.food
        thet[mask_food] += np.pi
        #phat[mask_food] = np.column_stack([np.cos(thet[mask_food]), np.sin(thet[mask_food])])
        agents["tgtphi"] += np.pi
        agents["laying"][mask_food] = True
        agents["steps"][mask_food] += np.round( params.rtol / (params.v_a * params.dt)).astype(int)

    #Reached colony
    mask_col = moving & (np.linalg.norm(pos - params.colony, axis=1) < params.rtol) & laying
    if np.any(mask_col):
        pos[mask_col] = params.colony
        thet[mask_col] = np.pi / 2.0
        #phat[mask_col] = np.column_stack([np.cos(thet[mask_col]), np.sin(thet[mask_col])])
        agents["rest_steps"][mask_col] = agents["tot_rest"][mask_col]
        agents["steps"][mask_col] = 0
        state[mask_col] = 0            # switch to resting
        agents["laying"][mask_col] = False

    #Resting agents decrement timers
    if np.any(resting):
        agents["rest_steps"][resting] -= 1
        mask_resume = (agents["rest_steps"] <= 0) & resting
        if np.any(mask_resume):
            state[mask_resume] = 1      #start moving again
            agents["rest_steps"][mask_resume] = 0
            agents["steps"][mask_resume] = 0

    #Foragers in OU returning
    mask_f_ou = isFor & (policy == 1) & moving & laying
    if np.any(mask_f_ou):
        #when steps reach 0, switch back to intrinsic
        zero_steps = mask_f_ou & (agents["steps"] <= 0)
        agents["policy"][zero_steps] = 0
        agents["thet"][zero_steps] -= np.pi/2

    #Increment / decrement step counters
    #For intrinsic movers:
    agents["steps"][int_mask] += 1
    #For OU movers, heading to food:
    agents["steps"][ou_mask & ~laying] += 1
    #For OU movers returning:
    agents["steps"][ou_mask & laying] -= 1

    #Write back (in-place)
    agents["pos"] = pos
    agents["thet"] = thet
    #agents["phat"] = phat
    agents["state"] = state

def intrinsic_update(agents, idx, grads):
    """Intrinsic movement update: chemotaxis + noise"""
    thet = agents["thet"][idx]
    phat = np.column_stack([np.cos(agents["thet"][idx]), np.sin(agents["thet"][idx])])
    pos = agents["pos"][idx]

    dtheta = params.G * (phat[:, 0] * grads[:, 1] - phat[:, 1] * grads[:, 0]) + np.random.randn(len(idx)) * params.n * np.sqrt(params.dt)
    thet = (thet + dtheta) % (2 * np.pi)

    #Update position and heading
    phat = np.column_stack([np.cos(thet), np.sin(thet)])

    #Write back
    agents["thet"][idx] = thet
    #agents["phat"][idx] = phat
    agents["pos"][idx] += params.v_a * (1 + agents["mu"][idx])[:, None] * phat * params.dt
    #print(dtheta)

def ou_update(agents, idx):
    """Ornstein-Uhlenbeck (OU) update"""
    thet = agents["thet"][idx]
    #phat = agents["phat"][idx]
    #pos = agents["pos"][idx]

    #OU process in theta space
    """
    dthet = params.alpha * (agents["tgtphi"][idx] - agents["thet"][idx]) * params.dt \
      + np.sqrt(2 * params.diff * params.dt) * np.random.randn(len(idx))"""
    #thet = (thet + dthet) % (2 * np.pi)
    phat = np.column_stack([np.cos(thet), np.sin(thet)])

    #agents["thet"][idx] = thet
    #agents["phat"][idx] = phat
    agents["pos"][idx] += params.v_a * (1 + agents["mu"][idx])[:, None] * phat * params.dt


#===============================================================================
#Animation
#===============================================================================

def animate_sim(steps=1400, interval=25):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_xlim(params.X.min(), params.X.max())
    ax.set_ylim(params.Y.min(), params.Y.max())

    #colony / food visuals
    for i in range(6):
        r = params.rtol * 0.4 * (1 + 0.3*i)
        a = 0.3 * (1 - i/6)
        ax.add_patch(plt.Circle(params.colony, r, color='red', alpha=a, lw=0, zorder=2))
        ax.add_patch(plt.Circle(params.food, r, color='mediumvioletred', alpha=a, lw=0, zorder=2))

    im = ax.imshow(c.T, origin='lower', cmap='RdPu', vmin=0, vmax=1,
                   extent=[params.X.min(), params.X.max(), params.Y.min(), params.Y.max()], animated=True)
    quiv = ax.quiver(agents["pos"][:,0], agents["pos"][:,1],
                     np.cos(agents["thet"]), np.sin(agents["thet"]),
                     scale=10, color='black', width=0.003, scale_units='xy')
    scat = ax.scatter(agents["pos"][:,0], agents["pos"][:,1],
                      c=['blue' if f else 'orange' for f in agents["isForager"]],
                      s=20, edgecolors='k')

    time_text = ax.text(0.02, 0.95, f"t = 0.00", transform=ax.transAxes)
    plt.tight_layout()

    def update(frame):
        # perform a few sim steps per frame to speed wall-clock
        for _ in range(10):
            sim_step()
        # update visuals
        im.set_data(c.T)

        pos = agents["pos"]
        scat.set_offsets(pos)
        quiv.set_offsets(pos)
        quiv.set_UVC(np.cos(agents["thet"]), np.sin(agents["thet"]))

        colors = []
        for idx in range(pos.shape[0]):
            colors.append("limegreen" if agents["laying"][idx] else ("blue" if agents["isForager"][idx] else "orange"))
        scat.set_facecolors(colors)
        time_text.set_text(f"t = {update.t:.2f}")
        update.t += params.dt
        return im, scat, quiv, time_text
    update.t = 0.0

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval, blit=True)
    plt.close()
    return ani

#===============================================================================
#Run
#===============================================================================

if __name__ == '__main__':
	params = SimParams()
	kernel = make_gaussian_kernel(sigma=params.sigma, h=params.h)
	
	#Field  --------------------------------------------------------------------
	A = LinearOperator((params.Nx*params.Ny, params.Nx*params.Ny), matvec=matvec, dtype=float)  #Matrix Operator to implement Implicit Euler Integration
	c = set_initial_semicircle()  #Setup Initial Trail
	cvec = c.ravel()
	
	#Setup Agents
	w = 20
	f = 6
	agents = initialize_agents(w, f)

	ani = animate_sim(steps=500, interval=50)
	plt.show()
