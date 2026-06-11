import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob, os
import SFI
import SFI.OLI_bases
import jax.numpy as jnp
from jax import random
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import pickle
from scipy.spatial.distance import jensenshannon
import csv

def get_experimentID_fightbouts(path):

    tracking_folder = os.path.dirname(path)

    loadpaths = glob.glob(os.path.join(tracking_folder, "*results.h5"))
    loadpaths.sort()

    expNames = [os.path.basename(p)[:23] for p in loadpaths]

    target_expName = os.path.basename(path)[:23]
    expIdx = expNames.index(target_expName)

    fightbout_path = os.path.join(tracking_folder, "fightBouts.h5")

    with h5py.File(fightbout_path, "r") as j:
        fb = j["fight_bout_info_noDurThresh"][:]

    fightbouts = fb[fb[:, 0].astype(int) == expIdx]


    return expIdx, fightbouts
#EXP_id , fightbout = get_experimentID_fightbouts( "/Users/marindevandijk/Documents/CLS 2025/Thesis/Coding/ZebraFish_project/Data/tracking_results/FishTank20200213_154940_tracking_results.h5")

def prepare_data(path,fightnumber = 0,infight =True):
    "Prepare the data make it ready to calculate dpp,theta1 and theta2"
    "if infight = True return data with only the frist infight bouts otherwise it returns total trajectory  "

    path = path
    f = h5py.File(path, "r")

    X = f["tracks_3D_smooth"][:]
    EXP_id , fightbout = get_experimentID_fightbouts(path)
    if infight == True and fightbout.size > 0:
        X_coordinates = X[fightbout[fightnumber,1]:fightbout[fightnumber,2],:,:,:]
    else:
        X_coordinates = X.copy()
    return X_coordinates,fightbout[fightnumber], EXP_id


def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def calculate_theta(fish0,fish1):
    vector_fish0 = (fish0[:,0,:] - fish0[:,1,:]) # difference in heading of head and pec
    orientation_fish0 = vector_fish0/np.linalg.norm(vector_fish0, axis=1, keepdims=True)

    theta0 = np.arctan2(orientation_fish0[:, 1],  orientation_fish0[:, 0])

    r_01 = fish1[:,1,:] - fish0[:,1,:] 
    phi_01 = np.arctan2(r_01[:,1],r_01[:,0]) #dy,dx

    theta0 = np.arctan2(vector_fish0[:, 1], vector_fish0[:, 0])
    psi_01 = phi_01 - theta0
    psi_01 = wrap_pi(psi_01) # wrap interval from -pi to pi
    return psi_01

def calculate_variables(coordinates_cleaned):
    "input are the coordinates and returns dpp,theta_i and theta_j"

    fish_i = coordinates_cleaned[:,0,:,:]
    fish_j = coordinates_cleaned[:,1,:,:]

    pec_fish_i = coordinates_cleaned[:,0,1,:]
    pec_fish_j = coordinates_cleaned[:,1,1,:]

    d_pp = np.linalg.norm((pec_fish_j-pec_fish_i), axis=1) 
    
    theta_i = calculate_theta(fish_i,fish_j)
    theta_j = calculate_theta(fish_j,fish_i)

    return d_pp, theta_i,theta_j
    
def clean_data(d_pp,theta_i,theta_j):
    mask = (np.isfinite(d_pp) &np.isfinite(theta_i) & np.isfinite(theta_j))
    return d_pp[mask],theta_i[mask],theta_j[mask]


def Build_segmented_data(dpp,theta1,theta2):
    Valid = (np.isfinite(dpp) &np.isfinite(theta1) & np.isfinite(theta2))
    valid_idx = np.where(Valid)[0]

    X_list = []
    time_list = []
    segid_list = []
    seg_ranges = []
    start = 0
    t_offset = 0
    seg_id = 0

    for k in range(1, len(valid_idx) + 1):
        if k == len(valid_idx) or valid_idx[k] != valid_idx[k - 1] + 1: # if there is the end of trajectory or a hole 

            seg_idx = valid_idx[start:k]

            if len(seg_idx) > 5:
                D_seg = dpp[seg_idx]
                th1_seg = np.unwrap(theta1[seg_idx])
                th2_seg = np.unwrap(theta2[seg_idx])

                X_list.append(np.column_stack([D_seg, th1_seg, th2_seg]))
                time_list.append(np.arange(len(seg_idx)) + t_offset)
                segid_list.append(np.full(len(seg_idx), seg_id, dtype=int))
                seg_ranges.append((seg_idx[0], seg_idx[-1]))

                t_offset += len(seg_idx) + 1
                seg_id += 1

            start = k

    X = np.vstack(X_list)
    time_idx = np.concatenate(time_list)
    segment_ids = np.concatenate(segid_list)
    return X, time_idx, segment_ids, seg_ranges

def subsample_random_segments(X, segment_ids, fraction=0.85):
    np.random.seed(5)
    unique_segments = np.unique(segment_ids)
    n_keep = int(fraction * len(unique_segments))
    keep_seg = np.random.choice(unique_segments,size=n_keep,replace= False)

    X_list = []
    time_list = []
    t_offset = 0
    for seg in keep_seg:
        indices = np.where(segment_ids == seg)[0]
        X_seg = X[indices]
        X_list.append(X_seg)
        time_list.append(np.arange(len(X_seg)) + t_offset)
        t_offset += len(X_seg) + 1

    X_new = np.vstack(X_list)
    time_idx_new = np.concatenate(time_list)
    return X_new, time_idx_new

def js_score(real, sim, bins, range_):
    hist_real, bin_edges = np.histogram(real, bins=bins, range=range_)
    hist_sim, _ = np.histogram(sim, bins=bin_edges, range=range_)

    p = hist_real / np.sum(hist_real)
    q = hist_sim / np.sum(hist_sim)

    return jensenshannon(p, q)

def average_js_score(real_dpp, real_t1, real_t2, traj_sim):
    score_dpp = js_score(real_dpp, np.array(traj_sim[:, 0]), bins=50, range_=(0, 30))
    score_t1  = js_score(real_t1,  np.array(traj_sim[:, 1]), bins=50, range_=(-np.pi, np.pi))
    score_t2  = js_score(real_t2,  np.array(traj_sim[:, 2]), bins=50, range_=(-np.pi, np.pi))
    return (score_dpp + score_t1 + score_t2) / 3

def Run_Force_inference(X,time_idx,K,M,lam):
    traj = SFI.StochasticTrajectoryData(X, time_idx, 0.01)
    poly_1d,poly_describe = SFI.OLI_bases.polynomial_basis(dim=1,order=K)
    fourier1d_F1 = SFI.OLI_bases.Fourier_basis(dim =1,order=M,center= jnp.array([0.0]),width = jnp.array([2*jnp.pi]))
    fourier1d_F2 = SFI.OLI_bases.Fourier_basis(dim =1,order=M,center = jnp.array([0.0]),width = jnp.array([2*jnp.pi]))
    
    def radial_basis(D):

        return jnp.exp(-D / lam)#p_exp #jnp.concatenate([p_poly,p_exp])

    def smooth_gate(z, sharpness=8.0):
        return 0.5 * (1.0 + jnp.tanh(sharpness * z))


    def C_function2(x):
        D  = x[0]
        th1 = x[1]
        th2 = x[2]
        k =1.7

        p =  radial_basis(D) 
        f1 = jnp.tanh(k*jnp.sin(th1))
        f2 = jnp.tanh(k *jnp.sin(th2))
        a1 = jnp.abs(wrap_pi(th1))
        a2 = jnp.abs(wrap_pi(th2))

        s = 6

        front1 = smooth_gate((jnp.pi/4) - a1, s)
        back1  = smooth_gate(a1 - (3*jnp.pi/4), s)
        side1  = 1.0 - front1 - back1

        front2 = smooth_gate((jnp.pi/4) - a2, s)
        back2  = smooth_gate(a2 - (3*jnp.pi/4), s)
        side2  = 1.0 - front2 - back2

        ang = jnp.array([
        1.0,
        f1,
        f2,
        front2 * f1,
        back2  * f1,
        front1 * f2,
        back1  * f2,
        #side1 * f2,
        #side2  * f1,

        jnp.sin(2.0 * th1),
        jnp.sin(2.0 * th2),
        jnp.sin(3.0 * th1),
        jnp.sin(3.0 * th2),
        jnp.cos(th1),
        jnp.cos(th2),
    ])

        phi = jnp.einsum("i,j->ij", p, ang).reshape(-1)
        return phi
        
    S = SFI.OverdampedLangevinInference(traj)
    S.compute_diffusion_constant(method="MSD")
    (funcs_and_grad, descriptor) = SFI.OLI_bases.basis_selector(
        {"type": "custom_scalar", "functions": C_function2},
        dimension=3,
        output="vector"
    )
  
    basis_F, grad_F = funcs_and_grad
    S.infer_force_linear(basis_linear=basis_F, basis_linear_gradient=grad_F)
    #S.sparsify_force()
    S.compute_force_error() 
    S.print_report()
    return S, descriptor

def endpoint_clustering(all_endpoints):
    D_values = np.unique(all_endpoints[:,0])
    clustered_all = []

    for D in D_values:
        pts = all_endpoints[all_endpoints[:,0] == D]
        
        rounded = np.round(pts[:,1:], 3)
        unique_angles = np.unique(rounded, axis=0)
        
        clustered = np.column_stack([np.full(len(unique_angles), D), unique_angles])
        clustered_all.append(clustered)

    clustered_all = np.vstack(clustered_all)
    return clustered_all

def cluster_endpoints_3d(all_endpoints, decimals=3):
    rounded = np.round(all_endpoints, decimals=decimals)
    clustered = np.unique(rounded, axis=0)
    return clustered

def Simulation_deterministic(S,x0,dt,N_steps,force_tol,n_consecutive = 20,D= None,theta1 =None, theta2 = None,early_stop= True):
    x = jnp.array(x0)
    xs = []
    converged_count = 0

    for step in range(N_steps):
        xs.append(x)

        drift = S.force_ansatz(x[None, :])[0]
        x = x + drift * dt

        x = x.at[0].set(D if D is not None else jnp.clip(x[0], 0.0, 20.0))
        x = x.at[1].set(theta1 if theta1 is not None else wrap_pi(x[1]))
        x = x.at[2].set(theta2 if theta2 is not None else wrap_pi(x[2]))

        if early_stop:
            force_norm = np.linalg.norm(np.array(S.force_ansatz(x[None, :])[0]))

            if force_norm < force_tol:
                converged_count += 1
            else:
                converged_count = 0

            if converged_count >= n_consecutive:
                xs.append(x) 
                break
    return jnp.stack(xs), (step+1)

def Find_endpoints(S_model,X_data,outdir,tag="model",save_last_n= 3000):
    accept_rate = []
    all_endpoints =[]
    startpoints = []
    all_forces = []
    last_trajs = []
    D_values = np.linspace(1, 8, 15)
    length = np.linspace(-np.pi, np.pi, 16,endpoint = False)

    outpath = os.path.join(outdir, f"Endpoints_{tag}.csv")
    n_start = 3840
    startpoints_data = X_data[np.all(np.isfinite(X_data), axis=1)]

    replace = len(startpoints_data) < n_start
    idx = np.random.choice(len(startpoints_data), size=n_start, replace=replace)

    outpath = os.path.join(outdir, f"Endpoints_{tag}.csv")

    accepted = 0

    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "model",
            "d0", "theta10", "theta20",
            "d_final", "theta1_final", "theta2_final",
            "F_d", "F_theta1", "F_theta2", "step_used"
        ])

        for i in idx:
            x0 = startpoints_data[i].copy()

            # keep angles in normal range
            x0[1] = wrap_pi(x0[1])   # thetaW
            x0[2] = wrap_pi(x0[2])   # thetaL

            traj_sim, step = Simulation_deterministic(
                S_model,
                x0,
                dt=0.01,
                N_steps=5000,
                force_tol=1e-3,
                n_consecutive=20,
                D=None,
                theta1=None,
                theta2=None,
                early_stop=True
            )

            traj_np = np.array(traj_sim)

            if len(traj_np) >= save_last_n:
                last_part = traj_np[-save_last_n:]
            else:
                last_part = np.full((save_last_n, 3), np.nan)
                last_part[-len(traj_np):] = traj_np

            final_point = traj_np[-1]
            force = np.array(S_model.force_ansatz(final_point[None, :])[0])

            last_trajs.append(last_part.astype(np.float32))
            all_endpoints.append(final_point)
            all_forces.append(force)
            startpoints.append(x0)

            writer.writerow([
                tag,
                x0[0], x0[1], x0[2],
                final_point[0], final_point[1], final_point[2],
                force[0], force[1], force[2],
                step
            ])
            f.flush()
            accepted += 1

    accept_rate.append(accepted / n_start)

    all_endpoints = np.array(all_endpoints)
    all_forces = np.array(all_forces)
    startpoints = np.array(startpoints)
    last_trajs = np.array(last_trajs)

    np.savez_compressed(
        os.path.join(outdir, f"endpoint_{save_last_n}_trajs_{tag}.npz"),
        last_trajs=last_trajs,
        startpoints=startpoints,
        endpoints=all_endpoints
    )
    return all_endpoints, all_forces, startpoints, accept_rate


def Simulation(S_model,x0,dt,N_steps,key):
    Diffusion = np.array(S_model.diffusion_average)
    L = jnp.linalg.cholesky(Diffusion)
    x = jnp.array(x0)
    xs = []
    for _ in range(N_steps):
        xs.append(x)
        drift = S_model.force_ansatz(x[None, :])[0] 
        key, subkey = random.split(key)
        xi = random.normal(subkey, (3,))

        x = x + drift * dt + jnp.sqrt(2*dt) *  (L @ xi)
        
        x = x.at[0].set(jnp.clip(x[0], 0.0, 30))  
        x = x.at[1].set(wrap_pi(x[1]))
        x = x.at[2].set(wrap_pi(x[2]))

    return jnp.stack(xs), key

def make_winner_df(tracking_folder):
    other_info_loadpath = os.path.join(tracking_folder, "winners_losers_inconclusive.h5")

    with h5py.File(other_info_loadpath, "r") as hf:
        winner_idxs = np.array(hf["winner_idxs"][:])
        conclusive = np.array(hf["conclusive_winner_loser"][:])
        already_established = np.array(hf["already_established_dominance"][:])

    df = pd.DataFrame({
        "EXP_id": np.arange(len(winner_idxs)),
        "winnerIdx": winner_idxs.astype(int),
        "conclusive": conclusive.astype(bool),
        "already_established": already_established.astype(bool),
    })

    df = df[~df["EXP_id"].isin([4, 9, 16, 21])]
    return df


path_2 = "Data/tracking_results/FishTank20200130_153857_tracking_results.h5"
path_3 ="Data/tracking_results/FishTank20200130_181614_tracking_results.h5"
path_5 = "Data/tracking_results/FishTank20200213_154940_tracking_results.h5"
#path_7 = "Data/tracking_results/FishTank20200217_160052_tracking_results.h5"
path_8 = "Data/tracking_results/FishTank20200218_153008_tracking_results.h5"
path_10 = "Data/tracking_results/FishTank20200327_154737_tracking_results.h5"
path_12 = "Data/tracking_results/FishTank20200331_162136_tracking_results.h5"
path_13 ="Data/tracking_results/FishTank20200520_152810_tracking_results.h5"
path_15 = "Data/tracking_results/FishTank20200525_161602_tracking_results.h5"
path_18 = "Data/tracking_results/FishTank20200824_151740_tracking_results.h5"
path_19 = "Data/tracking_results/FishTank20200828_155504_tracking_results.h5"
path_20 = "Data/tracking_results/FishTank20200902_160124_tracking_results.h5"


paths = {
    2: path_2,
    3: path_3,
    5: path_5,
    8: path_8,
    10: path_10,
    12: path_12,
    13: path_13,
    15: path_15,
    18: path_18,
    19: path_19,
    20: path_20,
}

tracking_folder = os.path.dirname(path_2)
winner_df = make_winner_df(tracking_folder)

experiments = [2,3,5,10,12,13,15,18,19,20]

window_after = 30000

X_full_list = []
segment_ids_full_list = []
time_idx_full_list = []

seg_offset_full = 0
time_offset_full = 0

for exp in experiments:
    path = paths[exp]

    # Load full recording, not only the fightbout
    X_all, fightbout, exp_id = prepare_data(path, fightnumber=0, infight=False)

    # Take 25k frames after the first fightbout
    fight_end = int(fightbout[2])
    X_coordinates = X_all[fight_end:fight_end + window_after]


    winner_row = winner_df[winner_df["EXP_id"] == exp_id]
    id_winner = int(winner_row["winnerIdx"].iloc[0])

    if id_winner == 1:
        X_coordinates = X_coordinates[:, [1, 0], :, :]

    dpp, theta1, theta2 = calculate_variables(X_coordinates)

    X_seg, time_idx_seg, segment_ids_seg, seg_ranges = Build_segmented_data(
        dpp, theta1, theta2
    )

    time_idx_seg = time_idx_seg - time_idx_seg[0]

    X_full_list.append(X_seg)
    segment_ids_full_list.append(segment_ids_seg + seg_offset_full)
    time_idx_full_list.append(time_idx_seg + time_offset_full)

    seg_offset_full += segment_ids_seg.max() + 1
    time_offset_full += time_idx_seg.max() + 1


X_full = np.vstack(X_full_list)
segment_ids_full = np.concatenate(segment_ids_full_list)
time_idx_full = np.concatenate(time_idx_full_list)

dpp_full = X_full[:, 0]
theta1_full = wrap_pi(X_full[:, 1])
theta2_full = wrap_pi(X_full[:, 2])


print("X_full:", X_full.shape)
print("time_idx_full:", time_idx_full.shape)
print("segments full:", len(np.unique(segment_ids_full)))
print("D range:", np.nanmin(dpp_full), np.nanmax(dpp_full))


q01, q50, q95 = np.percentile(dpp_full, [1, 50, 95])
lam_full = jnp.array([q01, q50, q95])
print(lam_full)
lam_common = jnp.array([ 1.2914723, 7.1262345 ,17.001316 ])

print('lambda value:', lam_common)

base_dir = os.environ.get("SLURM_SUBMIT_DIR", os.getcwd())
outdir = os.path.join(base_dir, "Results_reproduce", "All_full_25%_afterfight_weighted_2")
os.makedirs(outdir, exist_ok=True)


S_full, descriptor_full = Run_Force_inference(
    X_full,
    time_idx_full,
    K=3,
    M=4,
    lam=lam_common)

key = random.PRNGKey(0)

i_full = np.random.randint(0, len(X_full))
x0_full = X_full[i_full]

traj_sim_full, key = Simulation(
    S_full,
    x0_full,
    dt=0.01,
    N_steps=500000,
    key=key
)

traj_sim_full_np = np.array(traj_sim_full)

np.savez(
    os.path.join(outdir, "stochastic_simulated_trajectory_full.npz"),
    traj_sim_full=traj_sim_full_np,
    x0_full=x0_full,
)

fig, axs = plt.subplots(1, 3, figsize=(15,4))

axs[0].hist(traj_sim_full_np[:,0], alpha=0.5, density=True, label="sim full", bins=100)
axs[0].hist(X_full[:,0], alpha=0.5, density=True, label="real full", bins=100)
axs[0].legend()
axs[0].set_title(r"$d_{pp}$")

axs[1].hist(wrap_pi(traj_sim_full_np[:,1]), density=True, alpha=0.5, label="sim full", bins=100)
axs[1].hist(wrap_pi(X_full[:,1]), density=True, alpha=0.5, label="real full", bins=100)
axs[1].set_title(r"$\theta_W$")

axs[2].hist(wrap_pi(traj_sim_full_np[:,2]), density=True, alpha=0.5, label="sim full", bins=100)
axs[2].hist(wrap_pi(X_full[:,2]), density=True, alpha=0.5, label="real full", bins=100)
axs[2].set_title(r"$\theta_L$")

plt.tight_layout()
fig_path = os.path.join(outdir, "stochastic_simulation_full_distribution.png")
plt.savefig(fig_path, dpi=300)
plt.close()


js_full = average_js_score(
    X_full[:,0],
    wrap_pi(X_full[:,1]),
    wrap_pi(X_full[:,2]),
    traj_sim_full_np
)

print("JS full:", js_full)

all_endpoints_full, all_forces_full, startpoints_full, accept_rate_full = Find_endpoints(
    S_full,X_full,
    outdir,
    tag="full",
    save_last_n=3000
)

with open(os.path.join(outdir, "metadata.txt"), "w") as f:
    f.write("FULL DATASET MODEL\n")
    f.write(f"experiments: {experiments}\n")
    f.write("\n")

    f.write(f"X_full shape: {X_full.shape}\n")
    f.write(f"time_idx_full shape: {time_idx_full.shape}\n")
    f.write(f"segments full: {len(np.unique(segment_ids_full))}\n")
    f.write("\n")

    f.write("lambda_full:\n")
    f.write(f"{np.array(lam_common)}\n")
    f.write("\n")

    f.write("Jensen-Shannon score:\n")
    f.write(f"JS_full: {js_full}\n")

def save_sfi_model(S_model, descriptor, outdir, tag):

    save_dict = {
        "force_coefficients": np.array(S_model.phi),
        "diffusion_tensor": np.array(S_model.diffusion_average),
        "force_error": np.array(S_model.force_error),
        "descriptor": descriptor,
    }

    np.savez(
        os.path.join(outdir, f"SFI_model_data_{tag}.npz"),
        **save_dict
    )

save_sfi_model(S_full, descriptor_full, outdir, "full")

