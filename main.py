import os
import random
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, linregress
from scipy.stats import norm
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression                        
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.patheffects as pe
os.makedirs('csv', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)
os.system('cls' if os.name == 'nt' else 'clear')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
dpi= 150
def load_Case_Studies(id):
    p = SimpleNamespace()
    extra_data_cols = []
    extra_feature_cols = [ ]
    solvent_name_cas_lookup = []
    solute_props = []
    solvent_props = []
    if id == 1:
        df  = pd.read_csv("yinf_clean_dataset.csv")
        df = df[df["sys_type"]==1]
        df = df[df["ref"]!="Rabhi (2019) - Journal of Molecular Liquids 283 (2019) 380–390"]        
        extra_data_cols = ["ref", "year" ,"family"]     
        df = df.sort_values(by=['solute', 'solvent_name', 'temp'])
        corrections = [
            dict(solute="Cyclohexane",
                solvent_name="nonanediol",
                ref="Alessi (1984) - Chemical Engineering Communications (1984), 27(1-2), 59-67",
                temp=65.7+273.15,      # put the exact target temp here (K)
                y_inf=np.exp(1.57),
                note="TABLE III, Solutes=7 @ 65.7 C, id = 30595"),

            dict(solute="1,2-dichloroethane",
                solvent_name="n-icosane",
                ref="Martire (1965) - Journal of Chemical and Engineering Data 10.1 (1965): 40-43.",
                temp=74.1+273.15,
                y_inf=1.390,
                note="TABLE III, Solvent=A @ 65.7 C, id = 64766"),

            dict(solute="1,1-dichloroethane",
                solvent_name="n-icosane",
                ref="Martire (1965) - Journal of Chemical and Engineering Data 10.1 (1965): 40-43.",
                temp=74.1+273.15,
                y_inf=1.061,
                note="TABLE III, Solvent=A @ 65.7 C, id = 64765"),

            dict(solute="Dichloromethane",
                solvent_name="toluene",
                ref="Landau (1991) - Ind. Eng. Chem. Res. 1991, 30, 1900-1906",
                temp=20+273.15,                 
                y_inf=0.91,
                note="Table VIII. Limiting Activity Coefficients in Toluene @ 20 C, id = 51820"),

            dict(solute="Diethyl ether",
                solvent_name="Tetraethylene glycol dimethyl ether",
                ref="Schiller (1992) - J. Chem. Eng. Data 1992,37,503-508",
                temp=306.05,
                y_inf=1.461,
                note="Table I @ 303.15 K, Exact temperature T = 306.05 K., id = 52436"),

            dict(solute="Methyl acetate",
                solvent_name="water",
                ref="Sherman (1996) - Ind. Eng. Chem. Res. 1996, 35, 1044-1058",
                temp=25+273.15,
                y_inf=22.6,
                note="Table 5 @ 25 C , id = 52951"),
  
            dict(solute="Methyl formate",
                solvent_name="water",
                ref="Sherman (1996) - Ind. Eng. Chem. Res. 1996, 35, 1044-1058",
                temp=298.15,
                y_inf=15.5,
                note="Table 5 @ 25 C , id = 52950"),
           
            dict(solute="1-pentanol",
                solvent_name="water",
                ref="Sherman (1996) - Ind. Eng. Chem. Res. 1996, 35, 1044-1058",
                temp=298.15,
                y_inf=198,
                note="Table 5 @ 25 C , id = 52939"),
            
            dict(solute="Methyl propanoate",
                solvent_name="water",
                ref="Kojima (1997) - Fluid Phase Equilibria 131 (1997) 145-179",
                temp=293.15,
                y_inf=8.71,
                note="***"),

            dict(solute="1,2-dichlorobenzene",
                solvent_name="water",
                ref="Hovorka (1997) - J. Chem. Eng. Data 1997, 42, 924-933",
                temp=292.65,
                y_inf=63200.0,
                note="Table 4, id = 64432"),

            dict(solute="Benzene",
                solvent_name="n-hexane",
                ref="Li (1990) - Fluid Phase Equilibria\xa058.3 (1990): 307-318.",
                temp=298.15,
                y_inf=np.exp(0.3889),
                note=""),  
            ]
 
        def apply_corrections(df, corrections, temp_tol=1e-6, verbose=True):
            for k, c in enumerate(corrections, start=1):
                mask = (
                    (df["solute"] == c["solute"]) &
                    (df["solvent_name"] == c["solvent_name"]) &
                    (df["ref"] == c["ref"]) &
                    (np.isclose(df["temp"].astype(float), float(c["temp"]), atol=temp_tol))
                )

                matched = df.index[mask]
                if len(matched) == 0:
                    print(f"[{k}] NO MATCH: {c['solute']} | {c['solvent_name']} | {c['temp']} | {c['ref']}")
                    continue

                if len(matched) > 1:
                    print(f"[{k}] WARNING: {len(matched)} matches; updating ALL. Indexes: {matched.tolist()}")

                if verbose:
                    before = df.loc[mask, "y_inf"].copy()

                df.loc[mask, "y_inf"] = c["y_inf"]

                if verbose:
                    after = df.loc[mask, "y_inf"]
                    note = c.get("note", "")
                    
            return df

        df = apply_corrections(df, corrections)

        known_solutes ={"2 TBP + UO2(NO3)2"}
        known_solutes = set(known_solutes) 
        df = df.loc[~df["solute"].isin(known_solutes)].copy()

        df.rename(
            columns={
                "y_inf": "ln_IDAC_exp",
                "temp": "T [K]",
                "solute": "Solute",
                "solvent_name": "Solvent",  
            },
            inplace=True,
        )
 
        df["ln_IDAC_exp"] = np.log(df["ln_IDAC_exp"]  ) 
        solvent_name_cas_lookup = dict(
        zip(df["Solvent"], df["solvent_cas"]))
                          
        p.BETA_VAE = 1e-4 
        p.EMBED_DIM = 64*4*2 
        p.RECON_LOSS_WEIGHT = 0.5      
        p.HIDDEN_DIM_AE = [32, 8, 4]
        p.HIDDEN_DIM_Regressor = [32, 8, 4]
        p.LATENT_DIM = 4 

    elif id == 2:

        '''
        df = pd.read_excel(
            "ie4c00107_si_001.xlsx", 
            sheet_name="Table S3",
            skiprows=2,      # Skips the first 2 rows (0 and 1), so it starts at row 3
            header=0         # Tells pandas that the first row it sees (which was row 3) is the header
        )
        df = df.iloc[:21284, 1:] # This removes the first column (Column A)
        
        df.rename(
            columns={
                "Experimental value": "ln_IDAC_exp",
                "T": "T [K]",
                "Solute Smiles": "Solute",
                "Solvent Smiles": "Solvent",
   
            },
            inplace=True,
        )
        '''
        # # --- SAVE ---
        # df.to_pickle('df_case_2_raw.pkl')
        # --- LOAD ---
        df = pd.read_pickle('df_case_2_raw.pkl')
        df_Solute_Familty = pd.read_excel(
            "ie4c00107_si_001.xlsx", 
            sheet_name="Table S4",
            skiprows=1,       
            header=0          
        )
        df_Solvent_Familty = pd.read_excel(
            "ie4c00107_si_001.xlsx", 
            sheet_name="Table S5",
            skiprows=1,       
            header=0     
        )
        solute_props = df_Solute_Familty.rename(columns={
            'SMILES0': 'Solute',
            'Family': 'Solute_Family',
            'MolWt': 'Solute_MolWt',
            'IUPAC name': 'Solute_IUPAC_name'
        })

        solvent_props = df_Solvent_Familty.rename(columns={
            'SMILES1': 'Solvent',
            'Family': 'Solvent_Family',
            'MolWt': 'Solvent_MolWt',
            'IUPAC name': 'Solvent_IUPAC_name'
        })

        df = df.merge(
            solute_props[['Solute', 'Solute_Family', 'Solute_MolWt', 'Solute_IUPAC_name']],
            on='Solute',
            how='left'
        )

        df = df.merge(
            solvent_props[['Solvent', 'Solvent_Family', 'Solvent_MolWt', 'Solvent_IUPAC_name']],
            on='Solvent',
            how='left'
        )

        p.BETA_VAE = 1e-4
        p.EMBED_DIM = 64*4*2
        p.RECON_LOSS_WEIGHT = 0.5
        p.HIDDEN_DIM_AE = [32, 8, 4]
        p.HIDDEN_DIM_Regressor = [32, 8, 4]
        p.LATENT_DIM = 4 
        
    else:
        raise ValueError(f"Unsupported case study id: {id}")
    ##############################################
    df = pd.concat([df, df[extra_feature_cols]], ignore_index=True) if extra_feature_cols else df
    df = pd.concat([df, df[extra_data_cols   ]], ignore_index=True) if extra_data_cols    else df
    expected_cols = ["Solute", "Solvent", "T [K]", "ln_IDAC_exp"]
    expected_cols.extend(extra_feature_cols)
    expected_cols.extend(extra_data_cols)
    for c in expected_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found. Found columns: {list(df.columns)}")

    df = df[expected_cols].copy()
    df = df.dropna().reset_index(drop=True)
    df = df.sort_values( by=["Solute", "Solvent", "T [K]"], ascending=[True, True, True] )
    
    return df, p ,extra_feature_cols ,extra_data_cols,solvent_name_cas_lookup,solute_props,solvent_props

def high_error_plots(df_latent_analysis, ERROR_THRESHOLD): 
    df_latent_analysis['Is_High_Error'] = df_latent_analysis['Residual (Exp - Pred)'].abs() >= ERROR_THRESHOLD
    df_high_error = df_latent_analysis[df_latent_analysis['Is_High_Error']].copy()
    global_solute_counts = df_latent_analysis['Solute'].value_counts()
    global_solvent_counts = df_latent_analysis['Solvent'].value_counts()
    df_high_error['Solute_Freq_as_Solute'] = df_high_error['Solute'].map(global_solute_counts).fillna(0).astype(int)
    df_high_error['Solute_Freq_as_Solvent'] = df_high_error['Solute'].map(global_solvent_counts).fillna(0).astype(int)  
    df_high_error['Solvent_Freq_as_Solute'] = df_high_error['Solvent'].map(global_solute_counts).fillna(0).astype(int)
    df_high_error['Solvent_Freq_as_Solvent'] = df_high_error['Solvent'].map(global_solvent_counts).fillna(0).astype(int)
    df_high_error.to_csv('csv/high_error_details.csv', index=False)
    print(f"Saved {len(df_high_error)} high-error cases to 'high_error_details.csv'")

def plot_0(datasets,df_latent_analysis,fold):
    def calc_metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return r2, mae, mse
    
    all_true = []
    all_pred = []
    for name, (df, color) in datasets.items():
        all_true.extend(df["Experimental Ln(IDAC)"].values)
        all_pred.extend(df["Predicted Ln(IDAC)"].values)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    r2_all, mae_all, mse_all = calc_metrics(all_true, all_pred)
    print(
        rf"Overall $R^2$ = {r2_all:.4f}" "\n"
        rf"MAE = {mae_all:.4f}" "\n"
        rf"MSE = {mse_all:.4f}"
    )
    for name, (df, color) in datasets.items():
        y_true = df["Experimental Ln(IDAC)"].values
        y_pred = df["Predicted Ln(IDAC)"].values
        r2, mae, mse = calc_metrics(y_true, y_pred)
        n = len(df) 
        print(f"{name} (n={n}, $R^2$={r2:.3f}, MAE={mae:.4f} , MSE={mse:.4f})")
    ######################################################
    df_latent_analysis['Binary_systems'] = df_latent_analysis['Solvent'] + '_' + df_latent_analysis['Solute']
    pair_counts = df_latent_analysis['Binary_systems'].value_counts()
    freq_counts = pair_counts.value_counts().sort_index()       
    mpl.rcParams.update({
        "font.family": "Times New Roman",   # or Arial
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 9,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": False,  
    })
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        freq_counts.index.astype(str),
        freq_counts.values,
        color="gray",
        edgecolor="black",
        linewidth=1.0
    )
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=9,
            color="red"
        )
    ax.set_xlabel("Number of occurrences of a binary system")
    ax.set_ylabel("Number of binary systems")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    max_height = freq_counts.values.max()
    ax.set_ylim(0, max_height * 1.15)
    plt.tight_layout()
    plt.savefig(
        f"plots/Frequency_of_Binary_System_Occurrences_Fold_{fold}.png",
        dpi=dpi, bbox_inches="tight"   )
    plt.close()
    ######################################################  
    plt.style.use('seaborn-v0_8-whitegrid')
    x = df_latent_analysis["Repeat_Count"]
    y = df_latent_analysis["Residual (Exp - Pred)"].abs()
    COLOR_SPINES = "#2B2D42"  
    COLOR_POINTS = "#1B4965"
    COLOR_LINE = "#E05638"
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    scatter = ax.scatter(
        x,
        y,
        alpha=0.25,
        s=22,  
        color=COLOR_POINTS,
        edgecolors="none",
        zorder=2,
        label=r"Individual $|\Delta \ln \gamma^{\infty}|$",
    )
    trend_max = (
        df_latent_analysis.groupby("Repeat_Count")["Residual (Exp - Pred)"]
        .apply(lambda v: v.abs().max())
        .sort_index()
    )
    (line,) = ax.plot(
        trend_max.index,
        trend_max.values,
        color=COLOR_LINE,
        linewidth=2.2,
        zorder=4,
        path_effects=[
            pe.Stroke(linewidth=3.2, foreground="white", alpha=0.8),
            pe.Normal(),
        ],
        label=r"Maximum $|\Delta \ln \gamma^{\infty}|$",
    )
    ax.set_xlabel("Number of occurrences of a binary system", labelpad=8)
    ax.set_ylabel(
        r"$|\ln(\gamma_{\mathrm{exp}}^{\infty}) - \ln(\gamma_{\mathrm{est}}^{\infty})|$",
        labelpad=10,
    ) 

    ax.set_xlim(-2, max(x) + 5)
    ax.set_ylim(-0.25, max(y) + 0.6)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=5,
        width=1.1,
        colors=COLOR_SPINES,
    )
    for spine in ax.spines.values():
        spine.set_color(COLOR_SPINES)
        spine.set_linewidth(1.1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    leg = ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#E2E8F0",
        facecolor="#FFFFFF",
        borderpad=0.8,
    )
    leg.legend_handles[0].set_alpha(1.0)  
    plt.tight_layout()
    plt.savefig(
        f"plots/residual_vs_Repeat_Count_Fold_{fold}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close()
    ###################################################### 
    df_1 = df_latent_analysis[df_latent_analysis['Repeat_Count'] == 1].copy()
    df_2 = df_latent_analysis[df_latent_analysis['Repeat_Count'] == 2].copy()
    df_3 = df_latent_analysis[df_latent_analysis['Repeat_Count'] == 3].copy()
    df_4_plus = df_latent_analysis[df_latent_analysis['Repeat_Count'] >= 4].copy()

    datasets1 = {
        '(a) Binary Pair Occurrence = 1': df_1,
        '(b) Binary Pair Occurrence = 2': df_2,
        '(c) Binary Pair Occurrence = 3': df_3,
        '(d) Binary Pair Occurrence ≥ 4': df_4_plus
    }
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks") 
    plt.rcParams['axes.edgecolor'] = 'black'     
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "mathtext.fontset": "cm",       
        "mathtext.rm": "serif",
        "axes.linewidth": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.color": "black",
        "ytick.color": "black",
    })
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",         
        "mathtext.rm": "serif",
    })
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), sharex=True, sharey=True)
    axes = axes.flatten()
    palette = {'Train': '#1f77b4', 'Validation': '#2ca02c', 'Test': '#ff7f0e'}
    global_min = df_latent_analysis[['Experimental Ln(IDAC)', 'Predicted Ln(IDAC)']].min().min()
    global_max = df_latent_analysis[['Experimental Ln(IDAC)', 'Predicted Ln(IDAC)']].max().max()
    padding = (global_max - global_min) * 0.05
    lims = [global_min - padding, global_max + padding]

    for i, (title, df_subset) in enumerate(datasets1.items()):
        ax = axes[i]
        ax.set_title(title, fontsize=18, pad=12, fontweight='bold')
        ax.tick_params(top=True, right=True)
        
        if df_subset.empty:
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            continue
        
        y_true = df_subset['Experimental Ln(IDAC)']
        y_pred = df_subset['Predicted Ln(IDAC)']
        n_points = len(df_subset)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        line1 = rf"n = {n_points}"
        line2 = rf"$R^2$ = {r2:.4f}"
        line3 = rf"MAE = {mae:.4f}"
        line4 = rf"MSE = {mse:.4f}"
        annotation_text = f"{line1}\n{line2}\n{line3}\n{line4}"

        sns.scatterplot(
            data=df_subset,
            x='Experimental Ln(IDAC)',
            y='Predicted Ln(IDAC)',
            hue='Split',
            palette=palette,
            ax=ax,
            s=70,
            alpha=0.75,
            edgecolor='black', 
            linewidth=0.6,
            zorder=2
        )

        ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.8, zorder=1)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.text(
            0.05, 0.95, annotation_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', 
            bbox=dict(boxstyle='square,pad=0.6', fc='white', ec='black', lw=1.2, alpha=0.9),
            zorder=3
        )

        split_counts = df_subset['Split'].value_counts()
        handles, labels = ax.get_legend_handles_labels()
        
        new_labels = []
        new_handles = []
        for handle, label in zip(handles, labels):
            if label == 'Split': 
                continue
            
            count = split_counts.get(label, 0)
            new_labels.append(rf"{label} ($n={count}$)")
            new_handles.append(handle)
            
        ax.legend(
            new_handles, new_labels, 
            loc='lower right', 
            fontsize=13, 
            frameon=True,
            edgecolor='black',
            facecolor='white',
            framealpha=1.0,
            fancybox=False 
        )
        
        if i in [2, 3]:
            ax.set_xlabel(r"$\ln(\gamma_{\mathrm{exp}}^{\infty})$", fontsize=18)
        else:
            ax.set_xlabel('')
            
        if i in [0, 2]:
            ax.set_ylabel(r"$\ln(\gamma_{\mathrm{est}}^{\infty})$", fontsize=18)
        else:
            ax.set_ylabel('')

    plt.tight_layout()
    output_filename = f"plots/parity_by_occurrence_2x2_Fold_{fold}.png"
    plt.savefig(output_filename, dpi=dpi, bbox_inches='tight') 
    plt.close()
    print(f"Figure saved to {output_filename}")
    ###################################################### 
    df_plot1 = df_latent_analysis 
    df_plot1 = df_plot1[df_plot1['Split'].isin(['Test', 'Validation'])]
    df_plot1['Residual'] = df_plot1['Experimental Ln(IDAC)'] - df_plot1['Predicted Ln(IDAC)']
    df_1 = df_plot1[df_plot1['Repeat_Count'] == 1].copy()
    df_2 = df_plot1[df_plot1['Repeat_Count'] == 2].copy()
    df_3 = df_plot1[df_plot1['Repeat_Count'] == 3].copy()
    df_4_plus = df_plot1[df_plot1['Repeat_Count'] >= 4].copy()

    datasets1 = {
        '(a) Binary Pair Occurrence = 1': df_1,
        '(b) Binary Pair Occurrence = 2': df_2,
        '(c) Binary Pair Occurrence = 3': df_3,
        '(d) Binary Pair Occurrence ≥ 4': df_4_plus
    }
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "axes.linewidth": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.color": "black",
        "ytick.color": "black",
    })
 
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",          
        "mathtext.rm": "serif",
    })

    plt.rcParams['axes.edgecolor'] = 'black'    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), sharex=False, sharey=False)
    axes = axes.flatten()
    palette = {'Train': '#1f77b4', 'Validation': '#2ca02c', 'Test': '#ff7f0e'}
    for i, (title, df_subset) in enumerate(datasets1.items()):
        ax = axes[i]
        ax.set_title(title, fontsize=18, pad=12, fontweight='bold')
        ax.tick_params(top=True, right=True)
        
        if df_subset.empty:
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            continue

        y_true = df_subset['Experimental Ln(IDAC)']
        y_pred = df_subset['Predicted Ln(IDAC)']
        n_points = len(df_subset)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        line1 = rf"n = {n_points}"
        line2 = rf"$R^2$ = {r2:.4f}"
        line3 = rf"MAE = {mae:.4f}"
        line4 = rf"MSE = {mse:.4f}"
        annotation_text = f"{line1}\n{line2}\n{line3}\n{line4}"
        sns.histplot(
            data=df_subset,
            x='Residual',
            hue='Split',
            palette=palette,
            multiple='stack',
            bins=50,
            ax=ax,
            edgecolor='black',
            linewidth=1.0,
            alpha=0.8,
            zorder=2
        )
        ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.8, zorder=3)
        ax.text(
            0.05, 0.95, annotation_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', 
            bbox=dict(boxstyle='square,pad=0.6', fc='white', ec='black', lw=1.2, alpha=0.9),
            zorder=4
        )
        split_counts = df_subset['Split'].value_counts()
        if ax.get_legend() is not None:
            handles = ax.get_legend().legend_handles
            labels = [text.get_text() for text in ax.get_legend().texts]
            
            new_labels = []
            new_handles = []
            for handle, label in zip(handles, labels):
                count = split_counts.get(label, 0)
                new_labels.append(f"{label} ($n={count}$)")
                new_handles.append(handle)
                
            ax.legend(
                new_handles, new_labels, 
                loc='upper right', 
                fontsize=13, 
                frameon=True,
                edgecolor='black',
                facecolor='white',
                framealpha=1.0,
                fancybox=False 
            )
        
        if i in [2, 3]:
            ax.set_xlabel(r"$\ln(\gamma_{\mathrm{exp}}^{\infty}) - \ln(\gamma_{\mathrm{est}}^{\infty})$", fontsize=18 )
        else:
            ax.set_xlabel('')
            
        if i in [0, 2]:
            ax.set_ylabel("Frequency", fontsize=18)
        else:
            ax.set_ylabel('')

    plt.tight_layout()

    output_filename = f"plots/residual_histograms_by_occurrence_2x2_Fold_{fold}.png"
    plt.savefig(output_filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {output_filename}")
    
    df_plot1_ = df_latent_analysis 
    df_plot1_ = df_plot1_[df_plot1_['Split'].isin(['Train'])]
    df_plot1_['Residual'] = df_plot1_['Experimental Ln(IDAC)'] - df_plot1_['Predicted Ln(IDAC)']
    df_1_ = df_plot1_[df_plot1_['Repeat_Count'] == 1].copy()
    datasets1_ = {
        '(a) Binary Pair Occurrence = 1': df_1_,
    }
    for i, (title, df_subset) in enumerate(datasets1_.items()):
        y_true = df_subset['Experimental Ln(IDAC)']
        y_pred = df_subset['Predicted Ln(IDAC)']
        n_points = len(df_subset)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        print(
        f"Tarin,Binary Pair Occurrence = 1, for manscript   | MSE: {mse:.4f} | "
        f"MAE: {mae:.4f} | "
        f"R2: {r2:.4f}"
         )
    ######################################################    
    def plot_error_histograms_pro(datasets, fold,save_prefix=""):

        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.0
        })
        fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.5), sharex=True)
        colors = {
            "Train": "#4C78A8",       # Muted Blue
            "Validation": "#54A24B",  # Muted Green
            "Test": "#F58518"         # muted orange
        }
        residuals = {}
        all_res = []
        for name, (df, _) in datasets.items():
            res = df["Experimental Ln(IDAC)"].values - df["Predicted Ln(IDAC)"].values
            residuals[name] = res
            all_res.extend(res)
        
        all_res = np.array(all_res)
        x_min = all_res.min()
        x_max = all_res.max()
        if x_min >= x_max:
            x_min, x_max = all_res.min(), all_res.max()
        bins = np.linspace(x_min, x_max, 100)
        x_vals = np.linspace(x_min, x_max, 800)
        panel_letters = ["(a)", "(b)", "(c)"]
        for i, (ax, (name, res)) in enumerate(zip(axes, residuals.items())):
            ax.hist(
                res,
                bins=bins,
                density=True,
                alpha=0.55,
                color=colors.get(name, "gray"),
                edgecolor="white",
                linewidth=0.7
            )
            kde = gaussian_kde(res, bw_method=0.20)
            ax.plot(x_vals, kde(x_vals), color="black", linewidth=2)
            ax.axvline(0, linestyle="--", color="#d62728", linewidth=1.5, alpha=0.8)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, None) 
            ax.text(-0.07, 1.08, panel_letters[i], transform=ax.transAxes, 
                    fontsize=16, fontweight='bold', va='top', ha='right')
            ax.text(0.02, 0.92, f"{name}", transform=ax.transAxes, 
                    fontsize=14, va='top')
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", direction="out", length=5, width=1.2)
            ax.tick_params(axis="x", labelbottom=True, labelsize=12, labelcolor='black')
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')

            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
            ax.tick_params(which="minor", axis="x", length=3, width=0.8)
            ax.tick_params(which="minor", axis="y", length=3, width=0.8)

        fig.supxlabel(r"$\ln(\gamma_{\mathrm{exp}}^{\infty}) - \ln(\gamma_{\mathrm{est}}^{\infty})$", fontsize=16,fontweight='bold')
        fig.supylabel("Density", fontsize=16)
        plt.subplots_adjust(hspace=0.35, left=0.12, right=0.95, bottom=0.10, top=0.95)
        fig.savefig(
            f"plots/{save_prefix}error_histograms_final_Fold_{fold}.png",
            dpi=dpi,
            bbox_inches="tight"
        )
        plt.close(fig)
    plot_error_histograms_pro(datasets, fold , save_prefix="")
    print("All Finished") 
##########################################################################
# Model Class
##########################################################################
class β_CVAE(nn.Module):
    def __init__(
        self,
        n_solutes,
        n_solvents,
        embed_dim=32,
        hidden_dim_enc_dec=[64,32],
        hidden_dim_regressor=[64,32],
        latent_dim=8,
        n_ef = 0,
    ):
        super().__init__()
        self.solute_emb = nn.Embedding(n_solutes, embed_dim)      
        self.solvent_emb = nn.Embedding(n_solvents, embed_dim)
        chem_dim = embed_dim * 4  # [sol, solv, abs_diff, hadamard]
        temp_dim = 1 
        in_dim = chem_dim + temp_dim + n_ef      
               
        # --- Encoder ---
        enc_layers = []
        curr_dim = in_dim
        for h_dim in hidden_dim_enc_dec:
            enc_layers.extend([nn.Linear(curr_dim, h_dim),   nn.Sigmoid()  ])
            curr_dim = h_dim
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(hidden_dim_enc_dec[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc_dec[-1], latent_dim)
       
        # Dimension for the latent variable combined with condition
        z_cond_dim = latent_dim + temp_dim + n_ef
          
        # --- Decoder ---
        dec_layers = []
        curr_dim = z_cond_dim
        for h_dim in reversed(hidden_dim_enc_dec):
            dec_layers.extend([nn.Linear(curr_dim, h_dim),   nn.Sigmoid()  ])
            curr_dim = h_dim
        dec_layers.append(nn.Linear(hidden_dim_enc_dec[0], chem_dim))
        self.decoder = nn.Sequential(*dec_layers)
        
        reg_layers = []
        curr_dim  = z_cond_dim
        for h_dim in hidden_dim_regressor:
            reg_layers.extend([nn.Linear(curr_dim , h_dim),   nn.Sigmoid()  ])
            curr_dim = h_dim
        reg_layers.append(nn.Linear(hidden_dim_regressor[-1], 2-1))
        self.regressor = nn.Sequential(*reg_layers)
 
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, solute_id, solvent_id, temperature, temp_scaler, extra_features=None):
        sol = self.solute_emb(solute_id)
        solv = self.solvent_emb(solvent_id)
        abs_diff = torch.abs(sol - solv)
        hadamard = sol * solv
        x_chem = torch.cat([sol, solv, abs_diff, hadamard], dim=1)
        temp_vec = temperature.unsqueeze(1)
        x_enc = torch.cat([x_chem, temp_vec], dim=1)

        if extra_features is not None:
            x_enc = torch.cat([x_enc, extra_features], dim=1)   

        h = self.encoder(x_enc)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        z_cond = torch.cat([z, temp_vec], dim=1)
        if extra_features is not None:
            z_cond = torch.cat([z_cond, extra_features], dim=1)  
        x_chem_recon = self.decoder(z_cond)
        y_pred = self.regressor(z_cond)
        return y_pred, x_chem_recon, x_chem, mu, logvar, z
##########################################################################
# Dataset Definition
##########################################################################
class IDACDataset(Dataset):
    def __init__(self, solute_ids, solvent_ids, temperatures, targets, indices, extra_features=None, extra_data=None):
        self.solute_ids = solute_ids[indices]
        self.solvent_ids = solvent_ids[indices]
        self.temperatures = temperatures[indices]
        self.targets = targets[indices]
        self.extra_features = extra_features[indices] if extra_features is not None else None
        self.extra_data = extra_data[indices] if extra_data is not None else None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        item = {
            "solute_id": torch.tensor(self.solute_ids[idx], dtype=torch.long),
            "solvent_id": torch.tensor(self.solvent_ids[idx], dtype=torch.long),
            "temperature": torch.tensor(self.temperatures[idx], dtype=torch.float32),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32)
        }
        if self.extra_features is not None:
            item["extra_features"] = torch.tensor(self.extra_features[idx], dtype=torch.float32)
        if self.extra_data is not None:
            item["extra_data"] = self.extra_data[idx]
        return item

def get_fold_dataloaders(
    df,
    solute_ids,
    solvent_ids,
    raw_temperatures,
    batch_size,
    train_idx,
    val_idx,
    test_idx,
    target_transformer,
    extra_features=None,
):

    target_raw = df["ln_IDAC_exp"].to_numpy(dtype=np.float32)
    target_scaled = np.empty(len(df), dtype=np.float32)
    target_scaled[train_idx] = target_transformer.fit_transform(
        target_raw[train_idx].reshape(-1, 1)
    ).ravel().astype(np.float32)

    target_scaled[val_idx] = target_transformer.transform(
        target_raw[val_idx].reshape(-1, 1)
    ).ravel().astype(np.float32)

    target_scaled[test_idx] = target_transformer.transform(
        target_raw[test_idx].reshape(-1, 1)
    ).ravel().astype(np.float32)

    temp_scaler = StandardScaler()
    temperatures_scaled = np.empty(len(df), dtype=np.float32)

    temperatures_scaled[train_idx] = temp_scaler.fit_transform(
        raw_temperatures[train_idx].reshape(-1, 1)
    ).ravel().astype(np.float32)

    temperatures_scaled[val_idx] = temp_scaler.transform(
        raw_temperatures[val_idx].reshape(-1, 1)
    ).ravel().astype(np.float32)

    temperatures_scaled[test_idx] = temp_scaler.transform(
        raw_temperatures[test_idx].reshape(-1, 1)
    ).ravel().astype(np.float32)

    extra_feature_scaler = None
    extra_features_scaled = None

    if extra_features is not None:
        if isinstance(extra_features, pd.DataFrame):
            extra_features_raw = extra_features.to_numpy(dtype=np.float32)
        else:
            extra_features_raw = np.asarray(
                extra_features, dtype=np.float32
            )

        extra_feature_scaler = StandardScaler()
        extra_features_scaled = np.empty_like(
            extra_features_raw, dtype=np.float32
        )

        extra_features_scaled[train_idx] = (
            extra_feature_scaler.fit_transform(
                extra_features_raw[train_idx]
            ).astype(np.float32)
        )
        extra_features_scaled[val_idx] = (
            extra_feature_scaler.transform(
                extra_features_raw[val_idx]
            ).astype(np.float32)
        )
        extra_features_scaled[test_idx] = (
            extra_feature_scaler.transform(
                extra_features_raw[test_idx]
            ).astype(np.float32)
        )

    train_dataset = IDACDataset(
        solute_ids,
        solvent_ids,
        temperatures_scaled,
        target_scaled,
        train_idx,
        extra_features_scaled,
    )
    val_dataset = IDACDataset(
        solute_ids,
        solvent_ids,
        temperatures_scaled,
        target_scaled,
        val_idx,
        extra_features_scaled,
    )
    test_dataset = IDACDataset(
        solute_ids,
        solvent_ids,
        temperatures_scaled,
        target_scaled,
        test_idx,
        extra_features_scaled,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        temp_scaler,
        extra_feature_scaler,
    )
        
def parity_plots(datasets, fold):
    save_prefix=""
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
    })
    
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",         
        "mathtext.rm": "serif",
    })
    colors = {
        "Train": "#1f77b4",      
        "Validation": "#2ca02c",   
        "Test": "#ff7f0e"          
    }
    all_exp = []
    all_pred = []
    for name, (df, color) in datasets.items():
        all_exp.extend(df["Experimental Ln(IDAC)"].values)
        all_pred.extend(df["Predicted Ln(IDAC)"].values)

    all_exp = np.array(all_exp)
    all_pred = np.array(all_pred)
    min_val = min(all_exp.min(), all_pred.min())
    max_val = max(all_exp.max(), all_pred.max())
    margin = 0.05 * (max_val - min_val)
    lims = (min_val - margin, max_val + margin)

    def calc_metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return r2, mae, mse

    fig, ax = plt.subplots(figsize = (8.0, 6.4))

    all_true = []
    all_pred = []
    for name, (df, color) in datasets.items():
        all_true.extend(df["Experimental Ln(IDAC)"].values)
        all_pred.extend(df["Predicted Ln(IDAC)"].values)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    r2_all, mae_all, mse_all = calc_metrics(all_true, all_pred)
    ax.scatter(
        all_true,
        all_pred,
        s=12,
        alpha=0.25,
        color="lightgray",
        edgecolors="none",
        label="_nolegend_"
    )
    for name, (df, color) in datasets.items():

        y_true = df["Experimental Ln(IDAC)"].values
        y_pred = df["Predicted Ln(IDAC)"].values
        r2, mae, mse = calc_metrics(y_true, y_pred)
        n = len(df)
        ax.scatter(
            y_true,
            y_pred,
            s=18,
            alpha=0.6,
            color=colors.get(name, "gray"),
            edgecolors="none",
            label=f"{name} (n={n}, $R^2$={r2:.4f})"
        )
 
    ax.plot(lims, lims, "--", color="red", linewidth=1.8, label="Ideal parity")
    delta = 1.0
    ax.plot(lims, [x + delta for x in lims], ":", color="red", linewidth=1)
    ax.plot(lims, [x - delta for x in lims], ":", color="red", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\ln(\gamma_{\mathrm{exp}}^{\infty})$")
    ax.set_ylabel(r"$\ln(\gamma_{\mathrm{est}}^{\infty})$")
    metric_text = (
        rf"Overall $R^2$ = {r2_all:.4f}" "\n"
        rf"MAE = {mae_all:.4f}" "\n"
        rf"MSE = {mse_all:.4f}"
    )
    ax.text(
        0.05,
        0.95,
        metric_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize = 10,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="black",
            alpha=0.9
        )
    )

    ax.legend(frameon=True, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(
        f"plots/{save_prefix}parity_combined_Fold_{fold}.png",
        bbox_inches="tight",
        dpi=dpi
    )
    plt.close(fig)
    # ----------------------------
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    for name, (df, color) in datasets.items():
        y_true = df["Experimental Ln(IDAC)"].values
        residuals = df["Residual (Exp - Pred)"].values
        ax.scatter(
            y_true, residuals,
            s=18, alpha=0.6,
            color=colors.get(name, "gray"),
            edgecolors="none",
            linewidths=0.3,
            label=name
        ) 
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.axhline(1, linestyle=":", color="gray", linewidth=1)
    ax.axhline(-1, linestyle=":", color="gray", linewidth=1)
    ax.set_xlabel(r"$\ln(\gamma_{\mathrm{exp}}^{\infty})$")
    ax.set_ylabel(r"$\ln(\gamma_{\mathrm{exp}}^{\infty}) - \ln(\gamma_{\mathrm{est}}^{\infty})$")
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(f"plots/{save_prefix}residual_combined_Fold_{fold}.png", bbox_inches="tight",dpi=dpi)
    plt.close(fig)
    print("All plots have been generated and saved.")
       
def components_global_count(df):
    global_solute_counts = df['Solute'].value_counts()
    global_solvent_counts = df['Solvent'].value_counts()
    df['Solute_Freq_as_Solute'] = df['Solute'].map(global_solute_counts).fillna(0).astype(int)
    df['Solute_Freq_as_Solvent'] = df['Solute'].map(global_solvent_counts).fillna(0).astype(int)
    df['Solvent_Freq_as_Solute'] = df['Solvent'].map(global_solute_counts).fillna(0).astype(int)
    df['Solvent_Freq_as_Solvent'] = df['Solvent'].map(global_solvent_counts).fillna(0).astype(int)
    df.to_csv('csv/df_details.csv', index=False)
    return df

def check_db(df):
    print("*"*100)
    print("Columns:", df.columns.tolist())
    print("*"*100)
    print(df.info())
    print("*"*100)
    print(df.describe(include='all'))
    print("*"*100)
    missing = df.isna().sum()
    print("\nMissing values per column:\n", missing)
    solutes = set(df['Solute'].unique())
    solvents = set(df['Solvent'].unique())
    both = solutes.intersection(solvents)
    only_solute = solutes - solvents
    only_solvent = solvents - solutes
    print("*"*100)
    print(f"Analysis of Components in dataset':\n")
    print(f"Total Unique Solutes: {len(solutes)}")
    print(f"Total Unique Solvents: {len(solvents)}\n")
    print(f"Components Used as BOTH Solute and Solvent ({len(both)}):")
    print(f"Components Used ONLY as a Solute ({len(only_solute)}):")
    print(f"Components Used ONLY as a Solvent ({len(only_solvent)}):")
    df = components_global_count(df)
    df['Repeat_Count'] = df.groupby(['Solute', 'Solvent'])['ln_IDAC_exp'].transform('size')
    df = df.reset_index()
    return df
 
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
        
def remove_singleton_rows(df):
    all_components = pd.concat([df['Solute'], df['Solvent']]).dropna()
    counts = all_components.value_counts()
    singletons1 = set(counts[counts == 1].index)
    print("*"*100)
    print(f"Original dataset size: {len(df)} rows")
    print(f"Found {len(singletons1)} components with a frequency of exactly 1.")
    cleaned_df = df[~df['Solute'].isin(singletons1) & ~df['Solvent'].isin(singletons1)]
    rows_removed = len(df) - len(cleaned_df)
    print(f"Removed {rows_removed} rows.")
    print(f"Cleaned dataset size: {len(cleaned_df)} rows")
    print("*"*100)
    return cleaned_df

def plot_idac_distribution(df,pt):
    data = df['ln_IDAC_exp'].dropna().values.reshape(-1, 1)
    data_transformed = pt.fit_transform(data).flatten()
    data_raw = data.flatten()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # ---------------------------------------------------------
    # (a) LEFT PLOT: Raw Frequency Count
    # ---------------------------------------------------------
    ax1.hist(data_raw, bins=50, color="#2874A6", edgecolor="black", linewidth=1.0, alpha=0.85)
    ax1.set_xlabel(r'$\ln(\gamma^\infty)$', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Frequency (Count)', fontsize=14, fontweight='bold')
    ax1.set_title('(a) Raw Frequency', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=12)
    
    # ---------------------------------------------------------
    # (b) MIDDLE PLOT: Normalized Density + Fitted Gaussian (Raw)
    # ---------------------------------------------------------
    ax2.hist(data_raw, bins=50, density=True, color="#A9CCE3", edgecolor="black", linewidth=1.0, alpha=0.85, label='Data Density')
    mu_raw, std_raw = norm.fit(data_raw)
    xmin_raw, xmax_raw = ax2.get_xlim()
    x_raw = np.linspace(xmin_raw, xmax_raw, 100)
    p_raw = norm.pdf(x_raw, mu_raw, std_raw)
    
    ax2.plot(x_raw, p_raw, 'k', linewidth=2.5, label=rf'Gaussian Fit ($\mu={mu_raw:.2f}$, $\sigma={std_raw:.2f}$)')
    ax2.set_xlabel(r'$\ln(\gamma^\infty)$', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
    ax2.set_title('(b) Raw Density', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=12)
    ax2.legend(fontsize=11)
    # --------------------------------------------------------
    # (c) RIGHT PLOT: Normalized Density + Fitted Gaussian (Transformed)
    # ---------------------------------------------------------
    ax3.hist(data_transformed, bins=50, density=True, color="#48C9B0", edgecolor="black", linewidth=1.0, alpha=0.85, label='Transformed Density')
    mu_trans, std_trans = norm.fit(data_transformed)
    xmin_trans, xmax_trans = ax3.get_xlim()
    x_trans = np.linspace(xmin_trans, xmax_trans, 100)
    p_trans = norm.pdf(x_trans, mu_trans, std_trans)
    
    ax3.plot(x_trans, p_trans, 'k', linewidth=2.5, label=rf'Gaussian Fit ($\mu={mu_trans:.2f}$, $\sigma={std_trans:.2f}$)')
    ax3.set_xlabel(r'Transformed $\ln(\gamma^\infty)$', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
    ax3.set_title('(c) Yeo-Johnson Transformed Density', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='both', labelsize=12)
    ax3.legend(fontsize=11)
    
    # ---------------------------------------------------------
    # Final Adjustments and Save
    # ---------------------------------------------------------
    plt.tight_layout()
    plt.savefig("plots/idac_ln_distribution_1x3.png", bbox_inches='tight',dpi=dpi)
    plt.close()

def get_component_summary(df):
    solute_counts = df['Solute'].value_counts().rename('Solute_Count')
    solvent_counts = df['Solvent'].value_counts().rename('Solvent_Count')
    comp_df = pd.concat([solute_counts, solvent_counts], axis=1).fillna(0)
    comp_df['Total_Count'] = comp_df['Solute_Count'] + comp_df['Solvent_Count']

    def determine_role(row):
        if row['Solute_Count'] > 0 and row['Solvent_Count'] == 0:
            return 'Only Solute'
        elif row['Solute_Count'] == 0 and row['Solvent_Count'] > 0:
            return 'Only Solvent'
        else:
            return 'Both (Solute & Solvent)'

    comp_df['Role'] = comp_df.apply(determine_role, axis=1)

    def bin_frequency(count):
        if count == 1: return '1 time'
        elif count == 2: return '2 times'
        elif count == 3: return '3 times'
        elif count == 4: return '4 times'
        elif count == 5: return '5 times'
        else: return '6+ times'

    comp_df['Frequency_Bin'] = comp_df['Total_Count'].apply(bin_frequency)
    summary_table = comp_df.groupby(['Role', 'Frequency_Bin']).size().unstack(fill_value=0)
    columns_order = ['1 time', '2 times', '3 times', '4 times', '5 times', '6+ times']
    summary_table = summary_table[[col for col in columns_order if col in summary_table.columns]]
    return summary_table

def plot_component_distributions_comparison(df_original, df_revised):
    summary_orig = get_component_summary(df_original)
    summary_rev  = get_component_summary(df_revised)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    colors = ['#4C72B0', '#DD8452', '#55A868']

    # --- Plot (a) Original Dataset ---
    summary_orig.T.plot(kind='bar', ax=axes[0], width=0.75, color=colors, edgecolor='black', linewidth=0.8)
    axes[0].set_title('(a) Original Dataset', pad=15, fontweight='bold', loc='left')
    axes[0].set_xlabel('Total Appearances in Dataset', fontweight='bold', labelpad=10)
    axes[0].set_ylabel('Number of Unique Components', fontweight='bold', labelpad=10)
    
    # --- Plot (b) Revised Dataset ---
    summary_rev.T.plot(kind='bar', ax=axes[1], width=0.75, color=colors, edgecolor='black', linewidth=0.8)
    axes[1].set_title('(b) Revised Dataset (Filtered)', pad=15, fontweight='bold', loc='left')
    axes[1].set_xlabel('Total Appearances in Dataset', fontweight='bold', labelpad=10)
    
    for ax in axes:
        ax.tick_params(axis='x', rotation=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='bottom', 
                            xytext=(0, 3), 
                            textcoords='offset points',
                            fontsize=10)

    axes[0].legend(title='Component Role', title_fontproperties={'weight':'bold'}, frameon=False)
    axes[1].legend(title='Component Role', title_fontproperties={'weight':'bold'}, frameon=False)
    plt.tight_layout()
    plt.savefig('plots/component_distribution_1x2.png', bbox_inches='tight', dpi=dpi)
    plt.close()

regression_loss_fn = nn.MSELoss()
reconstruction_loss_fn = nn.MSELoss()
def run_epoch(model, loader, DEVICE, RECON_LOSS_WEIGHT, BETA_VAE, target_transformer, temp_scaler, optimizer=None):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss, total_reg_loss, total_recon_loss, total_kl_loss = 0, 0, 0, 0
    all_preds, all_targets = [], []

    for batch in loader:
        solute_id = batch["solute_id"].to(DEVICE)
        solvent_id = batch["solvent_id"].to(DEVICE)
        temperature = batch["temperature"].to(DEVICE)
        target = batch["target"].to(DEVICE).unsqueeze(1)
        if "extra_features" in batch:
            ef = batch["extra_features"].to(DEVICE)
        else:
            ef = None
        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            pred, x_recon, x_true, mu, logvar, z = model(solute_id, solvent_id, temperature,temp_scaler,ef)
            reg_loss = regression_loss_fn(pred, target)
            recon_loss = reconstruction_loss_fn(x_recon, x_true)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_div = kl_div / mu.size(0) # Average over batch
            loss = reg_loss + (RECON_LOSS_WEIGHT * recon_loss) + (BETA_VAE * kl_div)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * target.size(0)
        total_reg_loss += reg_loss.item() * target.size(0)
        total_recon_loss += recon_loss.item() * target.size(0)
        total_kl_loss += kl_div.item() * target.size(0)

        all_targets.append(target.detach().cpu().numpy())
        all_preds.append(pred.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    avg_reg_loss = total_reg_loss / len(loader.dataset)
    avg_recon_loss = total_recon_loss / len(loader.dataset)
    avg_kl_loss = total_kl_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    true_targets = target_transformer.inverse_transform(all_targets)
    true_preds = target_transformer.inverse_transform(all_preds)
    mse = (mean_squared_error(true_targets, true_preds))
    mae = mean_absolute_error(true_targets, true_preds)
    r2 = r2_score(true_targets, true_preds)
   
    return {
        "loss": avg_loss,
        "reg_loss": avg_reg_loss,
        "recon_loss": avg_recon_loss,
        "kl_loss": avg_kl_loss,
        "mse": mse,
        "mae": mae,
        "r2": r2
    }

def get_predictions(model, loader,DEVICE,
                    target_transformer, temp_scaler, 
                    solute_encoder, solvent_encoder ):
    model.eval()
    all_preds, all_targets = [], []
    all_solute_ids, all_solvent_ids = [], []
    all_temperatures = []
    all_z, all_mu, all_logvar = [], [], []
    with torch.no_grad():
        for batch in loader:
            solute_id = batch["solute_id"].to(DEVICE)
            solvent_id = batch["solvent_id"].to(DEVICE)
            temperature = batch["temperature"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            if "extra_features" in batch:
                        ef = batch["extra_features"].to(DEVICE)
            else:
                ef = None
            
            pred, x_recon, x_true, mu, logvar, z = model(solute_id, solvent_id, temperature,temp_scaler,ef)
            all_preds.append(pred.cpu().numpy().flatten())
            all_targets.append(target.cpu().numpy().flatten())
            all_solute_ids.append(solute_id.cpu().numpy())
            all_solvent_ids.append(solvent_id.cpu().numpy())
            all_temperatures.append(temperature.cpu().numpy())
            if z!=None:
                all_z.append(z.cpu().numpy())
                all_mu.append(mu.cpu().numpy())
                all_logvar.append(logvar.cpu().numpy()) 
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_solute_ids = np.concatenate(all_solute_ids)
    all_solvent_ids = np.concatenate(all_solvent_ids)
    all_temperatures = np.concatenate(all_temperatures)
    if all_z!=[]:
        all_z = np.concatenate(all_z)                # shape: [N, LATENT_DIM]
        all_mu = np.concatenate(all_mu)                # shape: [N, LATENT_DIM]
        all_logvar = np.concatenate(all_logvar)                # shape: [N, LATENT_DIM]
        
    preds_true = target_transformer.inverse_transform(all_preds.reshape(-1,1)).flatten()
    targets_true = target_transformer.inverse_transform(all_targets.reshape(-1,1)).flatten()
    temps_K = (all_temperatures * temp_scaler.scale_[0] + temp_scaler.mean_[0])
    df_res = pd.DataFrame({
        "Solute": solute_encoder.inverse_transform(all_solute_ids),
        "Solvent": solvent_encoder.inverse_transform(all_solvent_ids),
        "T [K]":temps_K,
        "Experimental Ln(IDAC)": targets_true,
        "Predicted Ln(IDAC)": preds_true,
        "Residual (Exp - Pred)": targets_true - preds_true
    })
    if all_z.size!=0:
        latent_dim = all_z.shape[1] 
        for i in range(latent_dim):
            df_res[f"z_{i}"] = all_z[:, i]
            df_res[f"mu_{i}"] = all_mu[:, i]
            df_res[f"logvar_{i}"] = all_logvar[:, i]
    
    return df_res

def Freq_plot(df_latent_analysis,name,Fold):
        plt.style.use('seaborn-v0_8-whitegrid')

        x = df_latent_analysis[name]
        y = df_latent_analysis["Residual (Exp - Pred)"].abs()
        COLOR_SPINES = "#2B2D42"  # Crisp Charcoal Spines
        COLOR_POINTS = "#1B4965"
        COLOR_LINE = "#E05638"
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        scatter = ax.scatter(
            x,
            y,
            alpha=0.25,
            s=22,  
            color=COLOR_POINTS,
            edgecolors="none",
            zorder=2,
            label=r"Individual $|\Delta \ln \gamma^{\infty}|$",
        )
        trend_max = (
            df_latent_analysis.groupby(name)["Residual (Exp - Pred)"]
            .apply(lambda v: v.abs().max())
            .sort_index()
        )
        (line,) = ax.plot(
            trend_max.index,
            trend_max.values,
            color=COLOR_LINE,
            linewidth=2.2,
            zorder=4,
            path_effects=[
                pe.Stroke(linewidth=3.2, foreground="white", alpha=0.8),
                pe.Normal(),
            ],
            label=r"Maximum $|\Delta \ln \gamma^{\infty}|$",
        )

        ax.set_xlabel(r"$f_{\min}$", fontsize=12, labelpad=8)
        ax.set_ylabel(
            r"$|\ln(\gamma_{\mathrm{exp}}^{\infty}) - \ln(\gamma_{\mathrm{est}}^{\infty})|$",
            labelpad=10,
        )  

        ax.set_xlim(-2, max(x) + 5)
        ax.set_ylim(-0.25, max(y) + 0.6)  
        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=5,
            width=1.1,
            colors=COLOR_SPINES,
        )
        for spine in ax.spines.values():
            spine.set_color(COLOR_SPINES)
            spine.set_linewidth(1.1)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        leg = ax.legend(
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            edgecolor="#E2E8F0",
            facecolor="#FFFFFF",
            borderpad=0.8,
        )
        leg.legend_handles[0].set_alpha(1.0)  
        plt.tight_layout()
        plt.savefig(
            f"plots/{name}_Fold_{fold}_all.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close() 
def find_idac_outlier_rows(df):
    n1 = 0 # 
    n2 = 0
    n3 = 0
    n4 = 0
    rows_to_remove = []
    for _, group in df.groupby(["Solute", "Solvent"]):
        group = group.copy()
        group["_idac_round"] = group["ln_IDAC_exp"].round(decimals=4)
        duplicated_mask = group.duplicated(subset=["T [K]", "_idac_round"],keep="first" )
        if duplicated_mask.any():
            rows_to_remove.extend(group.loc[duplicated_mask].index)
            group=group.loc[~duplicated_mask]
            n1 += duplicated_mask.sum()
        
        temps = group["T [K]"].values
        idac = group["ln_IDAC_exp"].values
        n = len(group)
  
        sensitivity_threshold = 0.1    
        if n == 2:
            delta_T = abs(temps[1] - temps[0])
            if delta_T > 0:
                sensitivity = abs((idac[1] - idac[0]) / delta_T)
                if sensitivity > sensitivity_threshold:
                    rows_to_remove.extend(group.index)
                    n2 += len(group.index)   
        if n==3:
            X = 1/group["T [K]"].values.reshape(-1, 1)
            y = group["ln_IDAC_exp"].values
            if not(np.abs(y)<0.5).all():
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                r2_threshold = 0.9
                r2 = r2_score(y, y_pred)
                outlier_mask = ( (np.abs((y-y_pred)/y)>0.1)  )
                if r2 < r2_threshold:
                    if outlier_mask.any():
                        rows_to_remove.extend(group.index[outlier_mask])
                        n3+=len(group.index[outlier_mask])  
        if n >= 4:
            iqr_multiplier = 1.5
            idac = group["ln_IDAC_exp"]
            q1 = idac.quantile(0.25)
            q3 = idac.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            outlier_mask = ((idac < lower_bound) | (idac > upper_bound))
            if outlier_mask.any():
                rows_to_remove.extend(group.index[outlier_mask])
                n4+= len(group.index[outlier_mask])
                group = group.drop(index=group.index[outlier_mask])
                X = 1/group["T [K]"].values.reshape(-1, 1)
                y = group["ln_IDAC_exp"].values
                if not(np.abs(y)<0.5).all():
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    r2_threshold = 0.9
                    r2 = r2_score(y, y_pred)
                    outlier_mask = ( (np.abs((y-y_pred)/y)>0.1)  )
                    if r2 < r2_threshold:
                        if outlier_mask.any():
                            rows_to_remove.extend(group.index[outlier_mask])
                            n4+=len(group.index[outlier_mask]) 
                            
    return sorted(set(rows_to_remove))

LEARNING_RATE = 1e-3
target_transformer = PowerTransformer(method="yeo-johnson", standardize=True)
id = 1 # 1 for case study 1 and 2 for the second case study
EPOCHS, patience, BATCH_SIZE  = 10000000, 250 , 128
SEED, ERROR_THRESHOLD, min_delta  = 123469, 1.0 , 1e-4 
set_seed(SEED)
##############################################
df, p, ef, ed, solvent_name_cas_lookup,solute_props,solvent_props = load_Case_Studies(id)
df_revised = df.copy()
cache_file_csv=f"full_property_database_{id}.csv"
df_prop = pd.read_csv(cache_file_csv)
              
if id == 2:
    comp_canonical_SMILE_lookup = dict(
        zip(df_prop["input"], df_prop["canonical_SMILES"]))
    df['Solute'] = df['Solute'].map(comp_canonical_SMILE_lookup)
    df['Solvent'] = df['Solvent'].map(comp_canonical_SMILE_lookup)
    
if id == 1:
    smiles_map = df_prop.set_index('input')['canonical_SMILES']
    df['Solute'] = df['Solute'].map(smiles_map).fillna(df['Solute'])
    df['Solvent'] = df['Solvent'].map(smiles_map).fillna(df['Solvent'])  

if id == 1:
    for i in range(20):
        print(i)
        rows_to_remove5 = find_idac_outlier_rows(df)
        if len(rows_to_remove5)== 0:
            break
        else:
            df = df.drop(index=rows_to_remove5).reset_index(drop=True)
            
    temps_by_pair = {
        ("CC(C)(C)Cl", "Cc1ccccc1"):273.1499974,
        ("CCCCl", "Cc1ccccc1"):273.1499995,
        ("Clc1ccccc1", "Cc1ccccc1"):	273.1499834,
        ("Cc1cc(C)cc(C)c1", "O"):298.1499885,
        ("N#Cc1ccccc1", "O"):298.149983,
        ("CCCCCC#N", "O"):298.1500006,
        ("CC(C)c1ccccc1", "O"):298.1499825,
        ("CC(=O)c1ccccc1", "O"):298.1500082,
        ("CCC(=O)O", "O"):298.1499945,
        ("OCCc1ccccc1", "O"):373.1499776,
        ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "CCCCO"):298.1499779,
        ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "CCCO"):298.1499853,
        ("CCCCCCCCCCO", "O"):289.1499845,
        ("ClC(Cl)Cl", "OCC(O)CO"):339.9999901,
        ("CO", "C1CCCC1"):288.1499899,
        ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "CCCO"):298.1499948,
        ("CCO", "CCCCCCCCCCC"):293.1500045,
        ("CCCCCC", "N#CCC#N"):298.1499989,
        ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "CCCCCCC"):298.149991,
        ("BrC(Br)C(Br)Br", "O"):298.1499908,
        ("c1ccncc1", "Cc1ccccc1"):298.1499978,
        ("CCCCCC",	"O=S1(=O)CCC(O)C1"):	373.1499807,
        ("CCCCOP(=O)(OCCCC)OCCCC", "ClC(Cl)Cl"):298.1499879,
        ("Cc1ccccc1O", "CC#N"):298.1500043,
        ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "CCCCCC"):298.1499999,
        ("Oc1ccccc1", "N#CCCOC(O)C(C(O)OCCC#N)(C(O)OCCC#N)C(O)OCCC#N"):393.1499868,
        ("p-isopropylbenzene", "CNC=O"):298.1500044,
        ("Oc1ccccc1", "N#CCCCCCCCCC#N"):393.1499966,
        ("Cc1ccc(O)cc1", "CC#N"):298.1499953,
        ("CCCCCCC", "NC=O"):298.1499827,
        ("CC(C)O", "CCCCCCCCCCCCCCCCCCO"):338.1500066,
        ("CCCCCCCCO", "CCCCCC"):298.1500067,
        ("C[N+](=O)[O-]", "CCCCCCCCCCCCCCCCCCO"):338.1499925,	
        ("CCO", "CCCCCCCCCCCCCCC"):323.1500074,
        ("CC(C)CC(C)(C)C", "CCCCCCC"):293.1499976
    }
    atol = 0.001  
    mask = df.apply(
        lambda row: (
            (row["Solute"], row["Solvent"]) in temps_by_pair
            and np.isclose(row["T [K]"], temps_by_pair[(row["Solute"], row["Solvent"])],
                        atol=atol, rtol=0)
        ),
        axis=1)
    df = df.loc[~mask].copy()  

    pairs_to_remove = [
                ("CC=O","O"),
                ("CCCCc1ccccc1", "O") ]
    mask = df.apply(lambda row: (row["Solute"], row["Solvent"]) in pairs_to_remove, axis=1)
    df = df[~mask]
############################################## 
pd.reset_option("display.max_rows")
df = check_db(df)
df_original = df.copy()
df = remove_singleton_rows(df) 
df = check_db(df)
plot_idac_distribution(df,target_transformer)
plot_component_distributions_comparison(df_original,df)
############################################## 
# df.to_pickle(f'my_data_{id}.pkl')
##############################################
df = pd.read_pickle(f'my_data_{id}.pkl')       
       
print("\nDataset preview:")
print(df.head())
print("\nNumber of rows:", len(df))
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
extra_features = df[ef] if len(ef) > 0 else None
solute_encoder = LabelEncoder()
solvent_encoder = LabelEncoder()
df["solute_id"] = solute_encoder.fit_transform(df["Solute"].astype(str))
df["solvent_id"] = solvent_encoder.fit_transform(df["Solvent"].astype(str))
temp_scaler = StandardScaler()
df["T_inv_scaled"] = temp_scaler.fit_transform(df[["T [K]"]])
target_raw = df["ln_IDAC_exp"].values.astype(np.float32)
temperatures = df["T_inv_scaled"].values.astype(np.float32)
solute_ids = df["solute_id"].to_numpy(dtype=np.int64)
solvent_ids = df["solvent_id"].to_numpy(dtype=np.int64)
raw_temperatures = df["T [K]"].to_numpy(dtype=np.float32)
n_solutes = df["solute_id"].nunique()
n_solvents = df["solvent_id"].nunique()
n_unique_directed = len(df[['Solute', 'Solvent']].drop_duplicates())
print(f"Number of unique directed binary systems: {n_unique_directed}")
 
N_FOLDS = 10
rng = np.random.RandomState(SEED)
fold_counts = np.zeros(N_FOLDS, dtype=int)
assigned_folds = np.full(len(df), fill_value=-1, dtype=int)
pair_groups = df.groupby(['Solute', 'Solvent']).groups
sorted_pairs = sorted(pair_groups.items(), key=lambda item: len(item[1]), reverse=True)
for (solute, solvent), indices in sorted_pairs:
    idx_list = list(indices)
    k = len(idx_list)
    if k == 1:
        best_fold = int(np.argmin(fold_counts))
        assigned_folds[idx_list[0]] = best_fold
        fold_counts[best_fold] += 1
        
    elif k == 2:
        best_pair = None
        min_load = float('inf')
        
        for f1 in range(N_FOLDS):
            for f2 in range(f1 + 1, N_FOLDS):
                diff = abs(f1 - f2)
                if diff > 1 and diff < (N_FOLDS - 1):
                    load = fold_counts[f1] + fold_counts[f2]
                    if load < min_load:
                        min_load = load
                        best_pair = (f1, f2)
        
        assigned_folds[idx_list[0]] = best_pair[0]
        assigned_folds[idx_list[1]] = best_pair[1]
        fold_counts[best_pair[0]] += 1
        fold_counts[best_pair[1]] += 1
        
    else:
        start_fold = int(np.argmin(fold_counts))
        for i, idx in enumerate(idx_list):
            f = (start_fold + i * 2) % N_FOLDS
            assigned_folds[idx] = f
            fold_counts[f] += 1

fold_partitions = [np.where(assigned_folds == k)[0] for k in range(N_FOLDS)]
fold_results = pd.DataFrame(index=df.index, columns=[f'Fold_{i+1}' for i in range(N_FOLDS)])
multi_pairs = [pair for pair, idxs in pair_groups.items() if len(idxs) > 1]
val_metrics_list = []
test_metrics_list = []
train_metrics_list = []
val_predictions_all = []
test_predictions_all = []
train_predictions_all = []

for fold_index in range(N_FOLDS):
    fold = fold_index + 1
    test_idx = fold_partitions[fold_index]
    val_idx = fold_partitions[(fold_index + 1) % N_FOLDS]
    train_idx = np.concatenate([
        fold_partitions[i]
        for i in range(N_FOLDS)
        if i not in {fold_index, (fold_index + 1) % N_FOLDS}
    ])
    fold_results.loc[train_idx, f'Fold_{fold}'] = 'Train'
    fold_results.loc[val_idx, f'Fold_{fold}'] = 'Val'
    fold_results.loc[test_idx, f'Fold_{fold}'] = 'Test'

    df_train = df.loc[train_idx]
    train_pairs_set = set(zip(df_train['Solute'], df_train['Solvent']))
    set_seed(SEED + fold)
    print("\n" + "=" * 72)
    print(f"FOLD {fold:02d}/{N_FOLDS}")
    print(
        f"Train: {len(train_idx):6d} "
        f"({100 * len(train_idx) / len(df):5.1f}%)"
    )
    print(
        f"Validation: {len(val_idx):6d} "
        f"({100 * len(val_idx) / len(df):5.1f}%)"
    )
    print(
        f"Test: {len(test_idx):6d} "
        f"({100 * len(test_idx) / len(df):5.1f}%)"
    )
    print("=" * 72)
    df['Split'] = 'None'
    df.loc[train_idx, 'Split'] = 'Train'
    df.loc[val_idx, 'Split'] = 'Validation'
    df.loc[test_idx, 'Split'] = 'Test'

    pair_counts = df.groupby(['Solute', 'Solvent']).agg(
        total=('Split', 'count'),
        train_count=('Split', lambda x: (x == 'Train').sum())
    ).reset_index()
    violations = pair_counts[(pair_counts['total'] > 1) & (pair_counts['train_count'] == 0)]
    if len(violations) > 0:
        print(f"Warning/Error: {len(violations)} pairs have >1 data point but NO data in Train!")
        print(violations.head())
    else:
        print("Check passed: All pairs with >1 data points have at least one sample in Train.")

    fold_target_transformer = PowerTransformer(method="yeo-johnson",standardize=True,)
    (
        train_loader,
        val_loader,
        test_loader,
        fold_temp_scaler,
        fold_extra_feature_scaler,
    ) = get_fold_dataloaders(
        df=df,
        solute_ids=solute_ids,
        solvent_ids=solvent_ids,
        raw_temperatures=raw_temperatures,
        batch_size=BATCH_SIZE,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        target_transformer=fold_target_transformer,
        extra_features=extra_features,
    )
    model = β_CVAE(
        n_solutes=n_solutes,
        n_solvents=n_solvents,
        embed_dim=p.EMBED_DIM,
        hidden_dim_enc_dec=p.HIDDEN_DIM_AE,
        hidden_dim_regressor=p.HIDDEN_DIM_Regressor,
        latent_dim=p.LATENT_DIM,
        n_ef=len(ef),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    best_val_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_path = (f"saved_models/best_idac_autoencoder_{id}_fold{fold:02d}.pt")

    for epoch in range(1, EPOCHS + 1):
        train_m = run_epoch(
            model,
            train_loader,
            DEVICE,
            p.RECON_LOSS_WEIGHT,
            p.BETA_VAE,
            fold_target_transformer,
            fold_temp_scaler,
            optimizer=optimizer,
        )

        val_m = run_epoch(
            model,
            val_loader,
            DEVICE,
            p.RECON_LOSS_WEIGHT,
            p.BETA_VAE,
            fold_target_transformer,
            fold_temp_scaler,
            optimizer=None,
        )

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:05d} | "
                f"Train MSE: {train_m['mse']:.4f} | "
                f"Val MSE: {val_m['mse']:.4f} | "
                f"Val R2: {val_m['r2']:.4f}"
            )

        if val_m["mse"] < best_val_mse - min_delta:
            best_val_mse = val_m["mse"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "best_idac_autoencoder": model.state_dict(),
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "best_val_mse": best_val_mse,
                    "train_indices": train_idx,
                    "validation_indices": val_idx,
                    "test_indices": test_idx,
                    "solute_encoder_classes":
                        solute_encoder.classes_,
                    "solvent_encoder_classes":
                        solvent_encoder.classes_,
                    "target_transformer":
                        fold_target_transformer,
                    "temp_scaler":
                        fold_temp_scaler,
                    "extra_feature_scaler":
                        fold_extra_feature_scaler,
                    "config": {
                        "n_solutes": n_solutes,
                        "n_solvents": n_solvents,
                        "embed_dim": p.EMBED_DIM,
                        "hidden_dim_AE": p.HIDDEN_DIM_AE,
                        "hidden_dim_Regressor":
                            p.HIDDEN_DIM_Regressor,
                        "latent_dim": p.LATENT_DIM,
                        "n_ef": len(ef),
                    },
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}; "
                f"best validation MSE: "
                f"{best_val_mse:.4f}"
            )
            break

    checkpoint = torch.load(
        best_model_path,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["best_idac_autoencoder"])

    final_val_m = run_epoch(
        model,
        val_loader,
        DEVICE,
        p.RECON_LOSS_WEIGHT,
        p.BETA_VAE,
        fold_target_transformer,
        fold_temp_scaler,
        optimizer=None,
    )

    final_test_m = run_epoch(
        model,
        test_loader,
        DEVICE,
        p.RECON_LOSS_WEIGHT,
        p.BETA_VAE,
        fold_target_transformer,
        fold_temp_scaler,
        optimizer=None,
    )
    final_train_m = run_epoch(
        model,
        train_loader,
        DEVICE,
        p.RECON_LOSS_WEIGHT,
        p.BETA_VAE,
        fold_target_transformer,
        fold_temp_scaler,
        optimizer=None,
    )
    val_metrics_list.append({
        "fold": fold,
        "best_epoch": best_epoch,
        "n_train": len(train_idx),
        "n_validation": len(val_idx),
        "n_test": len(test_idx),
        "mse": final_val_m["mse"],
        "mae": final_val_m["mae"],
        "r2": final_val_m["r2"],
        "mse": (final_val_m["mse"]),
    })

    test_metrics_list.append({
        "fold": fold,
        "best_epoch": best_epoch,
        "n_train": len(train_idx),
        "n_validation": len(val_idx),
        "n_test": len(test_idx),
        "mse": final_test_m["mse"],
        "mae": final_test_m["mae"],
        "r2": final_test_m["r2"],
        "mse": (final_test_m["mse"]),
    })
    train_metrics_list.append({
        "fold": fold,
        "best_epoch": best_epoch,
        "n_train": len(train_idx),
        "n_validation": len(val_idx),
        "n_test": len(test_idx),
        "mse": final_train_m["mse"],
        "mae": final_train_m["mae"],
        "r2": final_train_m["r2"],
        "mse": (final_train_m["mse"]),
    })
    print(
        f"Train | MSE: {final_train_m['mse']:.4f} | "
        f"MAE: {final_train_m['mae']:.4f} | "
        f"R2: {final_train_m['r2']:.4f}"
    )
    print(
        f"Validation | MSE: {final_val_m['mse']:.4f} | "
        f"MAE: {final_val_m['mae']:.4f} | "
        f"R2: {final_val_m['r2']:.4f}"
    )
    print(
        f"Test       | MSE: {final_test_m['mse']:.4f} | "
        f"MAE: {final_test_m['mae']:.4f} | "
        f"R2: {final_test_m['r2']:.4f}"
    )

    df_val_pred = get_predictions(
        model,
        val_loader,
        DEVICE,
        fold_target_transformer,
        fold_temp_scaler,
        solute_encoder,
        solvent_encoder,
    )
    df_val_pred["Fold"] = fold
    df_val_pred["Original_Row_Index"] = val_idx
    val_predictions_all.append(df_val_pred)

    df_test_pred = get_predictions(
        model,
        test_loader,
        DEVICE,
        fold_target_transformer,
        fold_temp_scaler,
        solute_encoder,
        solvent_encoder,
    )
    df_test_pred["Fold"] = fold
    df_test_pred["Original_Row_Index"] = test_idx
    test_predictions_all.append(df_test_pred)

    df_train_pred = get_predictions(
        model,
        train_loader,
        DEVICE,
        fold_target_transformer,
        fold_temp_scaler,
        solute_encoder,
        solvent_encoder,
    )
    df_train_pred["Fold"] = fold
    df_train_pred["Original_Row_Index"] = train_idx
    train_predictions_all.append(df_train_pred)
# =========== New ===================================================
    df_train_pred['Split'] = 'Train'
    df_val_pred  ['Split'] = 'Validation'
    df_test_pred ['Split'] = 'Test'
    df_combined = pd.concat([df_train_pred, df_val_pred, df_test_pred], ignore_index=True)
    datasets = {  'Train': (df_train_pred, 'blue'),'Validation': (df_val_pred, 'green'),
                'Test': (df_test_pred, 'orange')}
    parity_plots(datasets,fold)

    all_exp = []
    all_pred = []
    for name, (df_, color) in datasets.items():
        all_exp.extend(df_["Experimental Ln(IDAC)"].values)
        all_pred.extend(df_["Predicted Ln(IDAC)"].values)

    all_exp = np.array(all_exp)
    all_pred = np.array(all_pred)
    def calc_metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return r2, mae, mse
    r2_all, mae_all, mse_all = calc_metrics(all_exp, all_pred)
    print(
        f"All (train+test+val)       | MSE: {mse_all:.4f} | "
        f"MAE: {mae_all:.4f} | "
        f"R2: {r2_all:.4f}"
    )

    all_z      = df_combined.filter(regex='^z_').to_numpy()
    all_mu     = df_combined.filter(regex='^mu_').to_numpy()
    all_logvar = df_combined.filter(regex='^logvar_').to_numpy()
    if all_z.shape[1] == 2:
        df_combined['z1'] = all_z[:, 0]
        df_combined['z2'] = all_z[:, 1]
        df_combined['mu1'] = all_mu[:, 0]
        df_combined['mu2'] = all_mu[:, 1]
        df_combined['logvar1'] = all_logvar[:, 0]
        df_combined['logvar2'] = all_logvar[:, 1]
    else:
        pca = PCA(n_components=3)
        z_pca = pca.fit_transform(all_z)
        mu_pca = pca.fit_transform(all_mu)
        logvar_pca = pca.fit_transform(all_logvar)
        df_combined['z1'] = z_pca[:, 0]
        df_combined['z2'] = z_pca[:, 1]
        df_combined['z3'] = z_pca[:, 2]
        df_combined['mu1'] = mu_pca[:, 0]
        df_combined['mu2'] = mu_pca[:, 1]
        df_combined['mu3'] = mu_pca[:, 2]
        df_combined['logvar1'] = logvar_pca[:, 0]
        df_combined['logvar2'] = logvar_pca[:, 1]
        df_combined['logvar3'] = logvar_pca[:, 2]


    df_latent_analysis = df_combined[['Solute', 'Solvent', 'T [K]', 
                                    'Experimental Ln(IDAC)', 'Predicted Ln(IDAC)', 'Residual (Exp - Pred)',
                                    'Fold','Split','z1', 'z2','mu1','mu2','logvar1','logvar2']].copy()
    df_latent_analysis = df_combined[['Solute', 'Solvent', 'T [K]', 
                                    'Experimental Ln(IDAC)', 'Predicted Ln(IDAC)', 'Residual (Exp - Pred)',
                                    'Fold','Split','z1', 'z2','z3','mu1','mu2','mu3','logvar1','logvar2','logvar3']].copy()

    df_latent_analysis = components_global_count(df_latent_analysis)
    df_latent_analysis['Repeat_Count'] = df_latent_analysis.groupby(['Solute', 'Solvent']
                                        )['Experimental Ln(IDAC)'].transform('size')

    df_latent_analysis = df_latent_analysis.sort_values(by=['Solute', 'Solvent', 'T [K]'])

    high_error_plots(df_latent_analysis, ERROR_THRESHOLD)
    df_latent_analysis['group_has_high_error'] = df_latent_analysis.groupby(
        ['Solute', 'Solvent'])['Is_High_Error'].transform('any')

    df_latent_analysis = df_latent_analysis.sort_values(
        by=['group_has_high_error', 'Solute', 'Solvent', 'T [K]'],
        ascending=[False, True, True, True])

    df_latent_analysis = df_latent_analysis.drop(columns='group_has_high_error', errors='ignore')
    df_latent_analysis = df_latent_analysis.drop(columns='Cluster_Label_z', errors='ignore')
    df_latent_analysis = df_latent_analysis.drop(columns='Cluster_Label_mu', errors='ignore')
    df_latent_analysis = df_latent_analysis.drop(columns='Cluster_Label_logvar', errors='ignore')
    df_latent_analysis.to_csv(f'csv/df_latent_analysis_{fold}.csv', index=False)

    abs_err = df_latent_analysis['Residual (Exp - Pred)'].abs()
    abs_err = df_latent_analysis[df_latent_analysis["Split"]!="Train"]['Residual (Exp - Pred)'].abs()
    pct_lt_01 = (abs_err < 0.1).mean() * 100
    pct_lt_02 = (abs_err < 0.2).mean() * 100
    pct_lt_03 = (abs_err < 0.3).mean() * 100

    print(f'% rows with |error| < 0.1: {pct_lt_01:.2f}%')
    print(f'% rows with |error| < 0.2: {pct_lt_02:.2f}%')
    print(f'% rows with |error| < 0.3: {pct_lt_03:.2f}%')  
    
    n_total = len(abs_err)

    count_lt_01 = (abs_err < 0.1).sum()
    count_lt_02 = (abs_err < 0.2).sum()
    count_lt_03 = (abs_err < 0.3).sum()

    pct_lt_01 = (count_lt_01 / n_total) * 100
    pct_lt_02 = (count_lt_02 / n_total) * 100
    pct_lt_03 = (count_lt_03 / n_total) * 100

    print(f"Total non-train points: {n_total}")
    print(f"|Err| < 0.1: {count_lt_01} points ({pct_lt_01:.2f}%)")
    print(f"|Err| < 0.2: {count_lt_02} points ({pct_lt_02:.2f}%)")
    print(f"|Err| < 0.3: {count_lt_03} points ({pct_lt_03:.2f}%)")
    ###########################################################

    dff = df_latent_analysis.query("Repeat_Count == 1 and Split != 'Train'")  
    dff['f_min'] = dff[['Solute_Freq_as_Solute', 'Solvent_Freq_as_Solvent','Solute_Freq_as_Solvent','Solvent_Freq_as_Solute']].min(axis=1)
    Freq_plot(dff,"f_min",fold)
    plot_0(datasets,df_latent_analysis,fold)


df_val_summary = pd.DataFrame(val_metrics_list)
df_test_summary = pd.DataFrame(test_metrics_list)

df_val_oof = pd.concat(
    val_predictions_all,
    ignore_index=True,
).sort_values("Original_Row_Index")

df_test_oof = pd.concat(
    test_predictions_all,
    ignore_index=True,
).sort_values("Original_Row_Index")

def pooled_prediction_metrics(prediction_df):
    y_true = prediction_df[
        "Experimental Ln(IDAC)"
    ].to_numpy()
    y_pred = prediction_df[
        "Predicted Ln(IDAC)"
    ].to_numpy()

    return {
        "mse": (mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

pooled_val = pooled_prediction_metrics(df_val_oof)
pooled_test = pooled_prediction_metrics(df_test_oof)

print("\n" + "=" * 72)
print("10-FOLD VALIDATION RESULTS")
print("=" * 72)
print(df_val_summary.to_string(index=False))
print(
    f"\nFold MSE: "
    f"{(df_val_summary['mse']).mean():.4f} +/- "
    f"{(df_val_summary['mse']).std(ddof=1):.4f}"
)
print(
    f"Fold MSE: "
    f"{df_val_summary['mse'].mean():.4f} +/- "
    f"{df_val_summary['mse'].std(ddof=1):.4f}"
)
print(
    f"Fold MAE : "
    f"{df_val_summary['mae'].mean():.4f} +/- "
    f"{df_val_summary['mae'].std(ddof=1):.4f}"
)
print(
    f"Fold R2  : "
    f"{df_val_summary['r2'].mean():.4f} +/- "
    f"{df_val_summary['r2'].std(ddof=1):.4f}"
)
print(
    f"Pooled OOF validation | "
    f"MSE: {(pooled_val['mse']):.4f} | "
    f"MAE: {pooled_val['mae']:.4f} | "
    f"R2: {pooled_val['r2']:.4f}"
)

print("\n" + "=" * 72)
print("10-FOLD OUT-OF-FOLD TEST RESULTS")
print("=" * 72)
print(df_test_summary.to_string(index=False))
print(
    f"\nFold MSE: "
    f"{(df_test_summary['mse']).mean():.4f} +/- "
    f"{(df_test_summary['mse']).std(ddof=1):.4f}"
)
print(
    f"Fold MSE: "
    f"{df_test_summary['mse'].mean():.4f} +/- "
    f"{df_test_summary['mse'].std(ddof=1):.4f}"
)
print(
    f"Fold MAE : "
    f"{df_test_summary['mae'].mean():.4f} +/- "
    f"{df_test_summary['mae'].std(ddof=1):.4f}"
)
print(
    f"Fold R2  : "
    f"{df_test_summary['r2'].mean():.4f} +/- "
    f"{df_test_summary['r2'].std(ddof=1):.4f}"
)
print(
    f"Pooled OOF test | "
    f"MSE: {(pooled_test['mse']):.4f} | "
    f"MAE: {pooled_test['mae']:.4f} | "
    f"R2: {pooled_test['r2']:.4f}"
)
print("=" * 72)
df_val_summary.to_csv(    "csv/cv_validation_fold_metrics.csv",    index=False)
df_test_summary.to_csv(  "csv/cv_test_fold_metrics.csv", index=False)
df_val_oof.to_csv( "csv/cv_oof_validation_predictions.csv", index=False)
df_test_oof.to_csv(    "csv/cv_oof_test_predictions.csv",    index=False)

pd.DataFrame([
    {"split": "validation", **pooled_val},
    {"split": "test", **pooled_test},
]).to_csv(
    "csv/cv_pooled_oof_metrics.csv",
    index=False,
)
print("\nSaved:")
print("  csv/cv_validation_fold_metrics.csv")
print("  csv/cv_test_fold_metrics.csv")
print("  csv/cv_oof_validation_predictions.csv")
print("  csv/cv_oof_test_predictions.csv")
print("  csv/cv_pooled_oof_metrics.csv")
###########################################################
abs_err = df_test_oof['Residual (Exp - Pred)'].abs()
pct_lt_01 = (abs_err < 0.1).mean() * 100
pct_lt_02 = (abs_err < 0.2).mean() * 100
pct_lt_03 = (abs_err < 0.3).mean() * 100

print(f'% rows with |error| < 0.1: {pct_lt_01:.2f}%')
print(f'% rows with |error| < 0.2: {pct_lt_02:.2f}%')
print(f'% rows with |error| < 0.3: {pct_lt_03:.2f}%')

n_total = len(abs_err)

count_lt_01 = (abs_err < 0.1).sum()
count_lt_02 = (abs_err < 0.2).sum()
count_lt_03 = (abs_err < 0.3).sum()

pct_lt_01 = (count_lt_01 / n_total) * 100
pct_lt_02 = (count_lt_02 / n_total) * 100
pct_lt_03 = (count_lt_03 / n_total) * 100

print(f"Total non-train points: {n_total}")
print(f"|Err| < 0.1: {count_lt_01} points ({pct_lt_01:.2f}%)")
print(f"|Err| < 0.2: {count_lt_02} points ({pct_lt_02:.2f}%)")
print(f"|Err| < 0.3: {count_lt_03} points ({pct_lt_03:.2f}%)")
