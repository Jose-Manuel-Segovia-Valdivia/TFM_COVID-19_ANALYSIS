
######### MODELOS AVANZADOS #########

# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pathlib import Path
import statsmodels.api as sm
from matplotlib.colors import TwoSlopeNorm

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS
# =============================================================================

project_root = Path.cwd().parent
outputs_dir = project_root / "Data" / "outputs"
dir_filtered_weekly = outputs_dir / "filtered_weekly"
tables_mod = outputs_dir / "tables" / "modeling"
dir_plots_analyst_mod = outputs_dir / "plots" / "analyst" / "modeling"
dir_plots_maps_static = outputs_dir / "plots" / "maps" / "static"
geo_path = project_root / "Data" / "inputs" / "countries.geo.json"

for d in [tables_mod, dir_plots_analyst_mod, dir_plots_maps_static]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. CARGA DE DATOS SEMANALES
# =============================================================================
print("Cargando datos semanales para modelado...")
try:
    weekly = pd.read_csv(dir_filtered_weekly / "weekly_model_ready.csv", sep=";")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo 'weekly_model_ready.csv'. Asegúrate de haber ejecutado '01_data_preparation.py' primero.")
    exit()

weekly["week"] = pd.to_datetime(weekly["week"])
weekly = weekly.sort_values(["country", "week"]).reset_index(drop=True)

# =============================================================================
# 3. DETECCIÓN DE ANOMALÍAS (OUTLIERS)
# =============================================================================
print("Realizando análisis de outliers...")

# --- 3.1 Análisis sobre el dataset completo ---
w = weekly.copy()
for col in ["new_cases_per_million", "new_deaths_per_million"]:
    w[f"{col}_med8"] = w.groupby("country")[col].transform(lambda s: s.rolling(8, min_periods=4).median())
    w[f"{col}_resid"] = w[col] - w[f"{col}_med8"]
    w[f"{col}_z"] = w.groupby("country")[f"{col}_resid"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0)
    )

def heatmap_outliers(df, var="new_cases_per_million", topn=40, hide=1.0, clip=3.0, suffix=""):
    zmat = df.pivot_table(index="country", columns="week", values=f"{var}_z", aggfunc="mean")
    strong_cnt = (zmat.abs() >= 2.0).sum(axis=1)
    top_countries = strong_cnt.sort_values(ascending=False).head(topn).index
    z_plot = zmat.loc[top_countries].clip(-clip, clip).where(zmat.abs() >= hide).sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(18, 9))
    sns.heatmap(z_plot, cmap="coolwarm", center=0, vmin=-clip, vmax=clip, cbar_kws={"label": "z-score"}, ax=ax)
    
    # Formato de ejes y título
    ax.set_ylabel("País"); ax.set_xlabel("Semana")
    weeks = z_plot.columns.to_list()
    if len(weeks) > 1:
        tick_idx = np.linspace(0, len(weeks)-1, 16, dtype=int)
        ax.set_xticks(tick_idx + 0.5)
        ax.set_xticklabels([pd.to_datetime(weeks[i]).strftime("%Y-%m") for i in tick_idx], rotation=45, ha="right")

    # Lógica para crear un título descriptivo
    if var == "new_cases_per_million":
        var_title = "Nuevos Casos por Millón"
    elif var == "new_deaths_per_million":
        var_title = "Nuevas Muertes por Millón"
    else:
        var_title = var 

    plt.title(f"Outliers en {var_title} — z-score (Top {topn}){suffix}", fontsize=16)
    plt.tight_layout()
    plt.savefig(dir_plots_analyst_mod / f"outliers_heatmap_{var.replace('_per_million','')}{suffix}.png", dpi=220)
    plt.show()

heatmap_outliers(w, "new_cases_per_million")
heatmap_outliers(w, "new_deaths_per_million")

# --- 3.2 Análisis recortado a Enero 2024 ---
print("Realizando análisis de outliers con datos hasta Ene-2024...")
CUT_END = pd.Timestamp("2024-02-01")
w_cut = w[w["week"] < CUT_END].copy()
heatmap_outliers(w_cut, "new_cases_per_million", suffix=" (hasta Ene-2024)")
heatmap_outliers(w_cut, "new_deaths_per_million", suffix=" (hasta Ene-2024)")

# --- 3.3 Mapas de proporción de anomalías ---
def build_outlier_gdf(kind="cases"):
    col = f"new_{kind}_per_million"
    w_map = weekly[weekly["week"] < CUT_END].copy()
    
    # z-scores
    w_map[f"{col}_med8"] = w_map.groupby("country")[col].transform(lambda s: s.rolling(8, min_periods=4).median())
    w_map[f"{col}_resid"] = w_map[col] - w_map[f"{col}_med8"]
    z = w_map.groupby("country")[f"{col}_resid"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0))
    w_map["spike"] = z > 3
    w_map["drop"] = z < -3
    
    # Calcula tasas y une con datos geográficos
    rates = w_map.groupby(["code", "country"]).agg(spike_rate=("spike", "mean"), drop_rate=("drop", "mean")).reset_index()
    
    world = gpd.read_file(geo_path)
    world["iso3"] = world["id"]
    gdf = world.merge(rates, left_on="iso3", right_on="code", how="left")
    return gdf

def plot_single_outlier_map(gdf, rate_type, kind_text):
    """Genera y guarda un único mapa de anomalías (para picos o caídas)."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    # Determina título, mapa de color y nombre de archivo según el tipo
    if rate_type == "spike_rate":
        title_text = f"Proporción de semanas con PICOS de {kind_text}"
        cmap = "Reds"
    else: 
        title_text = f"Proporción de semanas con CAÍDAS de {kind_text}"
        cmap = "Blues"
        
    gdf.plot(column=rate_type, ax=ax, cmap=cmap, legend=True, 
             missing_kwds={"color": "lightgrey", "label": "Sin datos"}, 
             edgecolor="white", linewidth=0.3)
    
    ax.set_title(f"{title_text} (z-score > |3|) — hasta Ene-2024", fontsize=16)
    ax.set_axis_off()
    
    plt.tight_layout()
    filename = f"map_outliers_{kind_text}_{rate_type}_upto_2024-01.png"
    plt.savefig(dir_plots_maps_static / filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Mapa guardado: {filename}")

# --- Ejecución para CASOS ---
print("\n--- Generando mapas de anomalías para Casos ---")
gdf_cases = build_outlier_gdf(kind="cases")
plot_single_outlier_map(gdf_cases, "spike_rate", "casos")
plot_single_outlier_map(gdf_cases, "drop_rate", "casos")

# --- Ejecución para MUERTES ---
print("\n--- Generando mapas de anomalías para Muertes ---")
gdf_deaths = build_outlier_gdf(kind="deaths")
plot_single_outlier_map(gdf_deaths, "spike_rate", "muertes")
plot_single_outlier_map(gdf_deaths, "drop_rate", "muertes")

# =============================================================================
# 4. MODELO OLS: IMPACTO DE VACUNACIÓN EN MUERTES
# =============================================================================
print("\n--- Ejecutando Modelo OLS: Impacto de Vacunación ---")
p = weekly.copy()
p["pvax_lag3"] = p.groupby("country")["people_vaccinated_per_hundred"].shift(3)
want_cols = [
    "country", "week", "new_deaths_per_million", "pvax_lag3", "stringency_index",
    "reproduction_rate", "median_age", "gdp_per_capita", "hospital_beds_per_thousand"
]
panel = p[[c for c in want_cols if c in p.columns]].dropna().copy()
panel.to_csv(tables_mod / "vax_impact_panel.csv", sep=";", index=False)

X_cols = [
    "pvax_lag3", "stringency_index", "reproduction_rate", "median_age",
    "gdp_per_capita", "hospital_beds_per_thousand"
]
X_cols_final = [c for c in X_cols if c in panel.columns]

X = sm.add_constant(panel[X_cols_final])
y = panel["new_deaths_per_million"]

ols_vax = sm.OLS(y, X).fit(cov_type="HC3")
print("\n--- Resultados del Modelo OLS (Vacunación) ---")
print(ols_vax.summary())

# --- 4.1) Se guarda tabla de resultados ---
coef = ols_vax.params.rename("coef")
se = ols_vax.bse.rename("se")
tval = ols_vax.tvalues.rename("t")
pval = ols_vax.pvalues.rename("pval")
ci = ols_vax.conf_int(alpha=0.05).rename(columns={0:"ci_low", 1:"ci_high"})

tab_full = (
    pd.concat([coef, se, tval, pval, ci], axis=1)
      .reset_index().rename(columns={"index":"variable"})
      .assign(signif=lambda d: np.where(d["pval"] < 0.05, "*", ""))
      .round(4)
)
tab_key = tab_full[tab_full["variable"].isin(["const"] + X_cols_final)].copy()

tab_full.to_csv(tables_mod / "ols_vax_impact_full.csv", sep=";", index=False)
tab_key.to_csv(tables_mod / "ols_vax_impact_key.csv", sep=";", index=False)

# --- 4.2) Gráfico de coeficientes ---
plot_df = tab_key[tab_key["variable"] != "const"].copy()
plot_df = plot_df.reindex(plot_df["t"].abs().sort_values().index)

plt.figure(figsize=(9, 5))
plt.errorbar(
    x=plot_df["coef"],
    y=plot_df["variable"],
    xerr=[plot_df["coef"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["coef"]],
    fmt="o", capsize=3, linestyle='None'
)
plt.axvline(0, color="gray", ls="--", lw=1)
plt.title("Impacto de vacunación y controles sobre muertes/millón (OLS, IC95%)")
plt.xlabel("Coeficiente"); plt.ylabel("")
plt.tight_layout()
plt.savefig(dir_plots_analyst_mod / "ols_vax_coefplot.png", dpi=220)
plt.show()


# --- 4.3) Resumen final de la sección ---
print("OLS listo:")
print(f"  Obs: {int(ols_vax.nobs)}  | R²: {ols_vax.rsquared:.3f}  | R² adj: {ols_vax.rsquared_adj:.3f}")
print("  Tablas guardadas:", (tables_mod / 'ols_vax_impact_key.csv').name)
print("  Plots guardados:", (dir_plots_analyst_mod / 'ols_vax_coefplot.png').name)

# =============================================================================
# 5. MODELO OLS: ÍNDICE DE EFICIENCIA
# =============================================================================
print("\n--- Calculando Índice de Eficiencia (Método Original) ---")
q = weekly.copy()
q["pvax_lag3"] = q.groupby("country")["people_vaccinated_per_hundred"].shift(3)

agg = (q.groupby("country").agg(
    deaths_pm=("new_deaths_per_million", "median"),
    pvax_lag3=("pvax_lag3", "median"),
    stringency_index=("stringency_index", "mean"),
    reproduction_rate=("reproduction_rate", "mean"),
    median_age=("median_age", "first"),
    gdp_per_capita=("gdp_per_capita", "first"),
    hospital_beds_per_thousand=("hospital_beds_per_thousand", "first"),
    code=("code", "first"),
    continent=("continent", "first")
).dropna().reset_index())

X_cols = [
    "pvax_lag3", "stringency_index", "reproduction_rate", "median_age",
    "gdp_per_capita", "hospital_beds_per_thousand"
]
X_eff_cols = [c for c in X_cols if c in agg.columns]
X_eff = sm.add_constant(agg[X_eff_cols])
y_eff = agg["deaths_pm"]

ols_eff = sm.OLS(y_eff, X_eff).fit(cov_type="HC3")
agg["resid"] = ols_eff.resid
agg["eff_index"] = (-agg["resid"] - np.mean(-agg["resid"])) / np.std(-agg["resid"])

agg.sort_values("eff_index", ascending=False).to_csv(tables_mod / "efficiency_index_by_country_original.csv", sep=";", index=False)


# --- Gráfico de Barras Top/Bottom 10 ---
top = agg.nlargest(10, "eff_index")
bot = agg.nsmallest(10, "eff_index")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Gráficos 
axes[0].barh(top["country"], top["eff_index"], color="#1f77b4")
axes[0].invert_yaxis()
axes[1].barh(bot["country"], bot["eff_index"], color="#d62728")
axes[1].invert_yaxis()


# Asigna títulos con un tamaño de fuente mayor
axes[0].set_title("Top 10 eficiencia (z-score ↑ mejor)", fontsize=16)
axes[1].set_title("Bottom 10 eficiencia (z-score ↓ peor)", fontsize=16)

for ax in axes:
    ax.set_xlabel("Índice de eficiencia (z-score)", fontsize=14)   
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(dir_plots_analyst_mod / "efficiency_ranking_original.png", dpi=220)
plt.show()

# --- Mapas de Eficiencia y Residuos ---
print("Generando mapas de eficiencia...")
world = gpd.read_file(geo_path)
world["iso3"] = world["id"]
g_map = world.merge(agg[['code', 'resid', 'eff_index']], left_on="iso3", right_on="code", how="left")

def draw_map(gdf, metric, title, outpng, cmap, abs_clip=None):
    fig, ax = plt.subplots(figsize=(16, 8))
    world.plot(ax=ax, color="#e6e6e6", edgecolor="white", linewidth=0.2)

    data = gdf.copy()
    if abs_clip is None:
        vmax_abs = np.nanpercentile(np.abs(data[metric]), 98)
    else:
        vmax_abs = float(abs_clip)

    norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs)

    data.dropna(subset=[metric]).plot(
        ax=ax, column=metric, cmap=cmap, norm=norm,
        edgecolor="white", linewidth=0.3, legend=True,
        legend_kwds={"shrink": 0.6, "label": metric}
    )
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(outpng, dpi=300, bbox_inches="tight")
    plt.show()

# 1) Mapa del Índice de Eficiencia (z-score)
draw_map(
    gdf=g_map,
    metric="eff_index",
    title="Eficiencia Sanitaria — z-score del Residuo (Azul=Mejor, Rojo=Peor)",
    outpng=dir_plots_analyst_mod / "map_efficiency_zscore.png",
    cmap="coolwarm_r", # Invertido para que azul sea positivo/mejor
    abs_clip=3
)

# 2) Mapa del Residuo Bruto
draw_map(
    gdf=g_map,
    metric="resid",
    title="Residuo Bruto (Muertes Observadas − Predichas) — Azul=Mejor, Rojo=Peor",
    outpng=dir_plots_analyst_mod / "map_efficiency_residual.png",
    cmap="coolwarm", # Normal, azul es negativo/mejor
    abs_clip=None
)

# =============================================================================
# 6. ANÁLISIS DE INTERACCIÓN Y CUARTILES
# =============================================================================
print("\n--- Realizando análisis de interacción y por cuartiles de edad ---")
z = panel.copy()
for c in ["pvax_lag3", "median_age", "stringency_index", "reproduction_rate"]:
    if c in z.columns:
        z[c+"_std"] = (z[c]-z[c].mean())/z[c].std()

if 'pvax_lag3_std' in z.columns and 'median_age_std' in z.columns:
    z["pvaxXage"] = z["pvax_lag3_std"] * z["median_age_std"]
    
    int_cols = ["pvax_lag3_std", "median_age_std", "pvaxXage", "stringency_index_std", "reproduction_rate_std"]
    X_int = sm.add_constant(z[[c for c in int_cols if c in z.columns]])
    y_int = z["new_deaths_per_million"]

    ols_int = sm.OLS(y_int, X_int).fit(cov_type="HC3")
    print("\n--- Modelo de Interacción (Vacunación x Edad) ---")
    print(ols_int.summary())
    
    
    coef_i = ols_int.params.to_frame("coef").join(ols_int.conf_int().rename(columns={0:"ci_low", 1:"ci_high"}))
    
    vars_to_plot = ["pvax_lag3_std", "median_age_std", "pvaxXage"]
    coef_plot_i = coef_i.loc[coef_i.index.isin(vars_to_plot)].reset_index().rename(columns={"index":"var"})

    if not coef_plot_i.empty:
        plt.figure(figsize=(7, 4))
        plt.errorbar(coef_plot_i["coef"], coef_plot_i["var"],
                     xerr=[coef_plot_i["coef"] - coef_plot_i["ci_low"], coef_plot_i["ci_high"] - coef_plot_i["coef"]],
                     fmt="o", capsize=3, linestyle='None')
        plt.axvline(0, color="gray", ls="--")
        plt.title("Efecto Básico e Interacción Vacunación × Edad")
        plt.xlabel("Coeficiente"); plt.ylabel("")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(dir_plots_analyst_mod / "ols_vax_age_interaction.png", dpi=220)
        plt.show()

g = panel.dropna(subset=["median_age"]).copy()
g["age_q"] = pd.qcut(g["median_age"], 4, labels=["Q1 (más joven)", "Q2", "Q3", "Q4 (más mayor)"], duplicates='drop')

rows = []
for qlab, dfq in g.groupby("age_q", observed=False):
    try:
        Xq_cols = ["pvax_lag3", "stringency_index", "reproduction_rate"]
        dfq_clean = dfq[Xq_cols + ["new_deaths_per_million"]].dropna()
        if dfq_clean.shape[0] < 15: continue
            
        Xq = sm.add_constant(dfq_clean[Xq_cols])
        yq = dfq_clean["new_deaths_per_million"]
        m = sm.OLS(yq, Xq).fit(cov_type="HC3")
        
        b = m.params["pvax_lag3"]
        lo, hi = m.conf_int().loc["pvax_lag3"]
        rows.append({"age_q": qlab, "coef": b, "ci_low": lo, "ci_high": hi})
    except Exception:
        pass

if rows:
    band = pd.DataFrame(rows)
    plt.figure(figsize=(8, 5))
    x_err_lower = band["coef"] - band["ci_low"]
    x_err_upper = band["ci_high"] - band["coef"]
    plt.errorbar(band["coef"], band["age_q"], xerr=[x_err_lower, x_err_upper], fmt="o", capsize=4, linestyle='None')
    plt.axvline(0, color="gray", ls="--")
    plt.title("Efecto de vacunación por cuartiles de edad")
    plt.xlabel("Coeficiente pvax_lag3"); plt.ylabel("Cuartil de Edad Media")
    plt.tight_layout()
    plt.savefig(dir_plots_analyst_mod / "ols_vax_by_age_quartiles.png", dpi=220)
    plt.show()

print("\n--- Proceso de modelado avanzado finalizado. ---")