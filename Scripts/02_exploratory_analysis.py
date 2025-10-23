
######### ANÁLISIS EXPLORATORIO #########

# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from pathlib import Path

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS
# =============================================================================

project_root = Path.cwd().parent
outputs_dir = project_root / "Data" / "outputs"
dir_filtered_weekly = outputs_dir / "filtered_weekly"
dir_raw = outputs_dir / "raw"

# Rutas para guardar gráficos
dir_plots_sponsor_ev = outputs_dir / "plots" / "sponsor" / "evolution"
dir_plots_sponsor_key = outputs_dir / "plots" / "sponsor" / "key_relations"
dir_plots_analyst_ev = outputs_dir / "plots" / "analyst" / "evolution"
dir_plots_analyst_rel = outputs_dir / "plots" / "analyst" / "relations"

for d in [dir_plots_sponsor_ev, dir_plots_sponsor_key, dir_plots_analyst_ev, dir_plots_analyst_rel]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. CARGA DE DATOS
# =============================================================================
print("Cargando datos semanales y de países para análisis...")
try:
    df_countries_weekly = pd.read_csv(dir_filtered_weekly / "df_countries_graph_weekly.csv", sep=";")
    df_regions_weekly = pd.read_csv(dir_filtered_weekly / "df_regions_graph_weekly.csv", sep=";")
    df_countries = pd.read_csv(dir_raw / "df_countries.csv", sep=";") 
except FileNotFoundError:
    print("Error: No se encontraron los archivos de datos. Asegúrate de ejecutar '01_data_preparation.py' primero.")
    exit()


for df in [df_countries_weekly, df_regions_weekly]:
    if 'week_start' in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"])


# =============================================================================
# 3. MATRIZ DE CORRELACIÓN
# =============================================================================
print("Generando matriz de correlación...")
num_vars = [
    "total_cases", "total_deaths", "total_cases_per_million", "total_deaths_per_million", "new_cases",
    "new_deaths", "new_cases_per_million", "new_deaths_per_million", "new_cases_smoothed",
    "new_deaths_smoothed", "new_cases_smoothed_per_million", "new_deaths_smoothed_per_million",
    "population", "median_age", "gdp_per_capita", "diabetes_prevalence", "extreme_poverty",
    "hospital_beds_per_thousand", "stringency_index", "reproduction_rate", "people_vaccinated_per_hundred"
]
corr_cols = [c for c in num_vars if c in df_countries.columns]
corr = df_countries[corr_cols].corr()

# --- Inicio del código del gráfico ---
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
ax = sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 8})

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.title("Matriz de Correlación de Variables Clave")
plt.tight_layout()
plt.savefig(dir_plots_sponsor_key / "matriz_correlacion.png", dpi=200)
plt.show()

# =============================================================================
# 4. GRÁFICOS SPONSOR — EVOLUCIÓN MUNDO Y CONTINENTES
# =============================================================================
print("Generando gráficos de evolución para Sponsor...")
continents = ["Europe", "Africa", "Oceania", "North America", "South America", "Asia"]
cont = df_regions_weekly[df_regions_weekly["continent"].isin(continents)].copy()
cont["new_deaths_smoothed_scaled"] = cont["new_deaths_smoothed"] * 100
world = df_regions_weekly[df_regions_weekly["continent"] == "World"].copy()

# === 4.1) Mundo: casos y muertes (suavizados) ===
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(world["week_start"], world["new_cases_smoothed"], label="Nuevos casos (suavizados)")
ax.plot(world["week_start"], world["new_deaths_smoothed"] * 100, label="Nuevas muertes x100 (suavizadas)", color="red", linestyle="--")
ax.set_title("Mundo — Evolución semanal: Nuevos casos y Nuevas muertes x100 (suavizados)")
ax.set_xlabel("Fecha"); ax.set_ylabel("Recuento semanal")
ax.legend(); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig(dir_plots_sponsor_ev / "mundo_casos_muertes_suavizado.png", dpi=200)
plt.show()

# === 4.2) Facets por continente — Casos y Muertes (suavizadas) ===
cont_sorted = cont.sort_values("week_start")
g_comb = sns.FacetGrid(cont_sorted, col="continent", col_wrap=3, height=4, aspect=1.2, sharey=False, sharex=True)
g_comb.map_dataframe(sns.lineplot, x="week_start", y="new_cases_smoothed", label="Nuevos casos (suavizados)")
g_comb.map_dataframe(sns.lineplot, x="week_start", y="new_deaths_smoothed_scaled", label="Nuevas muertes ×100 (suavizadas)", color="red", linestyle="--")
g_comb.set_axis_labels("Fecha", "Recuento semanal")
g_comb.set_titles("{col_name}")
for ax in g_comb.axes.flatten():
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right", fontsize="small")
    for label in ax.get_xticklabels():
        label.set_rotation(45)
plt.tight_layout()
g_comb.figure.suptitle("Nuevos casos vs Nuevas muertes ×100 — Evolución semanal por continente", y=1.03, fontsize=16)
g_comb.figure.savefig(dir_plots_sponsor_ev / "continentes_casos_muertes_suavizado.png", dpi=200, bbox_inches="tight")
plt.show()

# === 4.3 y 4.4) Comparativas de continentes (Casos) ===
for with_asia in [True, False]:
    fig, ax = plt.subplots(figsize=(11, 6))
    continents_to_plot = continents if with_asia else [c for c in continents if c != "Asia"]
    suffix = "" if with_asia else "_sin_Asia"
    title = "Comparativa por continente — Nuevos casos (suavizados)" + ("" if with_asia else " sin Asia")
    for reg in continents_to_plot:
        serie = cont[cont["continent"] == reg]
        ax.plot(serie["week_start"], serie["new_cases_smoothed"], label=reg, alpha=0.95)
    ax.set_title(title); ax.set_xlabel("Fecha"); ax.set_ylabel("Nuevos casos (suavizados)")
    ax.legend(ncol=2, fontsize=9); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(dir_plots_sponsor_ev / f"continentes_casos_suavizado{suffix}.png", dpi=200)
    plt.show()

# === 4.5 y 4.6) Comparativas de continentes (Muertes) ===
for with_asia in [True, False]:
    fig, ax = plt.subplots(figsize=(11, 6))
    continents_to_plot = continents if with_asia else [c for c in continents if c != "Asia"]
    suffix = "" if with_asia else "_sin_Asia"
    title = "Comparativa por continente — Nuevas muertes (suavizadas)" + ("" if with_asia else " sin Asia")
    for reg in continents_to_plot:
        serie = cont[cont["continent"] == reg]
        ax.plot(serie["week_start"], serie["new_deaths_smoothed"], label=reg, alpha=0.95)
    ax.set_title(title); ax.set_xlabel("Fecha"); ax.set_ylabel("Nuevas muertes (suavizadas)")
    ax.legend(ncol=2, fontsize=9); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(dir_plots_sponsor_ev / f"continentes_muertes_suavizado{suffix}.png", dpi=200)
    plt.show()

# =============================================================================
# 5. GRÁFICOS ANALYST — EVOLUCIÓN PAÍSES SELECCIONADOS
# =============================================================================
print("Generando gráficos de evolución para Analyst...")
selected_countries = ["Spain", "Germany", "United States", "Brazil", "India", "China", "South Africa", "Australia"]
country_pairs = [["Spain", "Germany"], ["United States", "Brazil"], ["South Africa", "Australia"], ["India", "China"]]
df_selected = df_countries_weekly[df_countries_weekly["country"].isin(selected_countries)].copy()
if "new_deaths_smoothed_per_million" in df_selected.columns:
    df_selected["new_deaths_smoothed_per_million_scaled"] = df_selected["new_deaths_smoothed_per_million"] * 100

# --- 5.1) Países seleccionados en pares ---
for pair in country_pairs:
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)
    for ax, country in zip(axes, pair):
        subset = df_selected[df_selected["country"] == country]
        ax.plot(subset["week_start"], subset["new_cases_smoothed_per_million"], label="Nuevos casos/millón (suavizados)", color="steelblue")
        if "new_deaths_smoothed_per_million_scaled" in subset.columns:
            ax.plot(subset["week_start"], subset["new_deaths_smoothed_per_million_scaled"], label="Nuevas muertes/millón ×100 (suavizadas)", linestyle="--", color="red")
        ax.set_title(f"Evolución en {country}", fontsize=11)
        ax.set_ylabel("Recuento semanal por millón"); ax.grid(True, alpha=0.3); ax.legend()
        ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.get_xticklabels(), rotation=45)
    fig.suptitle(f"Nuevos casos vs Nuevas muertes por millón ×100 (suavizados) — {pair[0]} y {pair[1]}", fontsize=15)
    fig.text(0.5, 0.01, "Fecha", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    filename = f"{pair[0].replace(' ', '_')}_{pair[1].replace(' ', '_')}_evolucion_per_million_scaled.png"
    plt.savefig(dir_plots_analyst_ev / filename, dpi=200)
    plt.show()

# --- 5.2) Correlación Casos vs Muertes por país ---
if "new_deaths_smoothed_per_million_scaled" in df_selected.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_selected, x="new_cases_smoothed_per_million", y="new_deaths_smoothed_per_million_scaled", hue="country", palette="tab10")
    plt.title("Relación entre Nuevos Casos y Nuevas Muertes ×100 por millón (suavizados)")
    plt.xlabel("Nuevos casos/millón (suavizados)"); plt.ylabel("Nuevas muertes/millón ×100 (suavizadas)")
    plt.legend(title="País", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(dir_plots_analyst_rel / "correlation_new_cases_new_deaths_per_million_scaled.png", dpi=200)
    plt.show()

# =============================================================================
# 6. GRÁFICOS DE RELACIONES CLAVE (SNAPSHOT)
# =============================================================================
print("Generando gráficos de relaciones entre variables (snapshot)...")
df_snapshot = df_countries_weekly.sort_values("week_start").groupby("country").tail(1).copy()
continent_styles = {"Africa": {"color": "#E69F00", "marker": "X"}, "Asia": {"color": "#56B4E9", "marker": "o"}, "Europe": {"color": "#009E73", "marker": "^"}, "North America": {"color": "#F0E442", "marker": "s"}, "South America": {"color": "#0072B2", "marker": "P"}, "Oceania": {"color": "#D55E00", "marker": "D"}}
pal = {continent: style["color"] for continent, style in continent_styles.items()}

# --- 6.1) SPONSOR (Valores Absolutos) ---
rel_cols_abs = ["country", "continent", "population", "median_age", "gdp_per_capita", "diabetes_prevalence", "extreme_poverty", "hospital_beds_per_thousand", "total_cases", "total_deaths"]
df_snapshot_abs = df_snapshot[[c for c in rel_cols_abs if c in df_snapshot.columns]].dropna(subset=["population", "total_cases", "total_deaths"])

# Casos vs Muertes y Población vs Casos (log-log)
for x_var, y_var in [("total_cases", "total_deaths"), ("population", "total_cases")]:
    plt.figure(figsize=(10, 8))
    for continent, style in continent_styles.items():
        data = df_snapshot_abs[df_snapshot_abs["continent"] == continent]
        if not data.empty:
            plt.scatter(data[x_var], data[y_var], s=35, alpha=0.7, label=continent, color=style["color"], marker=style["marker"], edgecolor='k', linewidth=0.5)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel(f"{x_var.replace('_', ' ')}"); plt.ylabel(f"{y_var.replace('_', ' ')}")
    plt.title(f"Continentes - {x_var.replace('_', ' ').title()} vs {y_var.replace('_', ' ').title()} (Escala Log-Log)")
    plt.grid(True, which="both", linestyle='--', alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig(dir_plots_sponsor_key / f"Continentes - {x_var}_vs_{y_var}_loglog.png", dpi=200, bbox_inches="tight")
    plt.show()

    
# --- 6.2) ANALYST (Valores por Millón) ---
rel_cols_pm = ["country", "continent", "population", "median_age", "gdp_per_capita", "diabetes_prevalence", "extreme_poverty", "hospital_beds_per_thousand", "total_cases_per_million", "total_deaths_per_million"]
df_snapshot_pm = df_snapshot[[c for c in rel_cols_pm if c in df_snapshot.columns]].dropna(subset=["population", "total_cases_per_million", "total_deaths_per_million"])

# Casos vs Muertes por millón y Población vs Casos por millón (log-log)
for x_var, y_var in [("total_cases_per_million", "total_deaths_per_million"), ("population", "total_cases_per_million")]:
    plt.figure(figsize=(10, 8))
    for continent, style in continent_styles.items():
        data = df_snapshot_pm[df_snapshot_pm["continent"] == continent]    
        if not data.empty:
            if x_var in data.columns and y_var in data.columns:
                plt.scatter(data[x_var], data[y_var], s=35, alpha=0.7, label=continent, color=style["color"], marker=style["marker"], edgecolor='k', linewidth=0.5)
                
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel(f"{x_var.replace('_', ' ').title()}")
    plt.ylabel(f"{y_var.replace('_', ' ').title()}")
    plt.title(f"Continentes - {x_var.replace('_', ' ').title()} vs {y_var.replace('_', ' ').title()} (Escala Log-Log)")
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_plots_analyst_rel / f"Continentes - {x_var}_vs_{y_var}_loglog.png", dpi=200, bbox_inches="tight")
    plt.show()
    
    
# Boxplots y Violinplots con stripplot superpuesto
for x_var, y_var, q_var, q_label, plot_type in [
    ("median_age", "total_deaths_per_million", "age_band", "Tramos de Edad Media (cuartiles)", "box"),
    ("extreme_poverty", "total_deaths_per_million", "poverty_band", "Tramos de Pobreza Extrema (cuartiles)", "violin"),
    ("hospital_beds_per_thousand", "total_deaths_per_million", "beds_q", "Cuartiles de Camas/1.000 hab.", "violin")
]:
    if not all(v in df_snapshot_pm.columns for v in [x_var, y_var, "continent"]): continue
    tmp = df_snapshot_pm.dropna(subset=[x_var, y_var, "continent"]).copy()
    tmp = tmp[tmp[y_var] > 0]
    if tmp.empty or tmp[x_var].nunique() < 4: continue
    
    tmp[q_var] = pd.qcut(tmp[x_var], q=4, duplicates="drop")
    
    plt.figure(figsize=(11, 7))
    if plot_type == "box":
        sns.boxplot(data=tmp, x=q_var, y=y_var, color="lightgray", showfliers=False)
    else:
        sns.violinplot(data=tmp, x=q_var, y=y_var, color="lightgray", inner="quartile", cut=0)
    
    sns.stripplot(data=tmp, x=q_var, y=y_var, hue="continent", dodge=True, alpha=0.7, size=4, jitter=0.25, palette=pal, linewidth=0.5, edgecolor='gray')
    plt.yscale("log")
    plt.title(f"{x_var.replace('_', ' ').title()} y Muertes por Millón (Escala Log)", fontsize=14)
    plt.xlabel(q_label); plt.ylabel("Muertes por Millón (escala log)")
    plt.legend(title="Continente", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, which="major", axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(dir_plots_analyst_rel / f"{x_var}_vs_deaths_pm_{plot_type}_log.png", dpi=200, bbox_inches="tight")
    plt.show()

print("\n--- Proceso de análisis exploratorio finalizado. ---")