import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import f_oneway, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

#  CONFIGURACIÓN Y RUTAS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_path = os.path.join(BASE_DIR, "data", "processed", "merged_sales_social.csv")
reports_fig = os.path.join(BASE_DIR, "reports", "figures")
reports_metrics = os.path.join(BASE_DIR, "reports", "metrics")

os.makedirs(reports_fig, exist_ok=True)
os.makedirs(reports_metrics, exist_ok=True)

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

print("=" * 70)
print(" ANÁLISIS AVANZADO DE VENTAS Y SENTIMIENTO")
print("=" * 70)
print(f" Cargando datos desde: {processed_path}\n")

# 1. CARGA Y VALIDACIÓN DE DATOS
try:
    df = pd.read_csv(processed_path)
    df["date"] = pd.to_datetime(df["date"])
except FileNotFoundError:
    print(f" ERROR: No se encontró el archivo {processed_path}")
    print("   Ejecuta primero: python run_all.py")
    exit(1)

print(f" Dataset cargado correctamente")
print(f"   • Registros: {len(df):,}")
print(f"   • Rango de fechas: {df['date'].min().date()} a {df['date'].max().date()}")
print(f"   • Columnas: {len(df.columns)}\n")

# Validar columnas críticas
required_cols = ["quantity_sold", "sentiment", "sales", "restaurant_type", "date"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f" ERROR: Columnas faltantes: {missing_cols}")
    exit(1)

# Mostrar nulos
nulls = df.isnull().sum()
if nulls.sum() > 0:
    print("  Valores nulos detectados:")
    print(nulls[nulls > 0])
    print()

# 2. analisis de tendencia temporal
print("\n" + "=" * 70)
print(" ANÁLISIS 1: TENDENCIA TEMPORAL DE VENTAS")
print("=" * 70)

daily_sales = df.groupby("date")["quantity_sold"].sum()
daily_revenue = df.groupby("date")["sales"].sum()

# Crear figura con dos subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# Gráfico 1: Cantidad vendida
ax1.plot(daily_sales.index, daily_sales.values, color='#2E86AB', linewidth=1.5)
ax1.fill_between(daily_sales.index, daily_sales.values, alpha=0.3, color='#2E86AB')
ax1.set_title("Tendencia Diaria de Cantidad Vendida", fontsize=14, fontweight='bold')
ax1.set_ylabel("Unidades Vendidas", fontsize=11)
ax1.grid(alpha=0.3)

# Gráfico 2: Ingresos
ax2.plot(daily_revenue.index, daily_revenue.values, color='#06A77D', linewidth=1.5)
ax2.fill_between(daily_revenue.index, daily_revenue.values, alpha=0.3, color='#06A77D')
ax2.set_title("Tendencia Diaria de Ingresos", fontsize=14, fontweight='bold')
ax2.set_xlabel("Fecha", fontsize=11)
ax2.set_ylabel("Ingresos ($)", fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(reports_fig, "tendencia_ventas_mejorada.png"), dpi=150, bbox_inches='tight')
print(f" Gráfico guardado: tendencia_ventas_mejorada.png")

# Estadísticas de tendencia
print(f"\n Estadísticas de Ventas:")
print(f"   • Promedio diario: {daily_sales.mean():.2f} unidades")
print(f"   • Desviación estándar: {daily_sales.std():.2f}")
print(f"   • Día con más ventas: {daily_sales.idxmax().date()} ({daily_sales.max():.0f} unidades)")
print(f"   • Día con menos ventas: {daily_sales.idxmin().date()} ({daily_sales.min():.0f} unidades)")

# 3. analisis de correlacion sentimiento ventas
print("\n" + "=" * 70)
print(" ANÁLISIS 2: CORRELACIÓN SENTIMIENTO vs VENTAS")
print("=" * 70)

daily_sentiment = df.groupby("date")["sentiment"].mean()

merged_trend = pd.DataFrame({
    "sales": daily_sales,
    "revenue": daily_revenue,
    "sentiment": daily_sentiment
}).dropna()

# Calcular correlaciones
corr_quantity = merged_trend["sales"].corr(merged_trend["sentiment"])
corr_revenue = merged_trend["revenue"].corr(merged_trend["sentiment"])

# Test de significancia
_, p_value_quantity = pearsonr(merged_trend["sales"], merged_trend["sentiment"])
_, p_value_revenue = pearsonr(merged_trend["revenue"], merged_trend["sentiment"])

print(f"\n Resultados de Correlación:")
print(f"   • Correlación Sentimiento-Cantidad: {corr_quantity:.4f} (p={p_value_quantity:.4f})")
print(f"   • Correlación Sentimiento-Ingresos: {corr_revenue:.4f} (p={p_value_revenue:.4f})")

# Interpretar correlación
if abs(corr_quantity) > 0.7:
    strength = "muy fuerte"
elif abs(corr_quantity) > 0.5:
    strength = "fuerte"
elif abs(corr_quantity) > 0.3:
    strength = "moderada"
elif abs(corr_quantity) > 0.1:
    strength = "débil"
else:
    strength = "muy débil o inexistente"

direction = "positiva" if corr_quantity > 0 else "negativa"
significance = "significativa" if p_value_quantity < 0.05 else "no significativa"

print(f"   • Interpretación: Correlación {strength} {direction} ({significance})")

# Gráfico de correlación mejorado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scatter con línea de regresión
sns.regplot(x="sentiment", y="sales", data=merged_trend, ax=ax1, 
            scatter_kws={'alpha': 0.5, 's': 50}, line_kws={'color': 'red', 'linewidth': 2})
ax1.set_title(f"Sentimiento vs Cantidad Vendida\n(r = {corr_quantity:.3f})", 
              fontsize=12, fontweight='bold')
ax1.set_xlabel("Sentimiento Promedio", fontsize=11)
ax1.set_ylabel("Unidades Vendidas", fontsize=11)
ax1.grid(alpha=0.3)

sns.regplot(x="sentiment", y="revenue", data=merged_trend, ax=ax2,
            scatter_kws={'alpha': 0.5, 's': 50}, line_kws={'color': 'red', 'linewidth': 2})
ax2.set_title(f"Sentimiento vs Ingresos\n(r = {corr_revenue:.3f})", 
              fontsize=12, fontweight='bold')
ax2.set_xlabel("Sentimiento Promedio", fontsize=11)
ax2.set_ylabel("Ingresos ($)", fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(reports_fig, "correlacion_sentimiento_mejorada.png"), dpi=150, bbox_inches='tight')
print(f" Gráfico guardado: correlacion_sentimiento_mejorada.png")

# 4. analisis anova
print("\n" + "=" * 70)
print(" ANÁLISIS 3: COMPARACIÓN ENTRE TIPOS DE RESTAURANTE")
print("=" * 70)

# Preparar grupos para ANOVA
restaurant_types = df["restaurant_type"].unique()
groups = [df[df["restaurant_type"] == rt]["quantity_sold"].values for rt in restaurant_types]

# Ejecutar ANOVA
anova_result = f_oneway(*groups)

print(f"\n Resultados ANOVA:")
print(f"   • F-statistic: {anova_result.statistic:.4f}")
print(f"   • p-value: {anova_result.pvalue:.6f}")

if anova_result.pvalue < 0.05:
    print(f"   •  Hay diferencias SIGNIFICATIVAS entre tipos de restaurante")
else:
    print(f"   •  NO hay diferencias significativas entre tipos de restaurante")

# Estadísticas por tipo
print(f"\n Estadísticas por Tipo de Restaurante:")
type_stats = df.groupby("restaurant_type").agg({
    "quantity_sold": ["mean", "std", "count"],
    "sales": "sum"
}).round(2)
print(type_stats)

# Calcular estadísticas para gráficos
type_summary = df.groupby("restaurant_type").agg({
    "quantity_sold": ["mean", "std"]
}).reset_index()
type_summary.columns = ["restaurant_type", "mean", "std"]

# Identificar mejor y peor desempeño
best_restaurant = type_summary.loc[type_summary["mean"].idxmax(), "restaurant_type"]
worst_restaurant = type_summary.loc[type_summary["mean"].idxmin(), "restaurant_type"]

print(f"\n Ranking de Desempeño:")
ranking = type_summary.sort_values("mean", ascending=False)
for idx, row in ranking.iterrows():
    print(f"   {row['restaurant_type']}: {row['mean']:.2f} unidades (±{row['std']:.2f})")

# Gráfico mejorado: Barras + Box plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: Barras simples y limpias
colors = sns.color_palette("Set2", len(type_summary))
bars = ax1.bar(type_summary["restaurant_type"], type_summary["mean"], 
               color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

ax1.set_title("Ventas Promedio por Tipo de Restaurante", 
              fontsize=12, fontweight='bold')
ax1.set_xlabel("Tipo de Restaurante", fontsize=11)
ax1.set_ylabel("Unidades Vendidas (Promedio)", fontsize=11)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(alpha=0.3, axis='y')

# Añadir valores sobre las barras
for bar, mean_val in zip(bars, type_summary["mean"]):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 2, 
             f'{mean_val:.1f}', ha='center', va='bottom', 
             fontsize=11, fontweight='bold')

# Resaltar mejor y peor restaurante
for bar, restaurant in zip(bars, type_summary["restaurant_type"]):
    if restaurant == best_restaurant:
        bar.set_edgecolor('green')
        bar.set_linewidth(3)
        bar.set_alpha(1.0)
    elif restaurant == worst_restaurant:
        bar.set_edgecolor('red')
        bar.set_linewidth(3)
        bar.set_alpha(1.0)

# Gráfico 2: Box plot con puntos
sns.boxplot(data=df, x="restaurant_type", y="quantity_sold", ax=ax2, 
            palette="Set2", linewidth=1.5)
sns.stripplot(data=df, x="restaurant_type", y="quantity_sold", ax=ax2, 
              color='black', alpha=0.08, size=1.5)

ax2.set_title("Distribución de Ventas por Tipo de Restaurante", 
              fontsize=12, fontweight='bold')
ax2.set_xlabel("Tipo de Restaurante", fontsize=11)
ax2.set_ylabel("Cantidad Vendida", fontsize=11)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(alpha=0.3, axis='y')

# Añadir línea de media global
global_mean = df["quantity_sold"].mean()
ax2.axhline(y=global_mean, color='red', linestyle='--', 
            linewidth=2, alpha=0.7, label=f'Media Global: {global_mean:.1f}')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(reports_fig, "anova_restaurant_mejorado.png"), 
            dpi=150, bbox_inches='tight')
print(f" Gráfico guardado: anova_restaurant_mejorado.png")

# Análisis adicional: Heatmap de desempeño temporal (estacionalidad)
print(f"\n Generando análisis de estacionalidad por tipo...")
df['month'] = df['date'].dt.month
df['month_name'] = df['date'].dt.strftime('%b')

# Crear matriz de desempeño mensual
performance_matrix = df.groupby(['restaurant_type', 'month_name'])['quantity_sold'].mean().unstack(fill_value=0)

# Ordenar meses correctamente
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
available_months = [m for m in month_order if m in performance_matrix.columns]
performance_matrix = performance_matrix[available_months]

# Crear heatmap
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(performance_matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={'label': 'Ventas Promedio (unidades)'}, 
            linewidths=0.5, linecolor='gray', ax=ax)

ax.set_title("Desempeño Mensual por Tipo de Restaurante\n(Análisis de Estacionalidad)", 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("Mes", fontsize=11)
ax.set_ylabel("Tipo de Restaurante", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(reports_fig, "heatmap_estacionalidad_tipo.png"), 
            dpi=150, bbox_inches='tight')
print(f" Gráfico guardado: heatmap_estacionalidad_tipo.png")

# Identificar mejor mes por tipo de restaurante
print(f"\n Mejor mes por tipo de restaurante:")
for restaurant in performance_matrix.index:
    best_month = performance_matrix.loc[restaurant].idxmax()
    best_value = performance_matrix.loc[restaurant].max()
    print(f"   • {restaurant}: {best_month} ({best_value:.1f} unidades)")

# 5. MODELO ARIMA - PREDICCIÓN DE VENTAS
print("\n" + "=" * 70)
print(" ANÁLISIS 4: PRONÓSTICO DE VENTAS (ARIMA)")
print("=" * 70)

# Preparar serie temporal
sales_series = daily_sales.asfreq("D").ffill().bfill()
sales_series = sales_series.replace([np.inf, -np.inf], np.nan).dropna()

# Split train/test para validación
train_size = int(len(sales_series) * 0.9)
train, test = sales_series[:train_size], sales_series[train_size:]

print(f"\n Datos para modelado:")
print(f"   • Total de días: {len(sales_series)}")
print(f"   • Entrenamiento: {len(train)} días")
print(f"   • Prueba: {len(test)} días")

try:
    # Entrenar modelo
    print("\n Entrenando modelo ARIMA(5,1,2)...")
    model = ARIMA(train, order=(5, 1, 2))
    model_fit = model.fit()
    
    # Hacer predicciones sobre datos de prueba
    predictions = model_fit.forecast(steps=len(test))
    
    # Calcular métricas
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mape = np.mean(np.abs((test - predictions) / test)) * 100
    
    print(f"\n Modelo entrenado exitosamente")
    print(f"\n Métricas de Validación:")
    print(f"   • MAE (Error Absoluto Medio): {mae:.2f} unidades")
    print(f"   • RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f} unidades")
    print(f"   • MAPE (Error Porcentual Medio): {mape:.2f}%")
    
    # Pronóstico futuro (7 días)
    future_forecast = model_fit.forecast(steps=7)
    
    # Gráfico mejorado
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Subplot 1: Histórico completo + predicción
    ax1.plot(train.index, train.values, label="Entrenamiento", color='#2E86AB', linewidth=1.5)
    ax1.plot(test.index, test.values, label="Datos Reales (Test)", color='#06A77D', linewidth=1.5)
    ax1.plot(test.index, predictions, label="Predicción (Test)", 
             color='#F18F01', linestyle='--', linewidth=2)
    ax1.set_title("Validación del Modelo ARIMA", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Fecha", fontsize=11)
    ax1.set_ylabel("Unidades Vendidas", fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    
    # Subplot 2: Pronóstico futuro
    ax2.plot(sales_series.index, sales_series.values, 
             label="Histórico Completo", color='#2E86AB', linewidth=1.5)
    ax2.plot(future_forecast.index, future_forecast.values, 
             label="Pronóstico (7 días)", color='#C73E1D', 
             linestyle='--', linewidth=2.5, marker='o', markersize=6)
    ax2.axvline(x=sales_series.index[-1], color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax2.set_title("Pronóstico de Ventas - Próximos 7 Días", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Fecha", fontsize=11)
    ax2.set_ylabel("Unidades Vendidas", fontsize=11)
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(reports_fig, "arima_forecast_mejorado.png"), dpi=150, bbox_inches='tight')
    print(f" Gráfico guardado: arima_forecast_mejorado.png")
    
    print(f"\n Pronóstico para los próximos 7 días:")
    for i, (date, value) in enumerate(future_forecast.items(), 1):
        print(f"   • Día {i} ({date.date()}): {value:.0f} unidades")
    
except Exception as e:
    print(f" Error al entrenar ARIMA: {e}")
    future_forecast = None
    model_fit = None

# 6. guardar resultados numericos
print("\n" + "=" * 70)
print(" GUARDANDO RESULTADOS")
print("=" * 70)

with open(os.path.join(reports_metrics, "analysis_results.txt"), "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("REPORTE DE ANÁLISIS - RESTAURANT SALES\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("1. CORRELACIÓN SENTIMIENTO-VENTAS\n")
    f.write("-" * 50 + "\n")
    f.write(f"Correlación (Cantidad): {corr_quantity:.4f}\n")
    f.write(f"P-value: {p_value_quantity:.6f}\n")
    f.write(f"Correlación (Ingresos): {corr_revenue:.4f}\n")
    f.write(f"P-value: {p_value_revenue:.6f}\n\n")
    
    f.write("2. RESULTADO ANOVA\n")
    f.write("-" * 50 + "\n")
    f.write(f"F-statistic: {anova_result.statistic:.4f}\n")
    f.write(f"P-value: {anova_result.pvalue:.6f}\n\n")
    
    if future_forecast is not None:
        f.write("3. PRONÓSTICO ARIMA (7 días)\n")
        f.write("-" * 50 + "\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAPE: {mape:.2f}%\n\n")
        f.write("Pronóstico:\n")
        for date, value in future_forecast.items():
            f.write(f"  {date.date()}: {value:.0f} unidades\n")

print(f" Resultados guardados en: analysis_results.txt")

# 7. RESUMEN 
print("\n" + "=" * 70)
print(" RESUMEN")
print("=" * 70 + "\n")

# Sentimiento vs Ventas
if abs(corr_quantity) > 0.3 and p_value_quantity < 0.05:
    direction = "positiva" if corr_quantity > 0 else "negativa"
    print(f" SENTIMIENTO: Correlación {direction} significativa detectada")
    print(f"   → El sentimiento en redes sociales {'SI' if corr_quantity > 0 else 'NO'} impacta las ventas")
else:
    print(f"  SENTIMIENTO: No se detectó relación significativa con las ventas")

# ANOVA
if anova_result.pvalue < 0.05:
    print(f"\n TIPOS DE RESTAURANTE: Diferencias significativas encontradas")
    best_type = df.groupby("restaurant_type")["quantity_sold"].mean().idxmax()
    print(f"   → Mejor desempeño: {best_type}")
else:
    print(f"\n  TIPOS DE RESTAURANTE: No hay diferencias significativas")

# Pronóstico
if future_forecast is not None:
    avg_forecast = future_forecast.mean()
    trend = "alza" if avg_forecast > daily_sales.mean() else "baja"
    print(f"\n PRONÓSTICO: Generado para los próximos 7 días")
    print(f"   → Tendencia: {trend}")
    print(f"   → Promedio pronosticado: {avg_forecast:.0f} unidades/día")

print("\n" + "=" * 70)
print(" ANÁLISIS COMPLETO FINALIZADO CON ÉXITO")
print("=" * 70)
print(f"\n Revisa los resultados en:")
print(f"   • Gráficos: {reports_fig}")
print(f"   • Métricas: {reports_metrics}")
print()