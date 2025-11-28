import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
# CONFIGURACIÓN GENERAL
st.set_page_config(page_title="DataShop - Dashboard Analítico", layout="wide")

DATA_PATH = "data/processed/merged_sales_social.csv"

st.title(" DataShop — Dashboard Analítico de Ventas y Sentimiento")
st.write("Análisis completo: Desempeño, Estacionalidad, ARIMA, ANOVA y Correlación Sentimiento-Ventas")
st.write("---")

# CARGA DE DATOS
@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        # Agregar columnas derivadas de fecha
        df["day_of_week"] = df["date"].dt.day_name()
        df["month"] = df["date"].dt.month_name()
        df["year"] = df["date"].dt.year
        df["week"] = df["date"].dt.isocalendar().week
        return df
    except FileNotFoundError:
        st.error(f" No se encontró el archivo: {path}")
        return None

df = load_data(DATA_PATH)
if df is None:
    st.stop()

# SIDEBAR — FILTROS GLOBALES
st.sidebar.header(" Filtros Globales")

# Filtro de fecha
min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input(
    " Rango de fechas",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Filtro de restaurante
restaurants = ["Todos"] + sorted(df["restaurant_id"].astype(str).unique())
selected_restaurant = st.sidebar.selectbox(" Restaurante", restaurants)

# Filtro de tipo de restaurante
restaurant_types = ["Todos"] + sorted(df["restaurant_type"].unique())
selected_rest_type = st.sidebar.selectbox(" Tipo de Restaurante", restaurant_types)

# Filtro de tipo de comida
meal_types = ["Todos"] + sorted(df["meal_type"].unique())
selected_meal = st.sidebar.selectbox(" Tipo de comida", meal_types)

# Filtro de promoción
promo_filter = st.sidebar.selectbox(" Promoción", ["Todos", "Sí", "No"])

# Aplicar filtros
if len(date_range) == 2:
    df_filtered = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])]
else:
    df_filtered = df.copy()

if selected_restaurant != "Todos":
    df_filtered = df_filtered[df_filtered["restaurant_id"].astype(str) == selected_restaurant]

if selected_rest_type != "Todos":
    df_filtered = df_filtered[df_filtered["restaurant_type"] == selected_rest_type]

if selected_meal != "Todos":
    df_filtered = df_filtered[df_filtered["meal_type"] == selected_meal]

if promo_filter == "Sí":
    df_filtered = df_filtered[df_filtered["has_promotion"] == 1]
elif promo_filter == "No":
    df_filtered = df_filtered[df_filtered["has_promotion"] == 0]

st.sidebar.write("---")
st.sidebar.info(f" **{len(df_filtered):,}** registros seleccionados")

# MÉTRICAS PRINCIPALES (KPIs)
st.header(" Métricas Clave")

col1, col2, col3, col4 = st.columns(4)

total_sales = df_filtered["sales"].sum()
total_quantity = df_filtered["quantity_sold"].sum()
avg_sentiment = df_filtered["sentiment"].mean()
unique_items = df_filtered["menu_item_name"].nunique()

col1.metric(" Ventas Totales", f"${total_sales:,.2f}")
col2.metric(" Cantidad Vendida", f"{total_quantity:,}")
col3.metric(" Sentimiento Promedio", f"{avg_sentiment:.3f}")
col4.metric(" Productos Únicos", unique_items)

st.write("---")

# NAVEGACIÓN POR TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Desempeño General", 
    " Estacionalidad", 
    " Pronóstico ARIMA",
    " ANOVA (Comparación)",
    " Sentimiento vs Ventas"
])

# TAB 1: DESEMPEÑO GENERAL
with tab1:
    st.header(" Análisis de Desempeño")
    
    # Top 10 productos más vendidos
    st.subheader(" Top 10 Productos Más Vendidos")
    top_items = (
        df_filtered.groupby("menu_item_name")["quantity_sold"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    
    if len(top_items) > 0:
        fig_top = px.bar(
            top_items,
            x="quantity_sold",
            y="menu_item_name",
            orientation="h",
            title="Top 10 Productos",
            text="quantity_sold",
            color="quantity_sold",
            color_continuous_scale="Blues"
        )
        fig_top.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_top.update_layout(showlegend=False)
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("No hay datos suficientes para mostrar el Top 10.")
    st.write("---")
    
    # Ventas por tipo de restaurante
    st.subheader(" Desempeño por Tipo de Restaurante")
    
    res_type_sales = (
        df_filtered.groupby("restaurant_type").agg({
            "sales": "sum",
            "quantity_sold": "sum"
        }).reset_index().sort_values("sales", ascending=False)
    )
    
    if len(res_type_sales) > 0:
        fig_rest = px.bar(
            res_type_sales,
            x="sales",
            y="restaurant_type",
            orientation="h",
            title="Ventas Totales por Tipo de Restaurante",
            text="sales",
            color="sales",
            color_continuous_scale="Viridis"
        )
        fig_rest.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_rest.update_layout(
            showlegend=False,
            xaxis_title="Ventas ($)",
            yaxis_title="Tipo de Restaurante"
        )
        st.plotly_chart(fig_rest, use_container_width=True)
        
        # Métricas de desempeño
        col1, col2, col3 = st.columns(3)
        
        best_restaurant = res_type_sales.iloc[0]
        worst_restaurant = res_type_sales.iloc[-1]
        
        col1.metric(" Mejor Tipo", best_restaurant["restaurant_type"],f"${best_restaurant['sales']:,.0f}")
        col2.metric(" Menor Tipo", worst_restaurant["restaurant_type"],f"${worst_restaurant['sales']:,.0f}")
        col3.metric(" Diferencia", f"${best_restaurant['sales'] - worst_restaurant['sales']:,.0f}")
    else:
        st.info("No hay datos suficientes.")
    st.write("---")
    
    # Ventas por tipo de comida
    st.subheader(" Ventas por Tipo de Comida")
    
    meal_sales = (
        df_filtered.groupby("meal_type")["sales"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    
    if len(meal_sales) > 0:
        fig_meal = px.pie(
            meal_sales,
            values="sales",
            names="meal_type",
            title="Distribución de Ventas por Tipo de Comida",
            hole=0.4
        )
        st.plotly_chart(fig_meal, use_container_width=True)
    else:
        st.info("No hay datos suficientes.")
    st.write("---")
    
    # Impacto de promociones
    st.subheader(" Impacto de Promociones")
    
    promo_impact = df_filtered.groupby("has_promotion").agg({
        "sales": "sum",
        "quantity_sold": "sum"
    }).reset_index()
    promo_impact["has_promotion"] = promo_impact["has_promotion"].map({0: "Sin Promoción", 1: "Con Promoción"})
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_promo_sales = px.bar(
            promo_impact,
            x="has_promotion",
            y="sales",
            title="Ventas: Con vs Sin Promoción",
            text="sales",
            color="sales",
            color_continuous_scale="Blues"
        )
        fig_promo_sales.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_promo_sales.update_layout(showlegend=False)
        st.plotly_chart(fig_promo_sales, use_container_width=True)
    
    with col2:
        fig_promo_qty = px.bar(
            promo_impact,
            x="has_promotion",
            y="quantity_sold",
            title="Cantidad Vendida: Con vs Sin Promoción",
            text="quantity_sold",
            color="quantity_sold",
            color_continuous_scale="Greens"
        )
        fig_promo_qty.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_promo_qty.update_layout(showlegend=False)
        st.plotly_chart(fig_promo_qty, use_container_width=True)

# TAB 2: ESTACIONALIDAD
with tab2:
    st.header(" Análisis de Estacionalidad")
    
    # 1. Patrón Semanal
    st.subheader(" Patrón Semanal de Ventas")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    weekly_sales = (
        df_filtered.groupby("day_of_week")["sales"]
        .sum()
        .reindex(day_order)
        .reset_index()
    )
    
    if len(weekly_sales) > 0 and not weekly_sales["sales"].isna().all():
        fig_weekly = px.bar(
            weekly_sales,
            x="day_of_week",
            y="sales",
            title="Ventas Totales por Día de la Semana",
            color="sales",
            color_continuous_scale="Viridis",
            text="sales"
        )
        fig_weekly.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_weekly.update_layout(
            showlegend=False,
            xaxis_title="Día de la Semana",
            yaxis_title="Ventas ($)"
        )
        st.plotly_chart(fig_weekly, use_container_width=True)
        
        # Métricas de estacionalidad semanal
        weekly_sales_clean = weekly_sales.dropna()
        if len(weekly_sales_clean) > 0:
            best_day = weekly_sales_clean.loc[weekly_sales_clean["sales"].idxmax(), "day_of_week"]
            worst_day = weekly_sales_clean.loc[weekly_sales_clean["sales"].idxmin(), "day_of_week"]
            max_sales = weekly_sales_clean["sales"].max()
            min_sales = weekly_sales_clean["sales"].min()
            variation = ((max_sales / min_sales - 1) * 100) if min_sales > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric(" Mejor Día", best_day)
            col2.metric(" Peor Día", worst_day)
            col3.metric(" Variación", f"{variation:.1f}%")
    else:
        st.info("No hay datos suficientes para análisis semanal.")
    st.write("---")
    
    # 2. Patrón Mensual
    st.subheader(" Tendencia Mensual de Ventas")
    
    monthly_sales = (
        df_filtered.groupby(df_filtered["date"].dt.to_period("M"))["sales"]
        .sum()
        .reset_index()
    )
    monthly_sales["date"] = monthly_sales["date"].dt.to_timestamp()
    
    if len(monthly_sales) > 0:
        fig_monthly = px.line(
            monthly_sales,
            x="date",
            y="sales",
            title="Evolución Mensual de Ventas",
            markers=True
        )
        fig_monthly.update_traces(line_color='#06A77D', marker=dict(size=10))
        st.plotly_chart(fig_monthly, use_container_width=True)
    else:
        st.info("No hay datos suficientes para análisis mensual.")
    
    st.write("---")
    
    # 3. Heatmap: Día de semana vs Tipo de Restaurante
    st.subheader(" Heatmap: Desempeño por Día y Tipo de Restaurante")
    
    heatmap_data = df_filtered.groupby(["day_of_week", "restaurant_type"])["sales"].sum().reset_index()
    
    if len(heatmap_data) > 0:
        heatmap_pivot = heatmap_data.pivot(
            index="restaurant_type", 
            columns="day_of_week", 
            values="sales"
        )
        
        # Reordenar columnas si existen todos los días
        available_days = [d for d in day_order if d in heatmap_pivot.columns]
        if len(available_days) > 0:
            heatmap_pivot = heatmap_pivot[available_days]
        
        fig_heatmap = px.imshow(
            heatmap_pivot,
            labels=dict(x="Día de la Semana", y="Tipo de Restaurante", color="Ventas ($)"),
            title="Mapa de Calor: Ventas por Día y Tipo de Restaurante",
            color_continuous_scale="YlOrRd",
            text_auto=".0f",
            aspect="auto"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Identificar mejor combinación
        best_combo = heatmap_data.loc[heatmap_data["sales"].idxmax()]
        st.success(f" **Mejor combinación:** {best_combo['restaurant_type']} en {best_combo['day_of_week']} (${best_combo['sales']:,.0f})")
    else:
        st.info("No hay datos suficientes para el heatmap.")
    
    st.write("---")
    
    # 4. Análisis por clima
    st.subheader(" Ventas por Condición Climática")
    
    weather_sales = (
        df_filtered.groupby("weather_condition")["sales"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    
    if len(weather_sales) > 0:
        fig_weather = px.bar(
            weather_sales,
            x="weather_condition",
            y="sales",
            title="Ventas por Condición Climática",
            text="sales",
            color="sales",
            color_continuous_scale="Blues"
        )
        fig_weather.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_weather.update_layout(showlegend=False)
        st.plotly_chart(fig_weather, use_container_width=True)
    else:
        st.info("No hay datos suficientes.")

# TAB 3: PRONÓSTICO ARIMA
with tab3:
    st.header(" Pronóstico de Ventas (ARIMA)")
    
    try:
        from statsmodels.tsa.arima.model import ARIMA
        import numpy as np
        
        # Preparar datos diarios (usar df completo, no filtrado)
        df_arima = df.groupby("date")["sales"].sum().reset_index()
        df_arima["date"] = pd.to_datetime(df_arima["date"])
        df_arima = df_arima.set_index("date").asfreq("D")
        
        # Rellenar valores faltantes
        df_arima = df_arima.fillna(method='ffill').fillna(method='bfill')
        
        # Control de días a pronosticar
        forecast_days = st.slider(" Días a pronosticar:", 1, 30, 7)
        
        # Entrenar modelo
        with st.spinner(" Entrenando modelo ARIMA..."):
            model = ARIMA(df_arima["sales"], order=(5, 1, 2))
            model_fit = model.fit()
        
        # Generar pronóstico
        forecast = model_fit.forecast(steps=forecast_days)
        
        # Crear dataframe de pronóstico
        forecast_dates = pd.date_range(
            start=df_arima.index.max() + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D"
        )
        forecast_df = pd.DataFrame({
            "date": forecast_dates,
            "sales": forecast,
            "type": "Pronóstico"
        })
        
        # Datos históricos (últimos 30 días)
        historical_df = df_arima.tail(30).reset_index()
        historical_df["type"] = "Histórico"
        
        # Combinar
        combined_df = pd.concat([historical_df, forecast_df])
        
        # Visualizar
        fig_forecast = px.line(
            combined_df,
            x="date",
            y="sales",
            color="type",
            title=f"Pronóstico de Ventas - Próximos {forecast_days} Días",
            color_discrete_map={"Histórico": "#2E86AB", "Pronóstico": "#C73E1D"}
        )
        fig_forecast.update_traces(mode='lines+markers')
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Métricas del pronóstico
        col1, col2, col3 = st.columns(3)
        col1.metric(" Promedio Pronosticado", f"${forecast.mean():,.2f}")
        col2.metric(" Máximo Esperado", f"${forecast.max():,.2f}")
        col3.metric(" Mínimo Esperado", f"${forecast.min():,.2f}")
        
        # Tabla de valores pronosticados
        st.markdown("####  Valores Pronosticados")
        forecast_table = forecast_df[["date", "sales"]].copy()
        forecast_table["sales"] = forecast_table["sales"].apply(lambda x: f"${x:,.2f}")
        forecast_table["date"] = forecast_table["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)
        
        # Información del modelo
        with st.expander(" Información del Modelo ARIMA"):
            st.write(f"**Orden del modelo:** ARIMA(5, 1, 2)")
            st.write(f"**AIC:** {model_fit.aic:.2f}")
            st.write(f"**BIC:** {model_fit.bic:.2f}")
        
    except Exception as e:
        st.error(f"Error al generar pronóstico: {e}")
        st.info(" Asegúrate de tener suficientes datos históricos (mínimo 30 días)")

# TAB 4: ANOVA
with tab4:
    st.header(" ANOVA: Comparación entre Grupos")
    
    # Selector de variable a comparar
    comparison_var = st.selectbox(
        "🔍 Selecciona la variable a comparar:",
        ["meal_type", "restaurant_type", "has_promotion", "weather_condition"],
        format_func=lambda x: {
            "meal_type": "Tipo de Comida",
            "restaurant_type": "Tipo de Restaurante",
            "has_promotion": "Con/Sin Promoción",
            "weather_condition": "Condición Climática"
        }[x]
    )
    
    try:
        from scipy.stats import f_oneway
        
        # Preparar grupos
        groups = [group["sales"].values for _, group in df_filtered.groupby(comparison_var) if len(group) > 0]
        
        if len(groups) < 2:
            st.warning(" No hay suficientes grupos para realizar ANOVA")
        else:
            # Ejecutar ANOVA
            anova_stat, anova_p = f_oneway(*groups)
            
            # Mostrar resultados
            st.subheader(" Resultados del Test ANOVA")
            
            col1, col2 = st.columns(2)
            col1.metric(" F-statistic", f"{anova_stat:.4f}")
            col2.metric(" p-valor", f"{anova_p:.6f}")
            
            # Interpretación
            if anova_p < 0.05:
                st.success(" **Conclusión:** Hay diferencias SIGNIFICATIVAS entre los grupos (p < 0.05)")
            else:
                st.warning(" **Conclusión:** NO hay diferencias significativas entre los grupos (p ≥ 0.05)") 
            st.write("---")
            
            # Visualización de comparación
            st.subheader(" Comparación Visual entre Grupos")
            
            group_stats = df_filtered.groupby(comparison_var).agg({
                "sales": ["mean", "std", "count", "sum"]
            }).reset_index()
            group_stats.columns = [comparison_var, "mean", "std", "count", "total"]
            group_stats = group_stats.sort_values("mean", ascending=False)
            
            # Gráfico de barras con error bars
            fig_anova = px.bar(
                group_stats,
                x=comparison_var,
                y="mean",
                error_y="std",
                title=f"Ventas Promedio por {comparison_var.replace('_', ' ').title()}",
                text="mean",
                color="mean",
                color_continuous_scale="Blues"
            )
            fig_anova.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig_anova.update_layout(
                showlegend=False,
                xaxis_title=comparison_var.replace('_', ' ').title(),
                yaxis_title="Ventas Promedio ($)"
            )
            st.plotly_chart(fig_anova, use_container_width=True)
            
            # Box plot para ver distribuciones
            st.subheader(" Distribución de Ventas por Grupo")
            fig_box = px.box(
                df_filtered,
                x=comparison_var,
                y="sales",
                title="Distribución de Ventas",
                color=comparison_var
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Tabla de estadísticas
            st.subheader(" Estadísticas Detalladas por Grupo")
            display_stats = group_stats.copy()
            display_stats["mean"] = display_stats["mean"].apply(lambda x: f"${x:,.2f}")
            display_stats["std"] = display_stats["std"].apply(lambda x: f"${x:,.2f}")
            display_stats["total"] = display_stats["total"].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display_stats, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f" Error al ejecutar ANOVA: {e}")

# TAB 5: SENTIMIENTO VS VENTAS
with tab5:
    st.header(" Análisis: Sentimiento vs Ventas Diarias")
    
    # Agregación diaria
    daily_analysis = df_filtered.groupby("date").agg({
        "sales": "sum",
        "sentiment": "mean",
        "quantity_sold": "sum"
    }).reset_index()
    
    if len(daily_analysis) > 1:
        # Calcular correlaciones
        corr_sales = daily_analysis["sales"].corr(daily_analysis["sentiment"])
        corr_quantity = daily_analysis["quantity_sold"].corr(daily_analysis["sentiment"])
        
        # Mostrar correlaciones
        st.subheader(" Coeficientes de Correlación")
        
        col1, col2 = st.columns(2)
        col1.metric(" Correlación (Ventas $)", f"{corr_sales:.3f}")
        col2.metric(" Correlación (Cantidad)", f"{corr_quantity:.3f}")
        
        # Interpretación
        st.markdown("####  Interpretación")
        
        if abs(corr_sales) > 0.7:
            strength = "MUY FUERTE"
        elif abs(corr_sales) > 0.5:
            strength = "FUERTE"
        elif abs(corr_sales) > 0.3:
            strength = "MODERADA"
        elif abs(corr_sales) > 0.1:
            strength = "DÉBIL"
        else:
            strength = "MUY DÉBIL o INEXISTENTE"
        
        direction = "POSITIVA" if corr_sales > 0 else "NEGATIVA"
        
        st.info(f"**Correlación {strength} {direction}** entre sentimiento y ventas")
        
        if abs(corr_sales) > 0.5:
            st.success(" El sentimiento en redes sociales tiene un impacto significativo en las ventas")
        else:
            st.warning(" El impacto del sentimiento en ventas es limitado. Otros factores pueden ser más relevantes.")
        
        st.write("---")
        
        # Scatter plot con línea de tendencia
        st.subheader(" Gráfico de Dispersión: Sentimiento vs Ventas")
        
        fig_corr = px.scatter(
            daily_analysis,
            x="sentiment",
            y="sales",
            size="quantity_sold",
            trendline="ols",
            title="Relación entre Sentimiento Diario y Ventas",
            labels={
                "sentiment": "Sentimiento Promedio Diario",
                "sales": "Ventas Totales ($)",
                "quantity_sold": "Cantidad Vendida"
            }
        )
        fig_corr.update_traces(marker=dict(opacity=0.7))
        st.plotly_chart(fig_corr, use_container_width=True)
        st.write("---")
        
        # Serie temporal dual (dos ejes Y)
        st.subheader(" Evolución Temporal: Ventas vs Sentimiento")
        
        fig_dual = go.Figure()
        
        # Ventas
        fig_dual.add_trace(go.Scatter(
            x=daily_analysis["date"],
            y=daily_analysis["sales"],
            name="Ventas ($)",
            line=dict(color="#2E86AB", width=2),
            yaxis="y1"
        ))
        
        # Sentimiento
        fig_dual.add_trace(go.Scatter(
            x=daily_analysis["date"],
            y=daily_analysis["sentiment"],
            name="Sentimiento",
            line=dict(color="#F18F01", width=2),
            yaxis="y2"
        ))
        
        fig_dual.update_layout(
            title="Evolución Temporal: Ventas vs Sentimiento (Ejes Independientes)",
            xaxis=dict(title="Fecha"),
            yaxis=dict(title="Ventas ($)", side="left", showgrid=False),
            yaxis2=dict(title="Sentimiento", side="right", overlaying="y", showgrid=False),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_dual, use_container_width=True)
        
        # Análisis adicional: Sentimiento por tipo de restaurante
        st.write("---")
        st.subheader(" Sentimiento Promedio por Tipo de Restaurante")
        
        sentiment_by_type = df_filtered.groupby("restaurant_type").agg({
            "sentiment": "mean",
            "sales": "sum"
        }).reset_index().sort_values("sentiment", ascending=False)
        
        fig_sent_type = px.bar(
            sentiment_by_type,
            x="restaurant_type",
            y="sentiment",
            title="Sentimiento Promedio por Tipo de Restaurante",
            text="sentiment",
            color="sentiment",
            color_continuous_scale="RdYlGn"
        )
        fig_sent_type.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_sent_type.update_layout(showlegend=False)
        st.plotly_chart(fig_sent_type, use_container_width=True)
    
    else:
        st.info("No hay suficientes datos para calcular correlaciones. Selecciona un rango de fechas más amplio.")


# SECCIÓN ADICIONAL: DATOS Y DESCARGAS
st.write("---")
st.header(" Datos y Exportación")

with st.expander(" Ver Datos Filtrados"):
    st.dataframe(df_filtered, use_container_width=True)

with st.expander("Resumen Estadístico"):
    st.dataframe(df_filtered.describe(), use_container_width=True)

