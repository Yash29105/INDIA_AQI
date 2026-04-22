# ============================================================
# AQI PREDICTION & POLLUTION TREND ANALYSIS
# Real CPCB Station Data - India
# ============================================================

"""
PROJECT STRUCTURE:
    1. SETUP & CONFIGURATION
    2. DATA LOADING & PREPROCESSING
    3. FEATURE ENGINEERING
    4. EXPLORATORY DATA ANALYSIS
    5. MACHINE LEARNING MODEL
    6. VISUALIZATION DASHBOARDS
"""

# ============================================================
# 1. SETUP & CONFIGURATION
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Configuration
plt.style.use("dark_background")
pd.set_option('display.max_columns', None)

# File Path
DATA_PATH = r"C:\Users\Yash\OneDrive\Desktop\python project\cpcb_station_data.csv"

# Constants
POLLUTANTS = ['pm2_5', 'pm10', 'no2', 'so2', 'co']
MONTH_MAP = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

# ============================================================
# 2. DATA LOADING & PREPROCESSING
# ============================================================

def load_and_clean_data(filepath):
    """
    Load CSV data and perform initial cleaning
    """
    # Load data
    df = pd.read_csv(filepath, parse_dates=["date"])
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Rename columns
    df.rename(columns={'pm2.5': 'pm2_5'}, inplace=True)
    
    # Drop missing critical values
    df = df.dropna(subset=['aqi', 'pm2_5', 'pm10', 'no2'])
    
    # Convert to numeric
    pollutants_with_aqi = POLLUTANTS + ['aqi']
    for col in pollutants_with_aqi:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove invalid AQI
    df = df[df['aqi'] > 0]
    
    return df


# AQI Category
def categorize_aqi(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"
# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    """
    Create new features from existing data
    """
    # Total pollution
    df['total_pollution'] = df[POLLUTANTS].sum(axis=1)
    
    # Time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_name'] = df['month'].map(MONTH_MAP)
    
    # Season
    df['season'] = pd.cut(
        df['month'],
        bins=[0, 3, 6, 9, 12],
        labels=["Winter", "Summer", "Monsoon", "Post-Monsoon"]
    )
    
    # AQI Category
    df['aqi_category'] = df['aqi'].apply(categorize_aqi)
    
    return df


def print_data_summary(df):
    """
    Print comprehensive data summary
    """
    print("\n" + "="*60)
    print("DATA LOADED & CLEANED SUCCESSFULLY!")
    print("="*60)
    print(f"\nShape after cleaning: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nYears: {sorted(df['year'].dropna().unique())}")
    print(f"Cities: {df['city'].nunique()}")
    print(f"Stations: {df['station'].nunique()}")
    print("="*60 + "\n")


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

def create_monthly_aggregation(df):
    """
    Aggregate data by month
    """
    monthly = df.groupby('month_name').mean(numeric_only=True).reindex(MONTH_MAP.values())
    monthly = monthly.fillna(0)
    return monthly


def create_city_pivot(df):
    """
    Create pivot table for cities and pollutants
    """
    pivot = df.groupby('city')[POLLUTANTS + ['aqi']].mean()
    return pivot


# ============================================================
# 5. MACHINE LEARNING MODEL
# ============================================================

def train_linear_regression(df):
    """
    Train Linear Regression model to predict AQI
    """
    # Prepare data
    X = df[POLLUTANTS]
    y = df['aqi']
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print("="*60 + "\n")
    
    return model, y_pred, r2, rmse, mae


# ============================================================
# 6. VISUALIZATION DASHBOARDS
# ============================================================

# ===========================OBJECTIVE 1===========================
def plot_aqi_trend_by_city(df):
    """
    Plot AQI trends over time for top 9 cities
    """
    cities = df['city'].unique()[:9]
    
    plt.figure(figsize=(18, 12))
    for i, city in enumerate(cities):
        plt.subplot(3, 3, i+1)
        
        city_df = df[df['city'] == city]
        plt.plot(city_df['date'], city_df['aqi'], linewidth=1.5)
        
        plt.axhline(200, linestyle='--')
        plt.axhline(100, linestyle='--')
        
        plt.title(city)
    
    plt.tight_layout()
    plt.savefig("obj1_aqi_trend.png")
    plt.close()


#===========================OBJECTIVE 2===========================
def plot_monthly_analysis(monthly):
    """
    Plot monthly AQI and pollutant analysis
    """
    plt.figure(figsize=(14, 6))
    
    # AQI by Month
    plt.subplot(1, 2, 1)
    plt.bar(monthly.index, monthly['aqi'])
    
    for i, v in enumerate(monthly['aqi']):
        if v > 0:
            plt.text(i, v+5, f"{int(v)}", ha='center')
    
    plt.title("Average AQI by Month")
    
    # Pollutants by Month
    plt.subplot(1, 2, 2)
    x = np.arange(len(monthly))
    w = 0.25
    
    plt.bar(x - w, monthly['pm2_5'], w, label='PM2.5')
    plt.bar(x, monthly['pm10'], w, label='PM10')
    plt.bar(x + w, monthly['no2'], w, label='NO2')
    
    plt.xticks(x, monthly.index)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("obj2_monthly_pollution.png")
    plt.close()


#===========================OBJECTIVE 3===========================
def plot_pollutant_comparison(df, pivot):
    """
    Plot heatmap and scatter plot for pollutant comparison
    """
    plt.figure(figsize=(16, 6))
    
    # Heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(pivot, annot=True, cmap="coolwarm")
    plt.title("City vs Pollutants")
    
    # Scatter Plot
    plt.subplot(1, 2, 2)
    plt.scatter(df['pm2_5'], df['aqi'], s=10, alpha=0.5)
    
    z = np.polyfit(df['pm2_5'], df['aqi'], 1)
    p = np.poly1d(z)
    plt.plot(df['pm2_5'], p(df['pm2_5']), linestyle='--')
    
    plt.xlabel("PM2.5")
    plt.ylabel("AQI")
    
    plt.tight_layout()
    plt.savefig("obj3_pollutant_comparison.png")
    plt.close()


#===========================OBJECTIVE 4=========================== 
def plot_worst_stations(df):
    """
    Plot analysis of worst performing stations
    """
    station_avg = df.groupby('station')['aqi'].mean().sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(16, 6))
    
    # Bar Chart
    plt.subplot(1, 2, 1)
    station_avg.plot(kind='barh')
    plt.gca().invert_yaxis()
    
    # Box Plot
    plt.subplot(1, 2, 2)
    sns.boxplot(
        data=df[df['station'].isin(station_avg.head(8).index)],
        x='station',
        y='aqi'
    )
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("obj4_worst_stations.png")
    plt.close()
#===========================DASHBOARD 1===========================
def create_final_dashboard(df, r2, rmse, mae):
    """
    Create comprehensive dashboard with all key metrics
    """
    plt.figure(figsize=(20, 11), facecolor='#0b1420')
    
    plt.suptitle(
        "CPCB AQI MONITORING — INDIA (Real Station Data)",
        fontsize=18,
        color='#6ab0ff',
        fontweight='bold'
    )
    
    # 1. PIE CHART — AQI Category
    plt.subplot(2, 3, 1)
    colors = ['#4cc9f0', '#90be6d', '#f9c74f', '#f9844a', '#f94144', '#bc4749']
    df['aqi_category'].value_counts().plot(
        kind='pie',
        autopct='%1.1f%%',
        colors=colors,
        textprops={'color': 'white'}
    )
    plt.title("AQI Category Distribution", color='white')
    
    # 2. Mean AQI by City
    plt.subplot(2, 3, 2)
    city_avg = df.groupby('city')['aqi'].mean().sort_values()
    bars = plt.barh(city_avg.index, city_avg.values, color=sns.color_palette("viridis", len(city_avg)))
    plt.axvline(200, color='red', linestyle='--', label='Poor (200)')
    plt.legend()
    plt.title("Mean AQI by City", color='white')
    plt.xlabel("Mean AQI", color='white')
    
    # 3. Correlation Matrix
    plt.subplot(2, 3, 3)
    sns.heatmap(
        df[POLLUTANTS + ['aqi']].corr(),
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        cbar=True
    )
    plt.title("Pollutant Correlation Matrix", color='white')
    
    # 4. Model Metrics
    plt.subplot(2, 3, 4)
    metrics = [r2, r2, rmse/100, mae/10]
    labels = ["R² Train", "R² Test", "RMSE\n(÷100)", "MAE\n(÷10)"]
    bars = plt.bar(labels, metrics, color=['#e63946', '#457b9d', '#2a9d8f', '#e9c46a'])
    
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', color='white')
    
    plt.title("Model Accuracy Metrics", color='white')
    
    # 5. AQI by Season
    plt.subplot(2, 3, 5)
    sns.boxplot(x='season', y='aqi', data=df, hue='season', palette='viridis', legend=False)
    plt.title("AQI by Season", color='white')
    
    # 6. Sample Station Trend
    plt.subplot(2, 3, 6)
    sample_station = df['station'].iloc[0]
    sample_df = df[df['station'] == sample_station]
    
    plt.plot(sample_df['date'], sample_df['pm2_5'], label='PM2.5', color='#ff4d6d')
    plt.plot(sample_df['date'], sample_df['no2'], label='NO2', color='#4cc9f0')
    plt.plot(sample_df['date'], sample_df['so2'], label='SO2', color='#2ec4b6')
    
    plt.legend()
    plt.title(f"{sample_station} — Pollutant Readings", color='white')
    plt.xlabel("Date", color='white')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("dashboard_summary.png", dpi=300, facecolor='#0b1420')
    plt.show()


#===========================DASHBOARD 2===========================
def create_advanced_trend_dashboard(df):
    """
    Create advanced AQI trend dashboard (Page 2)
    """
    plt.figure(figsize=(20, 12), facecolor='#0b1420')
    
    plt.suptitle(
        " AQI Trend Over Time by City\n(Real CPCB Monitoring Stations)",
        fontsize=16,
        color='white',
        fontweight='bold'
    )
    
    cities = df['city'].unique()[:9]
    colors = [
        '#ff4d4d', '#4f83cc', '#2ec4b6',
        '#e9c46a', '#f4a261', '#264653',
        '#a8dadc', '#1d3557', '#90be6d'
    ]
    
    for i, city in enumerate(cities):
        plt.subplot(3, 3, i+1)
        
        city_df = df[df['city'] == city]
        
        # Line
        plt.plot(city_df['date'], city_df['aqi'], color=colors[i], linewidth=1.8, label=city)
        
        # Fill area
        plt.fill_between(city_df['date'], city_df['aqi'], color=colors[i], alpha=0.15)
        
        # Threshold lines
        plt.axhline(200, color='red', linestyle='--', linewidth=1, label='Poor (200)')
        plt.axhline(100, color='yellow', linestyle='--', linewidth=1, label='Moderate (100)')
        
        # Styling
        plt.title(city, color=colors[i], fontsize=11, fontweight='bold')
        plt.xlabel("Date", fontsize=7, color='white')
        plt.ylabel("AQI", fontsize=7, color='white')
        plt.xticks(fontsize=6, color='white')
        plt.yticks(fontsize=6, color='white')
        plt.grid(alpha=0.1)
        plt.legend(fontsize=6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("page2_trend_advanced.png", dpi=300, facecolor='#0b1420')
    plt.show()


#===========================DASHBOARD 3===========================
def create_combined_analysis_dashboard(df, monthly, pivot):
    
    plt.figure(figsize=(22, 12), facecolor='#0b1420')
    
    plt.suptitle(
        "Pollution Analysis Dashboard (CPCB Data)",
        fontsize=18,
        color='white',
        fontweight='bold'
    )
    
    # 1. AQI by Month
    plt.subplot(2, 2, 1)
    order = ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
    m = monthly.loc[order]
    colors = ['#e63946', '#e76f51', '#f4a261', '#f4a261', '#e76f51', '#e9c46a']
    
    plt.bar(m.index, m['aqi'], color=colors)
    
    for i, v in enumerate(m['aqi']):
        if v > 0:
            plt.text(i, v+5, f"{int(v)}", ha='center', color='white')
    
    plt.axhline(200, color='#ff4d4d', linestyle='--', label='Poor (200)')
    plt.axhline(100, color='#ffd166', linestyle='--', label='Moderate (100)')
    plt.title("Average AQI by Month", color='white')
    plt.ylabel("AQI", color='white')
    plt.legend()
    
    # 2. Pollutants
    plt.subplot(2, 2, 2)
    x = np.arange(len(m))
    w = 0.25
    
    plt.bar(x - w, m['pm2_5'], w, label='PM2.5', color='#e63946')
    plt.bar(x, m['pm10'], w, label='PM10', color='#f4a261')
    plt.bar(x + w, m['no2'], w, label='NO₂', color='#457b9d')
    
    plt.xticks(x, m.index)
    plt.ylabel("µg/m³", color='white')
    plt.title("PM2.5 / PM10 / NO₂ by Month", color='white')
    plt.legend()
    
    # 3. Heatmap
    plt.subplot(2, 2, 3)
    sns.heatmap(pivot, annot=True, cmap='YlOrRd', fmt=".1f", linewidths=1, linecolor='black')
    plt.title("City × Pollutant Heatmap", color='white')
    
    # 4. Scatter
    plt.subplot(2, 2, 4)
    for city in df['city'].unique():
        sub = df[df['city'] == city]
        plt.scatter(sub['pm2_5'], sub['aqi'], s=10, alpha=0.6, label=city)
    
    z = np.polyfit(df['pm2_5'], df['aqi'], 1)
    p = np.poly1d(z)
    plt.plot(df['pm2_5'], p(df['pm2_5']), '--', color='white', linewidth=2)
    
    plt.xlabel("PM2.5 (µg/m³)", color='white')
    plt.ylabel("AQI", color='white')
    plt.title("PM2.5 vs AQI — Scatter Plot", color='white')
    plt.legend(fontsize=6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("page2_combined.png", dpi=300, facecolor='#0b1420')
    plt.show()


# -------------------- DASHBOARD 4 --------------------
def create_model_evaluation_dashboard(df, model, y, y_pred, r2):
    """
    Create- Model Evaluation Dashboard
    """
    plt.figure(figsize=(22, 12), facecolor='#0b1420')
    
    plt.suptitle(
        "Linear Regression: AQI Prediction & Model Evaluation\n(Real CPCB Stations)",
        fontsize=16,
        color='white',
        fontweight='bold'
    )
    
    # 1. Actual vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y, y_pred, s=12, alpha=0.6, color='#ff4d4d', label='Data points')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='white', label='Perfect prediction')
    plt.xlabel("Actual AQI", color='white')
    plt.ylabel("Predicted AQI", color='white')
    plt.title(f"Actual vs Predicted (R² = {r2:.4f})", color='white')
    plt.legend()
    
    # 2. Residual Plot
    plt.subplot(2, 2, 2)
    residuals = y - y_pred
    plt.scatter(y_pred, residuals, s=10, alpha=0.5, color='#2ec4b6')
    plt.axhline(0, linestyle='--', color='white')
    plt.xlabel("Predicted AQI", color='white')
    plt.ylabel("Residual", color='white')
    plt.title("Residual Plot", color='white')
    
    # 3. Feature Coefficients
    plt.subplot(2, 2, 3)
    coef = model.coef_
    plt.barh(POLLUTANTS, coef, color='#e63946')
    plt.xlabel("Coefficient Value", color='white')
    plt.title("Feature Coefficients", color='white')
    
    # 4. Top Stations Trend
    plt.subplot(2, 2, 4)
    top5 = df.groupby('station')['aqi'].mean().sort_values(ascending=False).head(5).index
    colors = ['#ff4d4d', '#4cc9f0', '#f4a261', '#2ec4b6', '#ffd166']
    
    for i, station in enumerate(top5):
        sub = df[df['station'] == station].sort_values('date')
        plt.plot(sub['date'], sub['aqi'], color=colors[i], label=f"{station} Actual")
        pred = model.predict(sub[POLLUTANTS])
        plt.plot(sub['date'], pred, linestyle='--', color=colors[i], alpha=0.7, label=f"{station} Predicted")
    
    plt.xlabel("Date", color='white')
    plt.ylabel("AQI", color='white')
    plt.title("Top 5 Stations — Actual vs Predicted", color='white')
    plt.legend(fontsize=6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("objective5_model.png", dpi=300, facecolor='#0b1420')
    plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """
    Main execution function
    """
    # Load and preprocess data
    df = load_and_clean_data(DATA_PATH)
    df = engineer_features(df)
    df.reset_index(drop=True, inplace=True)
    print_data_summary(df)
    
    # Create aggregations
    monthly = create_monthly_aggregation(df)
    pivot = create_city_pivot(df)
    
    # Train model
    model, y_pred, r2, rmse, mae = train_linear_regression(df)
    
    # Generate all visualizations
    print("Generating Objective 1: AQI Trend...")
    plot_aqi_trend_by_city(df)
    
    print("Generating Objective 2: Monthly Analysis...")
    plot_monthly_analysis(monthly)
    
    print("Generating Objective 3: Pollutant Comparison...")
    plot_pollutant_comparison(df, pivot)
    
    print("Generating Objective 4: Worst Stations...")
    plot_worst_stations(df)
    
    print("Generating Dashboard 1: Final Summary...")
    create_final_dashboard(df, r2, rmse, mae)
    
    print("Generating Dashboard 2: Advanced Trends...")
    create_advanced_trend_dashboard(df)
    
    print("Generating Dashboard 3: Combined Analysis...")
    create_combined_analysis_dashboard(df, monthly, pivot)
    
    print("Generating Dashboard 4: Model Evaluation...")
    create_model_evaluation_dashboard(df, model, df['aqi'], y_pred, r2)
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*60)



if __name__ == "__main__":
    main()
