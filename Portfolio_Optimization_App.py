import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import show

# 📌 Scipy & Sklearn
from scipy.stats import norm
import scipy.stats as stats
from sklearn.model_selection import train_test_split

# 📌 Skfolio - Optimisation & Modélisation de Portefeuille
import skfolio as sk
from skfolio import Population, RiskMeasure, PerfMeasure, RatioMeasure
from skfolio.preprocessing import prices_to_returns
from skfolio.datasets import load_sp500_dataset
from skfolio.moments import ShrunkCovariance
from skfolio.prior import EmpiricalPrior
from skfolio.optimization import (
    MaximumDiversification, 
    EqualWeighted, 
    Random, 
    InverseVolatility, 
    MeanRisk, 
    RiskBudgeting, 
    ObjectiveFunction
)


# ----------------------------------------
# Configuration de la page Streamlit
# ----------------------------------------
st.set_page_config(layout="wide")

# ------------------------------------------------------------------
# 📌 Initialisation des bases de données en session_state
# ------------------------------------------------------------------
with st.sidebar:
    # Section Contact dans un bloc coloré
    st.markdown("""
    <div style="background-color: #f0f4f7; padding: 6px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
        <h3 style="color: #000000; text-align: center; margin: 0 0 0px 0;">Contact Me</h3>
        <p style="color: #000000; margin: 0px; line-height: 1.5;"><strong>Raphael EL-BAZE</strong></p>
        <p style="color: #000000; margin: 0px; line-height: 1.5;">📧 <a href="mailto:raphael.elbaze.pro@gmail.com" style="color: #0072b1;">raphael.elbaze.pro@gmail.com</a></p>
        <p style="color: #000000; margin: 0px; line-height: 1.5;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="width: 16px; vertical-align: middle;"> 
            <a href="https://www.linkedin.com/in/raphael-el-baze/" target="_blank" style="color: #0072b1;">https://www.linkedin.com/in/raphael-el-baze/</a>
        </p>
        <p style="color: #000000; margin: 0px; line-height: 1.8;">📞 +33 (0)6 68 86 91 30</p>
    </div>
    """, unsafe_allow_html=True)

    # Trait de séparation
    st.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.header("Portfolio Inputs")

# Sélection des tickers avec entrée manuelle
def get_sp500_tickers():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    sp500_tickers = pd.read_csv(url)["Symbol"].tolist()
    return sp500_tickers

all_tickers = get_sp500_tickers()

# Input
selected_tickers = st.sidebar.multiselect(
    "Choose Stock Tickers", 
    options=all_tickers, 
    default=["AAPL", "XOM", "JNJ", "PG", "GS", "NKE", "TSLA", "UNH", "AMZN", "BA"], 
    placeholder="Type or select tickers...")

# Choix des benchmarks
benchmark_ticker = st.sidebar.selectbox("Choose a Benchmark", ["^GSPC", "^DJI", "^IXIC", "^RUT", "^FTSE", "^N225"], index=0)

# Tickers + Benchmak Tickers
tickers_and_benchmarks = selected_tickers + [benchmark_ticker]

# Choix dates
start_date = st.sidebar.date_input("Start Date", dt.date(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.date.today())

# Téléchargement des données juste pour tickers
#@st.cache_data(ttl=3600)  # Cache permanent jusqu'à ce que les inputs changent
def get_market_data(tickers, start, end):
    data = yf.download(tickers=tickers, start=start, end=end)["Close"].dropna()
    return data

# Téléchargement des données juste pour selected_tickers
data_prices = get_market_data(selected_tickers, start_date, end_date)
data_returns = prices_to_returns(data_prices)

# Téléchargement des données pour tickers_and_benchmarks
data_tickers_and_benchmarks_prices = get_market_data(tickers_and_benchmarks, start_date, end_date)
data_tickers_and_benchmarks_returns = prices_to_returns(data_tickers_and_benchmarks_prices)

# Création des Tabs
main_tab = st.tabs(["Asset Analysis", 
                    "Portfolio Comparison", 
                    "Mean-Risk",
                    "Risk Budgeting"
                    ])

# ----------------------------------------
# 📌 Quick Guide sous les inputs
# ----------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠 Quick Guide")
st.sidebar.info("""
1️⃣ **Asset Analysis** 📊 *(Exploratory Data)*:
   - General Info (sector, market cap, etc.)
   - Correlation Matrix
   - Sector Allocation
   - Beta vs Benchmark
   - Returns Distribution & Stats

2️⃣ **Portfolio Comparison** 📈 → Backtest multiple portfolio strategies (**Train/Test split**).

3️⃣ **Mean-Risk Optimization** 🎯:
   - **Max Sharpe Ratio** *(maximize risk-adjusted return)*
   - **Min CVaR** *(reduce tail risk)*
   - **Efficient Frontier** *(optimal risk-return trade-off)*

4️⃣ **Risk Budgeting** ⚖️:
   - **Risk Parity - Variance** *(equal risk contribution)*
   - **Risk Budgeting - CVaR** *(tail risk allocation)*
   - **Risk Parity - Covariance Shrinkage** *(adjust covariance matrix)*

🚨 **IMPORTANT** 🚨  
🔹 Every time you **change an input (tickers, dates, parameters, etc.)**, you **MUST re-run the models** to refresh results.  
🔹 **Tabs are independent** → Run each separately.  
""")

#-----------------------------------------------------------------------------------------
# Tab 0: ASSET ANALYSIS
#-----------------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_ticker_info(tickers_and_benchmarks):
    """Récupère les informations fondamentales des tickers depuis Yahoo Finance et les met en cache."""
    #st.write("Fetching data from Yahoo Finance...")  # Vérifie si cette ligne s'affiche à chaque exécution
    info_dict = {}
    for ticker in tickers_and_benchmarks:
        try:
            info = yf.Ticker(ticker).info
            info_dict[ticker] = {
                "Sector": info.get("sector", "N/A"),
                "Industry": info.get("industry", "N/A"),
                "Market Cap": info.get("marketCap", "N/A"),
                "Country": info.get("country", "N/A")
            }
        except:
            continue
    return info_dict

with main_tab[0]:  
    st.markdown("""
        <h1 style='text-align: center; color: cyan;'>Asset Analysis</h1>
        <p style='text-align: center; font-size:18px; color: yellow;'>💡 Gain insights into asset performance, risk metrics, sector allocation, correlations, and their relationship with benchmarks.</p>
        """, unsafe_allow_html=True)

    # Sélecteur pour choisir les analyses à afficher
    selected_outputs = st.multiselect(
        "Select analyses to display:",
        ["Asset Information", "Correlation Matrix", "Sector Allocation", "Beta vs Benchmark", "Distribution of Returns"],
        default=["Asset Information", "Correlation Matrix", "Sector Allocation", "Beta vs Benchmark", "Distribution of Returns"]  # Affichage par défaut
    )
    
    st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

    info_dict = get_ticker_info(tickers_and_benchmarks)
    #st.write("Info fetched at:", dt.datetime.now())
    asset_info_df = pd.DataFrame.from_dict(info_dict, orient='index').T

    # 📌 1. Asset Information
    if "Asset Information" in selected_outputs:
        st.markdown("<h3 style='color: #2ECC71;'>Asset Information</h3>", unsafe_allow_html=True)
        st.dataframe(asset_info_df)
        #st.write("✅ Tickers successfully loaded:", data_prices.columns.tolist())

        st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

    # 📌 2. Matrice de Corrélation
    if "Correlation Matrix" in selected_outputs:
        st.markdown("<h3 style='color: #2ECC71;'>Correlation Matrix</h3>", unsafe_allow_html=True)
        corr_matrix = data_tickers_and_benchmarks_returns.corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        fig.update_layout(title="Correlation Matrix", xaxis_title="Ticker", yaxis_title="Ticker")
        st.plotly_chart(fig)

        st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

    # 📌 3. Répartition sectorielle
    if "Sector Allocation" in selected_outputs:
        st.markdown("<h3 style='color: #2ECC71;'>Sector Allocation</h3>", unsafe_allow_html=True)
        filtered_asset_info_df = asset_info_df.drop(columns=[benchmark_ticker], errors='ignore')
        sector_counts = filtered_asset_info_df.loc["Sector"].value_counts()
        fig = px.pie(names=sector_counts.index, values=sector_counts.values, title="Sector Allocation")
        st.plotly_chart(fig)

        st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

    # 📌 4. Beta vs Benchmark
    if "Beta vs Benchmark" in selected_outputs:
        st.markdown("<h3 style='color: #2ECC71;'>Beta vs Benchmark</h3>", unsafe_allow_html=True)
        beta_dict = {}
        cov_matrix = data_tickers_and_benchmarks_returns.cov()
        benchmark_var = data_tickers_and_benchmarks_returns[benchmark_ticker].var()
        for ticker in tickers_and_benchmarks:
            if ticker != benchmark_ticker:
                beta_dict[ticker] = cov_matrix.loc[ticker, benchmark_ticker] / benchmark_var
        beta_df = pd.DataFrame.from_dict(beta_dict, orient='index', columns=["Beta"]).T
        st.dataframe(beta_df)

        st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

    # 📌 5. Distribution des rendements
    if "Distribution of Returns" in selected_outputs:
        st.markdown("<h3 style='color: #2ECC71;'>Distribution of Returns</h3>", unsafe_allow_html=True)
        
        # Sélecteur dynamique des tickers
        selected_tickers_for_plot = st.multiselect(
            "Select Tickers to Display", 
            tickers_and_benchmarks, 
            default=[tickers_and_benchmarks[0]]  # Affichage du premier ticker par défaut
        )

        if selected_tickers_for_plot:
            fig = go.Figure()
            stat_metrics = {}
            dash_style = 'dash'  # Garder un style pointillé pour toutes les lois normales
            color_palette = px.colors.qualitative.Set1  # Palette de couleurs différentes

            for i, ticker in enumerate(selected_tickers_for_plot):
                # Récupération des rendements
                returns = data_tickers_and_benchmarks_returns[ticker]

                # Calcul des paramètres de la loi normale ajustée
                mu, sigma = returns.mean(), returns.std()
                x_vals = np.linspace(returns.min(), returns.max(), 100)
                normal_vals = norm.pdf(x_vals, mu, sigma)

                # Ajout de la densité des rendements (remplie)
                fig.add_trace(go.Histogram(
                    x=returns, 
                    histnorm='probability density', 
                    name=f"{ticker} Returns",
                    opacity=0.6
                ))

                # Ajout de la courbe de la loi normale ajustée avec une couleur différente
                fig.add_trace(go.Scatter(
                    x=x_vals, 
                    y=normal_vals, 
                    mode='lines', 
                    name=f"{ticker} Normal Fit",
                    line=dict(color=color_palette[i % len(color_palette)], width=2, dash=dash_style)
                ))

                # Stockage des statistiques
                stat_metrics[ticker] = {
                    "Mean": mu,
                    "Standard Deviation": sigma,
                    "Skewness": returns.skew(),
                    "Kurtosis": returns.kurtosis()
                }

            fig.update_layout(
                title="Distribution of Returns vs Normal Distribution",
                xaxis_title="Returns",
                yaxis_title="Density",
                barmode="overlay"
            )

            st.plotly_chart(fig, use_container_width=True)

            # 📌 Ajout des statistiques (Skewness, Kurtosis)
            st.markdown("<h3 style='color: #2ECC71;'>Statistical Properties of Returns</h3>", unsafe_allow_html=True)
            stat_df = pd.DataFrame.from_dict(stat_metrics, orient="index")
            st.dataframe(stat_df)

        st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)


            # ----------------------------------
            # #  METRICS TO VERIFY CALCULUS
            # ----------------------------------
            # # Récupérer les rendements des portefeuilles dans le training set
            # train_returns = {model_name: ptf_train_results_dict[model_name].returns_df for model_name in selected_models}
            # # Récupérer les rendements des portefeuilles dans le test set
            # test_returns = {model_name: ptf_test_results_dict[model_name].returns_df for model_name in selected_models}
            # # Afficher les rendements dans Streamlit pour vérification
            # st.subheader("Train Set Portfolio Returns")
            # for model, df in train_returns.items():
            #     st.write(f"**{model}**")
            #     st.dataframe(df) 
            # st.subheader("Test Set Portfolio Returns")
            # for model, df in test_returns.items():
            #     st.write(f"**{model}**")
            #     st.dataframe(df)  
            # st.subheader("Portfolio Cumulative Returns (Train Set)")
            # st.write(pop_train_cum_returns.data[0]["y"])

            # st.subheader("Portfolio Cumulative Returns (Test Set)")
            # st.write(pop_test_cum_returns.data[0]["y"])

            # # ----------------------------------
            # # #  EXPLORING PTF METHODS
            # # ----------------------------------
            # # Model
            # portfolio = ptf_train_results_dict["Equal Weighted"]
            # # Liste des attributs et méthodes disponibles
            # st.write(dir(portfolio))



# --------------------------------------------------------------------------------
# 📌 TAB 1: PORTFOLIO COMPARISON
# --------------------------------------------------------------------------------
with main_tab[1]:
    # Titre principal avec un style plus impactant
    st.markdown("""
        <h1 style='text-align: center; color: cyan;'>Portfolio Comparison</h1>
        <p style='text-align: center; font-size:18px; color: yellow;'>💡 Compare multiple portfolio optimization models on a Train/Test split.</p>
        """, unsafe_allow_html=True)

    # ---------------------------
    # 📌 SELECTION DES MODELES
    # ---------------------------
    model_options = [
        "Equal Weighted", "Inverse Volatility", "Random", "Maximum Diversification",
        "Mean-Risk - Maximum Sharpe Ratio", "Mean-Risk - Minimum CVaR",
        "Risk Parity - Variance", "Risk Budgeting - CVaR", "Risk Parity - Covariance Shrinkage"
    ]
    
    selected_models = st.multiselect("**📌 Select Models to Compare**", model_options, default=["Equal Weighted", "Inverse Volatility", "Random", "Maximum Diversification"])
    
    if "selected_models" not in st.session_state:
        st.session_state["selected_models"] = selected_models
    else:
        if st.session_state["selected_models"] != selected_models:
            st.session_state["selected_models"] = selected_models
            st.warning("⚠️ You have changed the input parameters. Please **rerun the portfolio comparison** to update the results.")

    if "model_parameters" not in st.session_state:
        st.session_state["model_parameters"] = {}

    # ---------------------------
    # 📌 PARAMÈTRES GÉNÉRAUX
    # ---------------------------
    st.markdown("""
        <h4 style='color: cyan;'>Global Parameters</h4>
        """, unsafe_allow_html=True)

    # Affichage compact du slider avec les valeurs à côté
    col1, col2 = st.columns([2, 1])
    with col1:
        test_size = st.slider("Select Test Set Percentage", 0.1, 0.9, 0.33, step=0.01)
    with col2:
        train_size = 1 - test_size
        st.markdown(f"<p style='margin-top:35px; text-align:center; font-size:16px; color: pink;'>Train: {train_size*100:.0f}% | Test: {test_size*100:.0f}%</p>", unsafe_allow_html=True)
    
    if "test_size" not in st.session_state:
        st.session_state["test_size"] = test_size
    else:
        if st.session_state["test_size"] != test_size:
            st.session_state["test_size"] = test_size
            st.warning("⚠️ You have changed the input parameters. Please **rerun the portfolio comparison** to update the results.")

    # Splitting sets
    data_returns_train, data_returns_test = train_test_split(data_returns, test_size=st.session_state["test_size"], shuffle=False)

    # ---------------------------------------------------
    # 📌 STOCKAGE DANS SESSION STATE
    # ---------------------------------------------------
    if "pop_train" not in st.session_state:
        st.session_state["pop_train"] = None
        st.session_state["pop_test"] = None
        #st.session_state["selected_models"] = []

    # ---------------------------------------------------
    # 📌 PARAMÈTRES DES MODELES SPÉCIFIQUES
    # ---------------------------------------------------
    if "Risk Budgeting - CVaR" in selected_models or "Risk Parity - Covariance Shrinkage" in selected_models:
        st.markdown("""
            <h4 style='color: cyan;'>Model-Specific Parameters</h4>
            """, unsafe_allow_html=True)

    # Gestion du modèle "Risk Budgeting - CVaR"
    if "Risk Budgeting - CVaR" in selected_models:
        st.markdown("✅ You have selected Risk Budgeting - CVaR. Please configure the risk budget.")

        # Initialisation du DataFrame pour le Risk Budgeting - CVaR
        risk_budget_df = pd.DataFrame(
            [[1.0] * len(data_returns_train.columns)],  # Valeurs par défaut de 1.0
            columns=data_returns_train.columns,
            index=["Risk Budget"]
        )

        # Affichage du tableau modifiable
        edited_risk_budget = st.data_editor(
            risk_budget_df, 
            use_container_width=True, 
            key="pop_risk_budget_table"
        )

        # Conversion en dictionnaire
        budget_param = edited_risk_budget.loc["Risk Budget"].to_dict()

    # Gestion du modèle "Risk Parity - Covariance Shrinkage"
    shrinkage_valid = True  # Pour activer/désactiver le bouton
    if "Risk Parity - Covariance Shrinkage" in selected_models:
        st.markdown("✅ You have selected Risk Parity - Covariance Shrinkage. Please enter a coefficient (0 to 1).")
        shrinkage_value = st.number_input("Shrinkage Coefficient (λ)", value=0.1, step=0.01)        # Vérification du coefficient
        if shrinkage_value < 0 or shrinkage_value > 1:
            st.error("🚨 Error: The coefficient must be between 0 and 1.")
            shrinkage_valid = False  # Désactive le bouton
        st.session_state["model_parameters"]["Risk Parity - Covariance Shrinkage"] = shrinkage_value
    
    # ---------------------------
    # 📌 TRAIN & TEST MODELS
    # ---------------------------
    # 📌 Style personnalisé pour le bouton Run
    st.markdown("""
        <style>
            div.stButton > button:first-child {
                background-color: rgb(75, 117, 255);
                color: black;
                font-weight: bold;
                padding: 10px;
                border-radius: 8px;
                border: none;
                width: 100%;
                transition: background-color 0.3s ease-in-out;
            }
            div.stButton > button:hover {
                background-color: rgb(150, 219, 0);
                color: black;
            }
            div.stButton > button:disabled {
                background-color: rgb(200, 200, 200);
                color: rgb(100, 100, 100);
                cursor: not-allowed;
            }
        </style>
        """, unsafe_allow_html=True)

    # Bouton d'exécution désactivé si shrinkage non valide
    run_button = st.button("🚀 Run Portfolio Comparison", disabled=not shrinkage_valid)    
    if run_button:
        ptf_train_results_dict = {}
        ptf_test_results_dict = {}

        for model_name in selected_models:
            if model_name == "Equal Weighted":
                model = EqualWeighted(
                    portfolio_params=dict(name=model_name)
                )

            elif model_name == "Inverse Volatility":
                model = InverseVolatility(
                    prior_estimator=EmpiricalPrior(), 
                    portfolio_params=dict(name=model_name)
                )
            
            elif model_name == "Random":
                model = Random(
                    portfolio_params=dict(name=model_name)
                )

            elif model_name == "Maximum Diversification":
                model = MaximumDiversification(
                    portfolio_params=dict(name=model_name)
                )
            
            elif model_name == "Mean-Risk - Maximum Sharpe Ratio":
                model = MeanRisk(
                    risk_measure=RiskMeasure.STANDARD_DEVIATION, 
                    objective_function=ObjectiveFunction.MAXIMIZE_RATIO, 
                    portfolio_params=dict(name=model_name)
                )
            
            elif model_name == "Mean-Risk - Minimum CVaR":
                model = MeanRisk(
                    risk_measure=RiskMeasure.CVAR, 
                    objective_function=ObjectiveFunction.MINIMIZE_RISK, 
                    portfolio_params=dict(name=model_name)
                )
            
            elif model_name == "Risk Parity - Variance":
                model = RiskBudgeting(
                    risk_measure=RiskMeasure.VARIANCE, 
                    portfolio_params=dict(name=model_name)
                )

            elif model_name == "Risk Budgeting - CVaR":
                model = RiskBudgeting(
                    risk_measure=RiskMeasure.CVAR, 
                    risk_budget=budget_param, 
                    portfolio_params=dict(name=f"{model_name} - Custom Budget")
                )
            
            elif model_name == "Risk Parity - Covariance Shrinkage":
                shrink_param = st.session_state["model_parameters"]["Risk Parity - Covariance Shrinkage"]
                model = RiskBudgeting(
                    risk_measure=RiskMeasure.VARIANCE,
                    prior_estimator=EmpiricalPrior(covariance_estimator=ShrunkCovariance(shrinkage=shrink_param)), 
                    portfolio_params=dict(name=f"{model_name} - λ={shrink_param}")
                )

            model.fit(data_returns_train)
            ptf_train_results_dict[model_name] = model.predict(data_returns_train)
            ptf_test_results_dict[model_name] = model.predict(data_returns_test)

        # Stocker les résultats dans session_state
        st.session_state["pop_train"] = Population(list(ptf_train_results_dict.values()))
        st.session_state["pop_test"] = Population(list(ptf_test_results_dict.values()))
        st.session_state["selected_models"] = selected_models
        st.session_state["ptf_train_results_dict"] = ptf_train_results_dict
        st.session_state["ptf_test_results_dict"] = ptf_test_results_dict
        
        # Display process succeeded!
        st.success(f"Your model has been trained and tested!")
        st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # 📌 AFFICHAGE DES RÉSULTATS SI DISPONIBLES
    # ---------------------------------------------------
    if st.session_state["pop_train"] is not None:

        #--------------------------------------
        # Compositions
        #--------------------------------------
        st.markdown("<h3 style='color: #2ECC71;'>Portfolio Composition</h3>", unsafe_allow_html=True)
        st.plotly_chart(st.session_state["pop_train"].plot_composition(), use_container_width=True, key="pop_comparison_composition")
        st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)
        
        #--------------------------------------
        # Plot Cum Returns
        #--------------------------------------
        st.markdown("<h3 style='color: #2ECC71;'>Portfolio Cumulative Returns</h3>", unsafe_allow_html=True)
        
        # Création de l'objet graphique
        fig = go.Figure()

        # Déterminer les limites des zones d’arrière-plan
        train_start_date = data_returns_train.index[0]
        train_end_date = data_returns_train.index[-1]
        test_start_date = data_returns_test.index[0]
        test_end_date = data_returns_test.index[-1]

        # Ajouter une zone d'arrière-plan pour le Train Set (bleu clair)
        fig.add_shape(
            type="rect",
            x0=train_start_date, x1=train_end_date,
            y0=0, y1=1,  # Étendre verticalement
            xref="x", yref="paper",
            fillcolor="rgba(0, 128, 255, 0.08)",  # Bleu très transparent
            layer="below",
            line_width=0,
        )

        # Ajouter une zone d'arrière-plan pour le Test Set (orange transparent)
        fig.add_shape(
            type="rect",
            x0=test_start_date, x1=test_end_date,
            y0=0, y1=1,
            xref="x", yref="paper",
            fillcolor="rgba(255, 165, 0, 0.12)",  # Orange légèrement plus transparent
            layer="below",
            line_width=0,
        )

        # Ajouter une annotation pour "Train Set"
        fig.add_annotation(
            x=train_start_date + (train_end_date - train_start_date) / 2,  
            y=1.05,  
            text="Train Set",  
            showarrow=False,
            font=dict(color="rgba(0, 128, 255, 0.7)", size=14, family="Arial"),
            bgcolor="rgba(0, 128, 255, 0.15)",
            align="center",
            xref="x",
            yref="paper",
        )

        # Ajouter une annotation pour "Test Set"
        fig.add_annotation(
            x=test_start_date + (test_end_date - test_start_date) / 2,  
            y=1.05,  
            text="Test Set",  
            showarrow=False,
            font=dict(color="rgba(255, 165, 0, 0.7)", size=14, family="Arial"),
            bgcolor="rgba(255, 165, 0, 0.15)",
            align="center",
            xref="x",
            yref="paper",
        )

        # Boucle sur tous les modèles sélectionnés pour afficher leurs rendements cumulés
        for i, model_name in enumerate(st.session_state["selected_models"]):

            # Vérifier que le modèle a bien été calculé dans train et test
            if (
                "ptf_train_results_dict" in st.session_state
                and "ptf_test_results_dict" in st.session_state
                and model_name in st.session_state["ptf_train_results_dict"]
                and model_name in st.session_state["ptf_test_results_dict"]
            ):

                # Récupération des rendements cumulés pour le modèle i
                pop_train_cum_returns = st.session_state["pop_train"].plot_cumulative_returns().data[i]  
                pop_test_cum_returns = st.session_state["pop_test"].plot_cumulative_returns().data[i]
                portfolio_test = st.session_state["pop_test"][i]
                portfolio_test_returns_df = portfolio_test.returns_df.squeeze()
                
                # Récupérer la dernière valeur du Train Set pour aligner le Test Set
                last_train_value = pop_train_cum_returns["y"][-1]

                # Initialisation du premier élément avec l'addition du dernier return train + premier return test
                test_returns_shifted = [last_train_value + portfolio_test_returns_df.iloc[0]]

                # Ajout des rendements suivants en additionnant à chaque fois au précédent
                for y in portfolio_test_returns_df.iloc[1:]:
                    test_returns_shifted.append(test_returns_shifted[-1] + y)

                # Définir une couleur unique par modèle
                model_color = px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]

                # Ajout du Train Set (ligne fine)
                fig.add_trace(go.Scatter(
                    x=pop_train_cum_returns["x"], 
                    y=[y * 100 for y in pop_train_cum_returns["y"]],  # Transformation en % pour affichage uniquement
                    mode='lines', 
                    name=f"Train Set - {model_name}",
                    line=dict(color=model_color, width=2),  # Ligne plus fine pour train
                    hovertemplate="<b>Portfolio: " + f"{model_name} - Train Set" + "</b><br>%{x}<br>Return: %{y:.2f}%<extra></extra>"
                ))

                # Ajout du Test Set (ligne plus épaisse pour le différencier)
                fig.add_trace(go.Scatter(
                    x=pop_test_cum_returns["x"], 
                    y=[y * 100 for y in test_returns_shifted],  # Transformation en % pour affichage uniquement,  
                    mode='lines', 
                    name=f"Test Set - {model_name}",
                    line=dict(color=model_color, width=3.5),  # Ligne plus épaisse
                    hovertemplate="<b>Portfolio: " + f"{model_name} - Test Set" + "</b><br>%{x}<br>Return: %{y:.2f}%<extra></extra>"
                ))

        # Ajouter une ligne verticale pour la date de split
        fig.add_vline(x=test_start_date, line_dash="dot", line_color="purple")

        # Ajouter une annotation pour la Split Date au-dessus de la ligne verticale
        fig.add_annotation(
            x=test_start_date, 
            y=1.1,  # Position en haut du graphe
            text=f"Split Date: {test_start_date.date()}", 
            showarrow=False,
            font=dict(color="white", size=12, family="Arial"),
            bgcolor="purple",
            opacity=0.8,
            xref="x",
            yref="paper",  # Se base sur l'axe général et non les valeurs spécifiques
            align="center"
        )

        # Mise en page finale
        fig.update_layout(
            title="Cumulative Returns (non-compounded) - Train & Test - All Portfolios",
            xaxis_title="Observations",
            yaxis_title="Cumulative Returns",  # Ajout de "%" à l'axe Y
            yaxis_tickformat=".2f",  # Arrondi à 2 décimales
            legend_title="Portfolios",
            template="plotly_dark"
        )

        # Affichage dans Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Modifier le titre du graphique pour le Train Set
        fig_pop_train_cum_returns = st.session_state["pop_train"].plot_cumulative_returns()
        fig_pop_train_cum_returns.update_layout(title="Cumulative Returns (non-compounded) - Train Set - All Portfolios")
        st.plotly_chart(fig_pop_train_cum_returns, use_container_width=True, key="pop_train_cum_returns")
        
        # Modifier le titre du graphique pour le Test Set
        fig_pop_test_cum_returns = st.session_state["pop_test"].plot_cumulative_returns()
        fig_pop_test_cum_returns.update_layout(title="Cumulative Returns (non-compounded) - Test Set - All Portfolios")
        st.plotly_chart(fig_pop_test_cum_returns, use_container_width=True, key="pop_test_cum_returns")

        st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

        #--------------------------------------
        # Summariess
        #--------------------------------------
        st.markdown("<h3 style='color: #2ECC71;'>Portfolio Summary</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h6 style='text-align: center;'>Train Set</h6>", unsafe_allow_html=True)
            st.write(st.session_state["pop_train"].summary())
        with col2:
            st.markdown("<h6 style='text-align: center;'>Test Set</h6>", unsafe_allow_html=True)
            st.write(st.session_state["pop_test"].summary())
        st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)
        


# --------------------------------------------------------
# 📌 TAB 2: MEAN-RISK
# --------------------------------------------------------
with main_tab[2]:
    # Création des sous-tabs pour la comparaison et les modèles spécifiques
    mean_risk_tabs = st.tabs(["Maximum Sharpe Ratio", "Minimum CVaR", "Efficient Frontier" ])

    # --------------------------------------------------------
    # 📌 Maximum Sharpe Ratio
    # --------------------------------------------------------
    with mean_risk_tabs[0]:
        st.markdown(
            "<h2 style='text-align: center; color: #4B9CD3;'>Mean-Risk - Maximum Sharpe Ratio</h2>",
            unsafe_allow_html=True
        )
        
        #--------------------------------------
        # Train & Test
        #--------------------------------------
        # Appliquer un style CSS au bouton Streamlit
        st.markdown("""
            <style>
                div.stButton > button:first-child {
                    background-color: rgb(75, 117, 255);
                    color: black;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 8px;
                    border: none;
                    width: 100%;
                }
                div.stButton > button:hover {
                    background-color: rgb(150, 219, 0);
                    color: black;
                }
            </style>
         """, unsafe_allow_html=True)
        
        # Initialisation des modèles en session_state s'ils n'existent pas
        if "max_sharpe_pop_train" not in st.session_state:
            st.session_state["max_sharpe_pop_train"] = None
            st.session_state["max_sharpe_pop_test"] = None
        
        # Bouton pour entraîner uniquement ce modèle
        if st.button("Train & Test - Max Sharpe"):
            model_max_sharpe = MeanRisk(
                risk_measure=RiskMeasure.STANDARD_DEVIATION,
                objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
                portfolio_params=dict(name="Max Sharpe"),
            )

            # Entraînement
            model_max_sharpe.fit(data_returns_train)
            max_sharpe_portfolio_train = model_max_sharpe.predict(data_returns_train)
            max_sharpe_portfolio_test = model_max_sharpe.predict(data_returns_test)

            # Stocker les résultats dans session_state
            st.session_state["max_sharpe_pop_train"] = Population([max_sharpe_portfolio_train])
            st.session_state["max_sharpe_pop_test"] = Population([max_sharpe_portfolio_test])
            
            # Display process succeeded!
            st.success("Your model has been trained and tested!")

        #--------------------------------------------------------------------------
        # Affichage des résultats uniquement si un modèle a été entraîné
        #--------------------------------------------------------------------------
        if st.session_state["max_sharpe_pop_train"] is not None:
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)
            
            #--------------------------------------
            # Compositions
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Composition</h3>", unsafe_allow_html=True)
            st.plotly_chart(st.session_state["max_sharpe_pop_train"].plot_composition(), use_container_width=True, key="max_sharpe_composition")
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

            #--------------------------------------
            # Plot Cum Returns
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Cumulative Returns</h3>", unsafe_allow_html=True)

            # GRAPHIQUE COMBINANT TRAIN ET TEST SET
            fig = go.Figure()

            # Déterminer les limites des zones d’arrière-plan
            train_start_date = data_returns_train.index[0]
            train_end_date = data_returns_train.index[-1]
            test_start_date = data_returns_test.index[0]
            test_end_date = data_returns_test.index[-1]

            # Ajouter une zone d'arrière-plan pour le Train Set (bleu clair)
            fig.add_shape(
                type="rect",
                x0=train_start_date, x1=train_end_date,
                y0=0, y1=1,  # Étendre verticalement
                xref="x", yref="paper",
                fillcolor="rgba(0, 128, 255, 0.08)",  # Bleu très transparent
                layer="below",
                line_width=0,
            )

            # Ajouter une zone d'arrière-plan pour le Test Set (orange transparent)
            fig.add_shape(
                type="rect",
                x0=test_start_date, x1=test_end_date,
                y0=0, y1=1,
                xref="x", yref="paper",
                fillcolor="rgba(255, 165, 0, 0.12)",  # Orange légèrement plus transparent
                layer="below",
                line_width=0,
            )

            # Ajouter une annotation pour "Train Set"
            fig.add_annotation(
                x=train_start_date + (train_end_date - train_start_date) / 2,  
                y=1.05,  
                text="Train Set",  
                showarrow=False,
                font=dict(color="rgba(0, 128, 255, 0.7)", size=14, family="Arial"),
                bgcolor="rgba(0, 128, 255, 0.15)",
                align="center",
                xref="x",
                yref="paper",
            )

            # Ajouter une annotation pour "Test Set"
            fig.add_annotation(
                x=test_start_date + (test_end_date - test_start_date) / 2,  
                y=1.05,  
                text="Test Set",  
                showarrow=False,
                font=dict(color="rgba(255, 165, 0, 7)", size=14, family="Arial"),
                bgcolor="rgba(255, 165, 0, 0.15)",
                align="center",
                xref="x",
                yref="paper",
            )

            # Récupérer les rendements cumulés
            max_sharpe_train_cum_returns = st.session_state["max_sharpe_pop_train"].plot_cumulative_returns().data[0]
            max_sharpe_test_cum_returns = st.session_state["max_sharpe_pop_test"].plot_cumulative_returns().data[0]
            max_sharpe_portfolio_test = st.session_state["max_sharpe_pop_test"][0] # Récupérer le premier portefeuille de la population
            max_sharpe_test_returns_df = max_sharpe_portfolio_test.returns_df.squeeze() # Accéder aux rendements simples

            # Dernière valeur du Train Set pour aligner le Test Set
            last_train_value = max_sharpe_train_cum_returns["y"][-1]

            # Initialisation du premier élément du Test Set pour alignement
            test_returns_shifted = [last_train_value + max_sharpe_test_returns_df.iloc[0]]

            # Ajout des rendements suivants en additionnant à chaque fois au précédent
            for y in max_sharpe_test_returns_df.iloc[1:]:
                test_returns_shifted.append(test_returns_shifted[-1] + y)

            # Définir une couleur unique
            model_color = px.colors.qualitative.Set1[0]

            # Ajout du Train Set (ligne fine)
            fig.add_trace(go.Scatter(
                x=max_sharpe_train_cum_returns["x"], 
                y=[y * 100 for y in max_sharpe_train_cum_returns["y"]],  # Transformation en %
                mode='lines', 
                name="Train Set - Max Sharpe Ratio",
                line=dict(color=model_color, width=2),
                hovertemplate="<b>Train Set</b><br>%{x}<br>Return: %{y:.2f}%<extra></extra>"
            ))

            # Ajout du Test Set (ligne plus épaisse)
            fig.add_trace(go.Scatter(
                x=max_sharpe_test_cum_returns["x"], 
                y=[y * 100 for y in test_returns_shifted],  # Transformation en %
                mode='lines', 
                name="Test Set - Max Sharpe Ratio",
                line=dict(color=model_color, width=3.5),
                hovertemplate="<b>Test Set</b><br>%{x}<br>Return: %{y:.2f}%<extra></extra>"
            ))

            # Ajouter une ligne verticale pour la date de split
            fig.add_vline(x=test_start_date, line_dash="dot", line_color="purple")

            # Ajouter une annotation pour la Split Date
            fig.add_annotation(
                x=test_start_date, 
                y=1.1,  
                text=f"Split Date: {test_start_date.date()}", 
                showarrow=False,
                font=dict(color="white", size=12, family="Arial"),
                bgcolor="purple",
                opacity=0.8,
                xref="x",
                yref="paper",
                align="center"
            )

            # Mise en page finale
            fig.update_layout(
                title="Cumulative Returns (non-compounded) - Train & Test - Max Sharpe Ratio",
                xaxis_title="Observations",
                yaxis_title="Cumulative Returns (%)",
                yaxis_tickformat=".2f",
                legend_title="Portfolio",
                template="plotly_dark"
            )

            # Affichage du graphique combiné
            st.plotly_chart(fig, use_container_width=True)

            # Modifier le titre du graphique pour le Train Set
            fig_max_sharpe_pop_train_cum_returns = st.session_state["max_sharpe_pop_train"].plot_cumulative_returns()
            fig_max_sharpe_pop_train_cum_returns.update_layout(title="Cumulative Returns (non-compounded) - Train Set - Max Sharpe Ratio")
            st.plotly_chart(fig_max_sharpe_pop_train_cum_returns, use_container_width=True, key="max_sharpe_train_cum_returns")
            
            # Modifier le titre du graphique pour le Test Set
            fig_max_sharpe_pop_test_cum_returns = st.session_state["max_sharpe_pop_test"].plot_cumulative_returns()
            fig_max_sharpe_pop_test_cum_returns.update_layout(title="Cumulative Returns (non-compounded) - Test Set - Max Sharpe Ratio")
            st.plotly_chart(fig_max_sharpe_pop_test_cum_returns, use_container_width=True, key="max_sharpe_test_cum_returns")

            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)
            
            #--------------------------------------
            # Summaries
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Summary</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h6 style='text-align: center;'>Train Set</h6>", unsafe_allow_html=True)
                st.write(st.session_state["max_sharpe_pop_train"].summary())
            with col2:
                st.markdown("<h6 style='text-align: center;'>Test Set</h6>", unsafe_allow_html=True)
                st.write(st.session_state["max_sharpe_pop_test"].summary())
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 📌 Minimum CVaR
    # --------------------------------------------------------
    with mean_risk_tabs[1]:
        st.markdown(
            "<h2 style='text-align: center; color: #4B9CD3;'>Mean-Risk - Min CVaR</h2>",
            unsafe_allow_html=True
        )

        #---------------------------------------------------
        # Appliquer un style CSS au bouton Streamlit
        #---------------------------------------------------
        st.markdown("""
            <style>
                div.stButton > button:first-child {
                    background-color: rgb(75, 117, 255);
                    color: black;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 8px;
                    border: none;
                    width: 100%;
                }
                div.stButton > button:hover {
                    background-color: rgb(150, 219, 0);
                    color: black;
                }
            </style>
        """, unsafe_allow_html=True)

        #---------------------------------------------------
        # Initialisation des variables en session_state
        #---------------------------------------------------
        if "min_cvar_pop_train" not in st.session_state:
            st.session_state["min_cvar_pop_train"] = None
            st.session_state["min_cvar_pop_test"] = None

        # Bouton pour entraîner uniquement ce modèle
        if st.button("Train & Test - Min CVaR"):
            model_min_cvar = MeanRisk(
                risk_measure=RiskMeasure.CVAR,
                objective_function=ObjectiveFunction.MINIMIZE_RISK,
                portfolio_params=dict(name="Min CVaR"),
            )

            # Entraînement
            model_min_cvar.fit(data_returns_train)
            min_cvar_portfolio_train = model_min_cvar.predict(data_returns_train)
            min_cvar_portfolio_test = model_min_cvar.predict(data_returns_test)

            # Stocker les résultats dans session_state
            st.session_state["min_cvar_pop_train"] = Population([min_cvar_portfolio_train])
            st.session_state["min_cvar_pop_test"] = Population([min_cvar_portfolio_test])
                        
            # Display process succeeded!
            st.success("Your model has been trained and tested!")

        #----------------------------------------------------------------
        # Affichage des résultats (même après changement de tab)
        #----------------------------------------------------------------
        if st.session_state["min_cvar_pop_train"] is not None:
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

            #--------------------------------------
            # Compositions
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Composition</h3>", unsafe_allow_html=True)            
            st.plotly_chart(st.session_state["min_cvar_pop_train"].plot_composition(), use_container_width=True, key="min_cvar_composition")
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

            #--------------------------------------
            # Plot Cum Returns
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Cumulative Returns</h3>", unsafe_allow_html=True)

            # GRAPHIQUE COMBINANT TRAIN ET TEST SET
            fig = go.Figure()

            # Déterminer les limites des zones d’arrière-plan
            train_start_date = data_returns_train.index[0]
            train_end_date = data_returns_train.index[-1]
            test_start_date = data_returns_test.index[0]
            test_end_date = data_returns_test.index[-1]

            # Ajouter une zone d'arrière-plan pour le Train Set (bleu clair)
            fig.add_shape(
                type="rect",
                x0=train_start_date, x1=train_end_date,
                y0=0, y1=1,  # Étendre verticalement
                xref="x", yref="paper",
                fillcolor="rgba(0, 128, 255, 0.08)",  # Bleu très transparent
                layer="below",
                line_width=0,
            )

            # Ajouter une zone d'arrière-plan pour le Test Set (orange transparent)
            fig.add_shape(
                type="rect",
                x0=test_start_date, x1=test_end_date,
                y0=0, y1=1,
                xref="x", yref="paper",
                fillcolor="rgba(255, 165, 0, 0.12)",  # Orange légèrement plus transparent
                layer="below",
                line_width=0,
            )

            # Ajouter une annotation pour "Train Set"
            fig.add_annotation(
                x=train_start_date + (train_end_date - train_start_date) / 2,  
                y=1.05,  
                text="Train Set",  
                showarrow=False,
                font=dict(color="rgba(0, 128, 255, 0.7)", size=14, family="Arial"),
                bgcolor="rgba(0, 128, 255, 0.15)",
                align="center",
                xref="x",
                yref="paper",
            )

            # Ajouter une annotation pour "Test Set"
            fig.add_annotation(
                x=test_start_date + (test_end_date - test_start_date) / 2,  
                y=1.05,  
                text="Test Set",  
                showarrow=False,
                font=dict(color="rgba(255, 165, 0, 7)", size=14, family="Arial"),
                bgcolor="rgba(255, 165, 0, 0.15)",
                align="center",
                xref="x",
                yref="paper",
            )

            # Récupérer les rendements cumulés
            min_cvar_train_cum_returns = st.session_state["min_cvar_pop_train"].plot_cumulative_returns().data[0]
            min_cvar_test_cum_returns = st.session_state["min_cvar_pop_test"].plot_cumulative_returns().data[0]
            min_cvar_portfolio_test = st.session_state["min_cvar_pop_test"][0] # Récupérer le premier portefeuille de la population
            min_cvar_test_returns_df = min_cvar_portfolio_test.returns_df.squeeze() # Accéder aux rendements simples

            # Dernière valeur du Train Set pour aligner le Test Set
            last_train_value = min_cvar_train_cum_returns["y"][-1]

            # Initialisation du premier élément du Test Set pour alignement
            test_returns_shifted = [last_train_value + min_cvar_test_returns_df.iloc[0]]

            # Ajout des rendements suivants en additionnant à chaque fois au précédent
            for y in min_cvar_test_returns_df.iloc[1:]:
                test_returns_shifted.append(test_returns_shifted[-1] + y)

            # Définir une couleur unique
            model_color = px.colors.qualitative.Set1[0]

            # Ajout du Train Set (ligne fine)
            fig.add_trace(go.Scatter(
                x=min_cvar_train_cum_returns["x"], 
                y=[y * 100 for y in min_cvar_train_cum_returns["y"]],  # Transformation en %
                mode='lines', 
                name="Train Set - Minimum CVaR",
                line=dict(color=model_color, width=2),
                hovertemplate="<b>Train Set</b><br>%{x}<br>Return: %{y:.2f}%<extra></extra>"
            ))

            # Ajout du Test Set (ligne plus épaisse)
            fig.add_trace(go.Scatter(
                x=min_cvar_test_cum_returns["x"], 
                y=[y * 100 for y in test_returns_shifted],  # Transformation en %
                mode='lines', 
                name="Test Set - Minimum CVaR",
                line=dict(color=model_color, width=3.5),
                hovertemplate="<b>Test Set</b><br>%{x}<br>Return: %{y:.2f}%<extra></extra>"
            ))

            # Ajouter une ligne verticale pour la date de split
            fig.add_vline(x=test_start_date, line_dash="dot", line_color="purple")

            # Ajouter une annotation pour la Split Date
            fig.add_annotation(
                x=test_start_date, 
                y=1.1,  
                text=f"Split Date: {test_start_date.date()}", 
                showarrow=False,
                font=dict(color="white", size=12, family="Arial"),
                bgcolor="purple",
                opacity=0.8,
                xref="x",
                yref="paper",
                align="center"
            )

            # Mise en page finale
            fig.update_layout(
                title="Cumulative Returns (non-compounded) - Train & Test - Minimum CVaR",
                xaxis_title="Observations",
                yaxis_title="Cumulative Returns (%)",
                yaxis_tickformat=".2f",
                legend_title="Portfolio",
                template="plotly_dark"
            )

            # Affichage du graphique combiné
            st.plotly_chart(fig, use_container_width=True)

            # Modifier le titre du graphique pour le Train Set
            fig_min_cvar_pop_train_cum_returns = st.session_state["min_cvar_pop_train"].plot_cumulative_returns()
            fig_min_cvar_pop_train_cum_returns.update_layout(title="Cumulative Returns (non-compounded) - Train Set - Minimum CVaR")
            st.plotly_chart(fig_min_cvar_pop_train_cum_returns, use_container_width=True, key="min_cvar_train_cum_returns")
            
            # Modifier le titre du graphique pour le Test Set
            fig_min_cvar_pop_test_cum_returns = st.session_state["min_cvar_pop_test"].plot_cumulative_returns()
            fig_min_cvar_pop_test_cum_returns.update_layout(title="Cumulative Returns (non-compounded) - Test Set - Minimum CVaR")
            st.plotly_chart(fig_min_cvar_pop_test_cum_returns, use_container_width=True, key="min_cvar_test_cum_returns")

            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

            #--------------------------------------
            # Summaries
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Summary</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h6 style='text-align: center;'>Train Set</h6>", unsafe_allow_html=True)
                st.write(st.session_state["min_cvar_pop_train"].summary())
            with col2:
                st.markdown("<h6 style='text-align: center;'>Test Set</h6>", unsafe_allow_html=True)
                st.write(st.session_state["min_cvar_pop_test"].summary())
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 📌 EFFICIENT FRONTIER
    # --------------------------------------------------------
    with mean_risk_tabs[2]:
        st.markdown(
            "<h2 style='text-align: center; color: #4B9CD3;'>Mean-Risk - Efficient Frontier</h2>",
            unsafe_allow_html=True
        )

        #--------------------------------------------------------
        # Sélection du mode d'optimisation, Train, Test
        #--------------------------------------------------------
        col1, col2 = st.columns(2)
        with col1:
            frontier_mode = st.selectbox(
                "Choose Efficient Frontier Mode",
                ["Number of Portfolios", "Minimum Returns Constraint"],
                index=0
            )
        with col2:
            if frontier_mode == "Number of Portfolios":
                num_portfolios = st.slider("Select Number of Portfolios on the Efficient Frontier", 5, 50, 10, step=1)
                min_return_list = None
            else:
                min_return_input = st.text_input(
                    "Enter Minimum Returns (comma-separated, annualized %)", "15, 20, 25, 30, 35"
                )
                min_return_list = np.array([float(x) / 100 / 252 for x in min_return_input.split(",")])
                num_portfolios = None

        #--------------------------------------------------------
        # Initialisation des variables en session_state
        #--------------------------------------------------------
        if "efficient_frontier_pop_train" not in st.session_state:
                st.session_state["efficient_frontier_pop_train"] = None
                st.session_state["efficient_frontier_pop_test"] = None
                st.session_state["last_frontier_mode"] = frontier_mode
                st.session_state["last_num_portfolios"] = num_portfolios
                st.session_state["last_min_return_list"] = min_return_list
                st.session_state["input_changed"] = False

        # Détecter un changement d'input
        input_changed = (
            st.session_state["last_frontier_mode"] != frontier_mode or
            st.session_state["last_num_portfolios"] != num_portfolios or
            not np.array_equal(st.session_state["last_min_return_list"], min_return_list)
        )
        st.session_state["input_changed"] = input_changed

        # Stocker les valeurs de session dans des variables locales pour plus de clarté
        efficient_frontier_pop_train = st.session_state["efficient_frontier_pop_train"]
        efficient_frontier_pop_test = st.session_state["efficient_frontier_pop_test"]   
        
        #--------------------------------------
        # Train & Test
        #--------------------------------------
        # Appliquer un style CSS au bouton Streamlit
        st.markdown("""
            <style>
                div.stButton > button:first-child {
                    background-color: rgb(75, 117, 255);
                    color: black;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 8px;
                    border: none;
                    width: 100%;
                }
                div.stButton > button:hover {
                    background-color: rgb(150, 219, 0);
                    color: black;
                }
            </style>
         """, unsafe_allow_html=True)

        # Afficher un avertissement si l'input a changé
        if st.session_state["input_changed"]:
            st.warning("⚠️ You have changed the input. Please re-train the model to update results.")

        # Bouton d'entraînement 
        if st.button("Train & Test - Efficient Frontier"):

            # Sélection du modèle
            if frontier_mode == "Number of Portfolios":
                model_eff_front = MeanRisk(
                    risk_measure=RiskMeasure.VARIANCE,
                    efficient_frontier_size=num_portfolios,
                    portfolio_params=dict(name="Variance"),
                )
            else:
                model_eff_front = MeanRisk(
                    risk_measure=RiskMeasure.VARIANCE,
                    min_return=min_return_list,
                    portfolio_params=dict(name="Variance"),
                )

            # Entraînement
            model_eff_front.fit(data_returns_train)
            eff_front_pop_train = model_eff_front.predict(data_returns_train)
            eff_front_pop_test = model_eff_front.predict(data_returns_test)

            # Ajout des tags train/test
            eff_front_pop_train.set_portfolio_params(tag="Train")
            eff_front_pop_test.set_portfolio_params(tag="Test")

            # Stocker dans session_state
            st.session_state["efficient_frontier_pop_train"] = eff_front_pop_train
            st.session_state["efficient_frontier_pop_test"] = eff_front_pop_test

            # Une fois le modèle exécuté, réinitialiser l'indicateur de changement
            st.session_state["input_changed"] = False

            # Mettre à jour les valeurs de référence pour éviter le message d'avertissement
            st.session_state["last_frontier_mode"] = frontier_mode
            st.session_state["last_num_portfolios"] = num_portfolios
            st.session_state["last_min_return_list"] = min_return_list
                
            # Display process succeeded!
            if frontier_mode == "Number of Portfolios":
                st.success(f"Your model has been trained and tested with {num_portfolios} portfolios!")
            else:
                st.success(f"Your model has been trained and tested with the following minimum returns: {min_return_input}!")

        #--------------------------------------
        # Efficient Frontier
        #--------------------------------------
        if st.session_state["efficient_frontier_pop_train"] is not None:
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

            # Titre principal de la frontière efficiente
            st.markdown("<h3 style='color: #2ECC71;'>Efficient Frontier</h3>", unsafe_allow_html=True)

            eff_front_pop_train_and_test = st.session_state["efficient_frontier_pop_train"] + st.session_state["efficient_frontier_pop_test"] 
            fig = eff_front_pop_train_and_test.plot_measures(
                x=RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
                y=PerfMeasure.ANNUALIZED_MEAN,
                color_scale=RatioMeasure.ANNUALIZED_SHARPE_RATIO,  # Valeur utilisée pour la couleur
                hover_measures=[RiskMeasure.MAX_DRAWDOWN, RatioMeasure.ANNUALIZED_SORTINO_RATIO],
            )
            # Mise à jour du titre de la figure
            fig.update_layout(
                title="Efficient Frontier - Risk vs. Return",
                title_x=0,  # Centrer le titre
                title_font=dict(size=16, color="white"),  # Modifier la taille et couleur du titre
            )
            st.plotly_chart(fig, use_container_width=True, key="efficient_frontier")

            # Ajout d'une séparation visuelle entre les deux colonnes
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

            #--------------------------------------
            # Compositions
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Composition</h3>", unsafe_allow_html=True)
            st.plotly_chart(st.session_state["efficient_frontier_pop_train"].plot_composition(), use_container_width=True, key="efficient_frontier_composition")
            # Ajout d'une séparation visuelle entre les deux colonnes
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

            #--------------------------------------
            # Plot Cum Returns
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Cumulative Returns</h3>", unsafe_allow_html=True)

            # Affichage des résultats combinés Train & Test pour chaque portefeuille
            fig = go.Figure()

            # Déterminer les dates de split
            train_start_date = data_returns_train.index[0]
            train_end_date = data_returns_train.index[-1]
            test_start_date = data_returns_test.index[0]
            test_end_date = data_returns_test.index[-1]

            # Ajouter une zone d'arrière-plan pour le Train Set (bleu clair)
            fig.add_shape(
                type="rect",
                x0=train_start_date, x1=train_end_date,
                y0=0, y1=1,  # Étendre verticalement
                xref="x", yref="paper",
                fillcolor="rgba(0, 128, 255, 0.08)",  # Bleu très transparent
                layer="below",
                line_width=0,
            )

            # Ajouter une zone d'arrière-plan pour le Test Set (orange clair)
            fig.add_shape(
                type="rect",
                x0=test_start_date, x1=test_end_date,
                y0=0, y1=1,
                xref="x", yref="paper",
                fillcolor="rgba(255, 165, 0, 0.12)",  # Orange légèrement plus transparent
                layer="below",
                line_width=0,
            )

            # Ajouter une annotation pour "Train Set"
            fig.add_annotation(
                x=train_start_date + (train_end_date - train_start_date) / 2,  
                y=1.05,  
                text="Train Set",  
                showarrow=False,
                font=dict(color="rgba(0, 128, 255, 0.7)", size=14, family="Arial"),
                bgcolor="rgba(0, 128, 255, 0.15)",
                align="center",
                xref="x",
                yref="paper",
            )

            # Ajouter une annotation pour "Test Set"
            fig.add_annotation(
                x=test_start_date + (test_end_date - test_start_date) / 2,  
                y=1.05,  
                text="Test Set",  
                showarrow=False,
                font=dict(color="rgba(255, 165, 0, 0.7)", size=14, family="Arial"),
                bgcolor="rgba(255, 165, 0, 0.15)",
                align="center",
                xref="x",
                yref="paper",
            )

            # Boucle sur tous les portefeuilles générés (car on a plusieurs ptfs avec efficient frontier)
            for i, (train_ptf, test_ptf) in enumerate(zip(st.session_state["efficient_frontier_pop_train"], st.session_state["efficient_frontier_pop_test"])):
                
                # Récupérer les rendements cumulés des portefeuilles train & test
                train_cum_returns = train_ptf.plot_cumulative_returns().data[0]
                test_cum_returns = test_ptf.plot_cumulative_returns().data[0]
                test_returns = test_ptf.returns_df.squeeze()  # Récupérer les rendements simples du test set

                # Récupérer la dernière valeur du Train Set pour aligner le Test Set
                last_train_value = train_cum_returns["y"][-1]

                # Initialisation du premier élément du Test Set
                test_returns_shifted = [last_train_value + test_returns.iloc[0]]

                # Ajout des rendements suivants en les cumulant
                for y in test_returns.iloc[1:]:
                    test_returns_shifted.append(test_returns_shifted[-1] + y)

                # Définir une couleur unique par portefeuille
                model_color = px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]

                # Ajout du Train Set
                fig.add_trace(go.Scatter(
                    x=train_cum_returns["x"], 
                    y=[y * 100 for y in train_cum_returns["y"]],  # Transformation en %
                    mode='lines', 
                    name=f"Train Set - Portfolio {i+1}",
                    line=dict(color=model_color, width=2),
                    hovertemplate=f"<b>Portfolio {i+1} - Train Set</b><br>%{{x}}<br>Return: %{{y:.2f}}%<extra></extra>"
                ))

                # Ajout du Test Set
                fig.add_trace(go.Scatter(
                    x=test_cum_returns["x"], 
                    y=[y * 100 for y in test_returns_shifted],  # Transformation en %
                    mode='lines', 
                    name=f"Test Set - Portfolio {i+1}",
                    line=dict(color=model_color, width=3.5),
                    hovertemplate=f"<b>Portfolio {i+1} - Test Set</b><br>%{{x}}<br>Return: %{{y:.2f}}%<extra></extra>"
                ))

            # Ajouter une ligne verticale pour la date de split
            fig.add_vline(x=test_start_date, line_dash="dot", line_color="purple")

            # Ajouter une annotation pour la Split Date
            fig.add_annotation(
                x=test_start_date, 
                y=1.1,  
                text=f"Split Date: {test_start_date.date()}", 
                showarrow=False,
                font=dict(color="white", size=12, family="Arial"),
                bgcolor="purple",
                opacity=0.8,
                xref="x",
                yref="paper",
                align="center"
            )

            # Mise en page finale
            fig.update_layout(
                title="Cumulative Returns (non-compounded) - Train & Test - Efficient Frontier",
                xaxis_title="Observations",
                yaxis_title="Cumulative Returns (%)",
                yaxis_tickformat=".2f",  
                legend_title="Portfolios",
                template="plotly_dark"
            )

            # Affichage dans Streamlit
            st.plotly_chart(fig, use_container_width=True, key="efficient_frontier_plot_cum_ret_train&test")

            # Modifier le titre du graphique pour le Train Set
            fig_efficient_frontier_pop_train_cum_returns = st.session_state["efficient_frontier_pop_train"].plot_cumulative_returns()
            fig_efficient_frontier_pop_train_cum_returns.update_layout(title="Cumulative Returns (non-compounded) - Train Set - Efficient Frontier")
            st.plotly_chart(fig_efficient_frontier_pop_train_cum_returns, use_container_width=True, key="efficient_frontier_train_cum_returns")
            
            # Modifier le titre du graphique pour le Test Set
            fig_efficient_frontier_pop_test_cum_returns = st.session_state["efficient_frontier_pop_test"].plot_cumulative_returns()
            fig_efficient_frontier_pop_test_cum_returns.update_layout(title="Cumulative Returns (non-compounded) - Test Set - Efficient Frontier")
            st.plotly_chart(fig_efficient_frontier_pop_test_cum_returns, use_container_width=True, key="efficient_frontier_test_cum_returns")

            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

            #--------------------------------------
            # Summaries
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Summary</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h6 style='text-align: center;'>Train Set</h6>", unsafe_allow_html=True)
                st.write(st.session_state["efficient_frontier_pop_train"].summary())
            with col2:
                st.markdown("<h6 style='text-align: center;'>Test Set</h6>", unsafe_allow_html=True)
                st.write(st.session_state["efficient_frontier_pop_test"].summary())
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)


# --------------------------------------------------------
# 📌 TAB 3: RISK BUDGETING
# --------------------------------------------------------
with main_tab[3]:
    # Création des sous-tabs pour la comparaison et les modèles spécifiques
    risk_budg_tabs = st.tabs(["Risk Parity - Variance", "Risk Budgeting - CVaR", "Risk Parity - Covariance Shrinkage" ])

    # --------------------------------------------------------
    # 📌 Risk Parity - Variance
    # --------------------------------------------------------
    with risk_budg_tabs[0]:
        st.markdown(
            "<h2 style='text-align: center; color: #4B9CD3;'>Risk Parity - Variance</h2>",
            unsafe_allow_html=True
        )

        #--------------------------------------
        # Train & Test
        #--------------------------------------
        # Appliquer un style CSS au bouton Streamlit
        st.markdown("""
            <style>
                div.stButton > button:first-child {
                    background-color: rgb(75, 117, 255);
                    color: black;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 8px;
                    border: none;
                    width: 100%;
                }
                div.stButton > button:hover {
                    background-color: rgb(150, 219, 0);
                    color: black;
                }
            </style>
         """, unsafe_allow_html=True)
        
        # Initialisation des modèles en session_state s'ils n'existent pas
        if "risk_par_var_pop_train" not in st.session_state:
            st.session_state["risk_par_var_pop_train"] = None
            st.session_state["risk_par_var_pop_test"] = None
        
        # Bouton pour entraîner uniquement ce modèle
        if st.button("Train & Test - Risk Parity - Variance"):
            model_risk_par_var = RiskBudgeting(
                            risk_measure=RiskMeasure.VARIANCE,
                            portfolio_params=dict(name="Risk Parity - Variance"),
            )

            # Entraînement
            model_risk_par_var.fit(data_returns_train)
            risk_par_var_portfolio_train = model_risk_par_var.predict(data_returns_train)
            risk_par_var_portfolio_test = model_risk_par_var.predict(data_returns_test)

            # Stocker les résultats dans session_state
            st.session_state["risk_par_var_pop_train"] = Population([risk_par_var_portfolio_train])
            st.session_state["risk_par_var_pop_test"] = Population([risk_par_var_portfolio_test])
            
            # Display process succeeded!
            st.success("Your model has been trained and tested!")

        #--------------------------------------------------------------------------
        # Affichage des résultats uniquement si un modèle a été entraîné
        #--------------------------------------------------------------------------
        if st.session_state["risk_par_var_pop_train"] is not None:
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)
            
            #--------------------------------------
            # Compositions
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Composition</h3>", unsafe_allow_html=True)
            st.plotly_chart(st.session_state["risk_par_var_pop_train"].plot_composition(), use_container_width=True, key="risk_par_var_composition")
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

            #--------------------------------------
            # Plot Cum Returns
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Cumulative Returns</h3>", unsafe_allow_html=True)

            # GRAPHIQUE COMBINANT TRAIN ET TEST SET
            fig = go.Figure()

            # Déterminer les limites des zones d’arrière-plan
            train_start_date = data_returns_train.index[0]
            train_end_date = data_returns_train.index[-1]
            test_start_date = data_returns_test.index[0]
            test_end_date = data_returns_test.index[-1]

            # Ajouter une zone d'arrière-plan pour le Train Set (bleu clair)
            fig.add_shape(
                type="rect",
                x0=train_start_date, x1=train_end_date,
                y0=0, y1=1,  # Étendre verticalement
                xref="x", yref="paper",
                fillcolor="rgba(0, 128, 255, 0.08)",  # Bleu très transparent
                layer="below",
                line_width=0,
            )

            # Ajouter une zone d'arrière-plan pour le Test Set (orange transparent)
            fig.add_shape(
                type="rect",
                x0=test_start_date, x1=test_end_date,
                y0=0, y1=1,
                xref="x", yref="paper",
                fillcolor="rgba(255, 165, 0, 0.12)",  # Orange légèrement plus transparent
                layer="below",
                line_width=0,
            )

            # Ajouter une annotation pour "Train Set"
            fig.add_annotation(
                x=train_start_date + (train_end_date - train_start_date) / 2,  
                y=1.05,  
                text="Train Set",  
                showarrow=False,
                font=dict(color="rgba(0, 128, 255, 0.7)", size=14, family="Arial"),
                bgcolor="rgba(0, 128, 255, 0.15)",
                align="center",
                xref="x",
                yref="paper",
            )

            # Ajouter une annotation pour "Test Set"
            fig.add_annotation(
                x=test_start_date + (test_end_date - test_start_date) / 2,  
                y=1.05,  
                text="Test Set",  
                showarrow=False,
                font=dict(color="rgba(255, 165, 0, 7)", size=14, family="Arial"),
                bgcolor="rgba(255, 165, 0, 0.15)",
                align="center",
                xref="x",
                yref="paper",
            )

            # Récupérer les rendements cumulés
            risk_par_var_train_cum_returns = st.session_state["risk_par_var_pop_train"].plot_cumulative_returns().data[0]
            risk_par_var_test_cum_returns = st.session_state["risk_par_var_pop_test"].plot_cumulative_returns().data[0]
            risk_par_var_portfolio_test = st.session_state["risk_par_var_pop_test"][0] # Récupérer le premier portefeuille de la population
            risk_par_var_test_returns_df = risk_par_var_portfolio_test.returns_df.squeeze() # Accéder aux rendements simples

            # Dernière valeur du Train Set pour aligner le Test Set
            last_train_value = risk_par_var_train_cum_returns["y"][-1]

            # Initialisation du premier élément du Test Set pour alignement
            test_returns_shifted = [last_train_value + risk_par_var_test_returns_df.iloc[0]]

            # Ajout des rendements suivants en additionnant à chaque fois au précédent
            for y in risk_par_var_test_returns_df.iloc[1:]:
                test_returns_shifted.append(test_returns_shifted[-1] + y)

            # Définir une couleur unique
            model_color = px.colors.qualitative.Set1[0]

            # Ajout du Train Set (ligne fine)
            fig.add_trace(go.Scatter(
                x=risk_par_var_train_cum_returns["x"], 
                y=[y * 100 for y in risk_par_var_train_cum_returns["y"]],  # Transformation en %
                mode='lines', 
                name="Train Set - Risk Parity - Variance",
                line=dict(color=model_color, width=2),
                hovertemplate="<b>Train Set</b><br>%{x}<br>Return: %{y:.2f}%<extra></extra>"
            ))

            # Ajout du Test Set (ligne plus épaisse)
            fig.add_trace(go.Scatter(
                x=risk_par_var_test_cum_returns["x"], 
                y=[y * 100 for y in test_returns_shifted],  # Transformation en %
                mode='lines', 
                name="Test Set - Risk Parity - Variance",
                line=dict(color=model_color, width=3.5),
                hovertemplate="<b>Test Set</b><br>%{x}<br>Return: %{y:.2f}%<extra></extra>"
            ))

            # Ajouter une ligne verticale pour la date de split
            fig.add_vline(x=test_start_date, line_dash="dot", line_color="purple")

            # Ajouter une annotation pour la Split Date
            fig.add_annotation(
                x=test_start_date, 
                y=1.1,  
                text=f"Split Date: {test_start_date.date()}", 
                showarrow=False,
                font=dict(color="white", size=12, family="Arial"),
                bgcolor="purple",
                opacity=0.8,
                xref="x",
                yref="paper",
                align="center"
            )

            # Mise en page finale
            fig.update_layout(
                title="Cumulative Returns (non-compounded) - Train & Test - Risk Parity - Variance",
                xaxis_title="Observations",
                yaxis_title="Cumulative Returns (%)",
                yaxis_tickformat=".2f",
                legend_title="Portfolio",
                template="plotly_dark"
            )

            # Affichage du graphique combiné
            st.plotly_chart(fig, use_container_width=True)

            # Modifier le titre du graphique pour le Train Set
            fig_risk_par_var_pop_train_cum_returns = st.session_state["risk_par_var_pop_train"].plot_cumulative_returns()
            fig_risk_par_var_pop_train_cum_returns.update_layout(title="Cumulative Returns (non-compounded) - Train Set - Risk Parity - Variance")
            st.plotly_chart(fig_risk_par_var_pop_train_cum_returns, use_container_width=True, key="risk_par_var_train_cum_returns")
            
            # Modifier le titre du graphique pour le Test Set
            fig_risk_par_var_pop_test_cum_returns = st.session_state["risk_par_var_pop_test"].plot_cumulative_returns()
            fig_risk_par_var_pop_test_cum_returns.update_layout(title="Cumulative Returns (non-compounded) - Test Set - Risk Parity - Variance")
            st.plotly_chart(fig_risk_par_var_pop_test_cum_returns, use_container_width=True, key="risk_par_var_test_cum_returns")

            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

            #--------------------------------------
            # Summaries
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Summary</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h6 style='text-align: center;'>Train Set</h6>", unsafe_allow_html=True)
                st.write(st.session_state["risk_par_var_pop_train"].summary())
            with col2:
                st.markdown("<h6 style='text-align: center;'>Test Set</h6>", unsafe_allow_html=True)
                st.write(st.session_state["risk_par_var_pop_test"].summary())
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # 📌 Risk Budgeting - CVaR
    # --------------------------------------------------------
    with risk_budg_tabs[1]:
        st.markdown(
            "<h2 style='text-align: center; color: #4B9CD3;'>Risk Budgeting - CVaR</h2>",
            unsafe_allow_html=True
        )

        # Initialisation des budgets si non définis
        if "risk_budg_cvar" not in st.session_state:
            st.session_state["risk_budg_cvar"] = {asset: 1.0 for asset in data_returns_train.columns}

        # Convertir en DataFrame avec tickers en colonnes et une seule ligne pour Risk Budget
        risk_budget_df = pd.DataFrame([st.session_state["risk_budg_cvar"]], index=["Risk Budget"])

        # Affichage du tableau modifiable avec les valeurs en ligne
        st.markdown("### Configure your Risk Budget in the table below")
        edited_risk_budget = st.data_editor(
            risk_budget_df, 
            use_container_width=True, 
            key="risk_budget_table"
        )

        # Mise à jour du session_state avec les nouvelles valeurs
        st.session_state["risk_budg_cvar"] = edited_risk_budget.loc["Risk Budget"].to_dict()

        # Initialisation de la population si elle n'existe pas
        if "risk_budg_cvar_pop_train" not in st.session_state:
            st.session_state["risk_budg_cvar_pop_train"] = None
            st.session_state["risk_budg_cvar_pop_test"] = None

        # Bouton pour entraîner uniquement ce modèle
        if st.button("Train & Test - Risk Budgeting - CVaR"):
            model_risk_budg_cvar = RiskBudgeting(
                risk_measure=RiskMeasure.CVAR,  # Vérifie que CVaR est bien correct
                risk_budget=st.session_state["risk_budg_cvar"],  # Utilise le budget défini par l'utilisateur
                portfolio_params=dict(name="Risk Budgeting - CVaR"),
            )

            # Entraînement
            model_risk_budg_cvar.fit(data_returns_train)
            risk_budg_cvar_portfolio_train = model_risk_budg_cvar.predict(data_returns_train)
            risk_budg_cvar_portfolio_test = model_risk_budg_cvar.predict(data_returns_test)

            # Stocker les résultats dans session_state
            st.session_state["risk_budg_cvar_pop_train"] = Population([risk_budg_cvar_portfolio_train])
            st.session_state["risk_budg_cvar_pop_test"] = Population([risk_budg_cvar_portfolio_test])

            # Display process succeeded!
            st.success("Your model has been trained and tested!")

        #--------------------------------------------------------------------------
        # Affichage des résultats uniquement si un modèle a été entraîné
        #--------------------------------------------------------------------------
        if st.session_state["risk_budg_cvar_pop_train"] is not None:
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)
            
            #--------------------------------------
            # Compositions
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Composition</h3>", unsafe_allow_html=True)
            st.plotly_chart(
                st.session_state["risk_budg_cvar_pop_train"].plot_composition(), 
                use_container_width=True, 
                key="risk_budg_cvar_composition"
            )
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)


            #--------------------------------------
            # Plot Cum Returns
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Cumulative Returns</h3>", unsafe_allow_html=True)

            # GRAPHIQUE COMBINANT TRAIN ET TEST SET
            fig = go.Figure()

            # Déterminer les limites des zones d’arrière-plan
            train_start_date = data_returns_train.index[0]
            train_end_date = data_returns_train.index[-1]
            test_start_date = data_returns_test.index[0]
            test_end_date = data_returns_test.index[-1]

            # Ajouter une zone d'arrière-plan pour le Train Set (bleu clair)
            fig.add_shape(
                type="rect",
                x0=train_start_date, x1=train_end_date,
                y0=0, y1=1,  # Étendre verticalement
                xref="x", yref="paper",
                fillcolor="rgba(0, 128, 255, 0.08)",  # Bleu très transparent
                layer="below",
                line_width=0,
            )

            # Ajouter une zone d'arrière-plan pour le Test Set (orange transparent)
            fig.add_shape(
                type="rect",
                x0=test_start_date, x1=test_end_date,
                y0=0, y1=1,
                xref="x", yref="paper",
                fillcolor="rgba(255, 165, 0, 0.12)",  # Orange légèrement plus transparent
                layer="below",
                line_width=0,
            )

            # Ajouter une annotation pour "Train Set"
            fig.add_annotation(
                x=train_start_date + (train_end_date - train_start_date) / 2,  
                y=1.05,  
                text="Train Set",  
                showarrow=False,
                font=dict(color="rgba(0, 128, 255, 0.7)", size=14, family="Arial"),
                bgcolor="rgba(0, 128, 255, 0.15)",
                align="center",
                xref="x",
                yref="paper",
            )

            # Ajouter une annotation pour "Test Set"
            fig.add_annotation(
                x=test_start_date + (test_end_date - test_start_date) / 2,  
                y=1.05,  
                text="Test Set",  
                showarrow=False,
                font=dict(color="rgba(255, 165, 0, 7)", size=14, family="Arial"),
                bgcolor="rgba(255, 165, 0, 0.15)",
                align="center",
                xref="x",
                yref="paper",
            )

            # Récupérer les rendements cumulés
            risk_budg_cvar_train_cum_returns = st.session_state["risk_budg_cvar_pop_train"].plot_cumulative_returns().data[0]
            risk_budg_cvar_test_cum_returns = st.session_state["risk_budg_cvar_pop_test"].plot_cumulative_returns().data[0]
            risk_budg_cvar_portfolio_test = st.session_state["risk_budg_cvar_pop_test"][0] # Récupérer le premier portefeuille de la population
            risk_budg_cvar_test_returns_df = risk_budg_cvar_portfolio_test.returns_df.squeeze() # Accéder aux rendements simples

            # Dernière valeur du Train Set pour aligner le Test Set
            last_train_value = risk_budg_cvar_train_cum_returns["y"][-1]

            # Initialisation du premier élément du Test Set pour alignement
            test_returns_shifted = [last_train_value + risk_budg_cvar_test_returns_df.iloc[0]]

            # Ajout des rendements suivants en additionnant à chaque fois au précédent
            for y in risk_budg_cvar_test_returns_df.iloc[1:]:
                test_returns_shifted.append(test_returns_shifted[-1] + y)

            # Définir une couleur unique
            model_color = px.colors.qualitative.Set1[0]

            # Ajout du Train Set (ligne fine)
            fig.add_trace(go.Scatter(
                x=risk_budg_cvar_train_cum_returns["x"], 
                y=[y * 100 for y in risk_budg_cvar_train_cum_returns["y"]],  # Transformation en %
                mode='lines', 
                name="Train Set - Risk Budgeting - CVaR",
                line=dict(color=model_color, width=2),
                hovertemplate="<b>Train Set</b><br>%{x}<br>Return: %{y:.2f}%<extra></extra>"
            ))

            # Ajout du Test Set (ligne plus épaisse)
            fig.add_trace(go.Scatter(
                x=risk_budg_cvar_test_cum_returns["x"], 
                y=[y * 100 for y in test_returns_shifted],  # Transformation en %
                mode='lines', 
                name="Test Set - Risk Budgeting - CVaR",
                line=dict(color=model_color, width=3.5),
                hovertemplate="<b>Test Set</b><br>%{x}<br>Return: %{y:.2f}%<extra></extra>"
            ))

            # Ajouter une ligne verticale pour la date de split
            fig.add_vline(x=test_start_date, line_dash="dot", line_color="purple")

            # Ajouter une annotation pour la Split Date
            fig.add_annotation(
                x=test_start_date, 
                y=1.1,  
                text=f"Split Date: {test_start_date.date()}", 
                showarrow=False,
                font=dict(color="white", size=12, family="Arial"),
                bgcolor="purple",
                opacity=0.8,
                xref="x",
                yref="paper",
                align="center"
            )

            # Mise en page finale
            fig.update_layout(
                title="Cumulative Returns (non-compounded) - Train & Test - Risk Budgeting - CVaR",
                xaxis_title="Observations",
                yaxis_title="Cumulative Returns (%)",
                yaxis_tickformat=".2f",
                legend_title="Portfolio",
                template="plotly_dark"
            )

            # Affichage du graphique combiné
            st.plotly_chart(fig, use_container_width=True)

            # Modifier le titre du graphique pour le Train Set
            fig_risk_budg_cvar_pop_train_cum_returns = st.session_state["risk_budg_cvar_pop_train"].plot_cumulative_returns()
            fig_risk_budg_cvar_pop_train_cum_returns.update_layout(title="Cumulative Returns (non-compounded) - Train Set - Risk Budgeting - CVaR")
            st.plotly_chart(fig_risk_budg_cvar_pop_train_cum_returns, use_container_width=True, key="risk_budg_cvar_train_cum_returns")
            
            # Modifier le titre du graphique pour le Test Set
            fig_risk_budg_cvar_pop_test_cum_returns = st.session_state["risk_budg_cvar_pop_test"].plot_cumulative_returns()
            fig_risk_budg_cvar_pop_test_cum_returns.update_layout(title="Cumulative Returns (non-compounded) - Test Set - Risk Budgeting - CVaR")
            st.plotly_chart(fig_risk_budg_cvar_pop_test_cum_returns, use_container_width=True, key="risk_budg_cvar_test_cum_returns")

            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

            #--------------------------------------
            # Summaries
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Summary</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h6 style='text-align: center;'>Train Set</h6>", unsafe_allow_html=True)
                st.write(st.session_state["risk_budg_cvar_pop_train"].summary())
            with col2:
                st.markdown("<h6 style='text-align: center;'>Test Set</h6>", unsafe_allow_html=True)
                st.write(st.session_state["risk_budg_cvar_pop_test"].summary())
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # 📌 Risk Parity - Covariance Shrinkage
    # --------------------------------------------------------
    with risk_budg_tabs[2]:  # Vérifie que l'index correspond bien à l'onglet correct
        st.markdown(
            "<h2 style='text-align: center; color: #4B9CD3;'>Risk Parity - Covariance Shrinkage</h2>",
            unsafe_allow_html=True
        )

        #--------------------------------------
        # Input utilisateur pour le Shrinkage
        #--------------------------------------
        st.markdown("### Configure the Shrinkage Parameter")

        # Initialisation des variables en session_state pour suivre les changements
        if "last_shrinkage_value" not in st.session_state:
            st.session_state["last_shrinkage_value"] = None
            st.session_state["shrinkage_input_changed"] = False

        shrinkage_value = st.number_input(
            label="Enter Shrinkage coefficient (λ)", 
            value=0.1, 
            step=0.01,
            format="%.2f"
        )

        # Détecter un changement de coefficient
        input_changed = st.session_state["last_shrinkage_value"] is not None and shrinkage_value != st.session_state["last_shrinkage_value"]
        st.session_state["shrinkage_input_changed"] = input_changed

        # Affichage d'un avertissement si l'input a changé
        if input_changed:
            st.warning("⚠️ The shrinkage coefficient has changed. Please re-train the model to update results.")

        # Mettre à jour la dernière valeur enregistrée
        st.session_state["last_shrinkage_value"] = shrinkage_value

        if shrinkage_value < 0 or shrinkage_value > 1:
            st.error("🚨 Error: The coefficient must be between 0 and 1.")
            shrinkage_valid = False  # Désactive le bouton
        else:
            shrinkage_valid = True  # Active le bouton si la valeur est valide

        #--------------------------------------
        # Train & Test
        #--------------------------------------
        # Appliquer un style CSS au bouton Streamlit
        st.markdown("""
            <style>
                div.stButton > button:first-child {
                    background-color: rgb(75, 117, 255);
                    color: black;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 8px;
                    border: none;
                    width: 100%;
                }
                div.stButton > button:hover {
                    background-color: rgb(150, 219, 0);
                    color: black;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Initialisation des modèles en session_state s'ils n'existent pas
        if "risk_par_cov_shr_pop_train" not in st.session_state:
            st.session_state["risk_par_cov_shr_pop_train"] = None
            st.session_state["risk_par_cov_shr_pop_test"] = None

        # Bouton désactivé si shrinkage_value est invalide
        run_button_risk_par_cov_shr = st.button("Train & Test - Risk Parity - Covariance Shrinkage", disabled=not shrinkage_valid)    
        if run_button_risk_par_cov_shr:            
            
            model_risk_par_cov_shr = RiskBudgeting(
                risk_measure=RiskMeasure.VARIANCE,
                prior_estimator=EmpiricalPrior(covariance_estimator=ShrunkCovariance(shrinkage=shrinkage_value)),  # Utilisation de l'input utilisateur
                portfolio_params=dict(name="Risk Parity - Covariance Shrinkage"),
            )

            # Entraînement
            model_risk_par_cov_shr.fit(data_returns_train)
            risk_par_cov_shr_portfolio_train = model_risk_par_cov_shr.predict(data_returns_train)
            risk_par_cov_shr_portfolio_test = model_risk_par_cov_shr.predict(data_returns_test)

            # Stocker les résultats dans session_state
            st.session_state["risk_par_cov_shr_pop_train"] = Population([risk_par_cov_shr_portfolio_train])
            st.session_state["risk_par_cov_shr_pop_test"] = Population([risk_par_cov_shr_portfolio_test])
            
            # Display process succeeded!
            st.success(f"Your model has been trained and tested with shrinkage = {shrinkage_value}!")

        #--------------------------------------------------------------------------
        # Affichage des résultats uniquement si un modèle a été entraîné
        #--------------------------------------------------------------------------
        if st.session_state["risk_par_cov_shr_pop_train"] is not None:
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)
            
            #--------------------------------------
            # Compositions
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Composition</h3>", unsafe_allow_html=True)
            st.plotly_chart(st.session_state["risk_par_cov_shr_pop_train"].plot_composition(), use_container_width=True, key="risk_par_cov_shr_composition")
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

            #--------------------------------------
            # Plot Cum Returns
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Cumulative Returns</h3>", unsafe_allow_html=True)

            # GRAPHIQUE COMBINANT TRAIN ET TEST SET
            fig = go.Figure()

            # Déterminer les limites des zones d’arrière-plan
            train_start_date = data_returns_train.index[0]
            train_end_date = data_returns_train.index[-1]
            test_start_date = data_returns_test.index[0]
            test_end_date = data_returns_test.index[-1]

            # Ajouter une zone d'arrière-plan pour le Train Set (bleu clair)
            fig.add_shape(
                type="rect",
                x0=train_start_date, x1=train_end_date,
                y0=0, y1=1,  # Étendre verticalement
                xref="x", yref="paper",
                fillcolor="rgba(0, 128, 255, 0.08)",  # Bleu très transparent
                layer="below",
                line_width=0,
            )

            # Ajouter une zone d'arrière-plan pour le Test Set (orange transparent)
            fig.add_shape(
                type="rect",
                x0=test_start_date, x1=test_end_date,
                y0=0, y1=1,
                xref="x", yref="paper",
                fillcolor="rgba(255, 165, 0, 0.12)",  # Orange légèrement plus transparent
                layer="below",
                line_width=0,
            )

            # Ajouter une annotation pour "Train Set"
            fig.add_annotation(
                x=train_start_date + (train_end_date - train_start_date) / 2,  
                y=1.05,  
                text="Train Set",  
                showarrow=False,
                font=dict(color="rgba(0, 128, 255, 0.7)", size=14, family="Arial"),
                bgcolor="rgba(0, 128, 255, 0.15)",
                align="center",
                xref="x",
                yref="paper",
            )

            # Ajouter une annotation pour "Test Set"
            fig.add_annotation(
                x=test_start_date + (test_end_date - test_start_date) / 2,  
                y=1.05,  
                text="Test Set",  
                showarrow=False,
                font=dict(color="rgba(255, 165, 0, 7)", size=14, family="Arial"),
                bgcolor="rgba(255, 165, 0, 0.15)",
                align="center",
                xref="x",
                yref="paper",
            )

            # Récupérer les rendements cumulés
            risk_par_cov_shr_train_cum_returns = st.session_state["risk_par_cov_shr_pop_train"].plot_cumulative_returns().data[0]
            risk_par_cov_shr_test_cum_returns = st.session_state["risk_par_cov_shr_pop_test"].plot_cumulative_returns().data[0]
            risk_par_cov_shr_portfolio_test = st.session_state["risk_par_cov_shr_pop_test"][0] # Récupérer le premier portefeuille de la population
            risk_par_cov_shr_test_returns_df = risk_par_cov_shr_portfolio_test.returns_df.squeeze() # Accéder aux rendements simples

            # Dernière valeur du Train Set pour aligner le Test Set
            last_train_value = risk_par_cov_shr_train_cum_returns["y"][-1]

            # Initialisation du premier élément du Test Set pour alignement
            test_returns_shifted = [last_train_value + risk_par_cov_shr_test_returns_df.iloc[0]]

            # Ajout des rendements suivants en additionnant à chaque fois au précédent
            for y in risk_par_cov_shr_test_returns_df.iloc[1:]:
                test_returns_shifted.append(test_returns_shifted[-1] + y)

            # Définir une couleur unique
            model_color = px.colors.qualitative.Set1[0]

            # Ajout du Train Set (ligne fine)
            fig.add_trace(go.Scatter(
                x=risk_par_cov_shr_train_cum_returns["x"], 
                y=[y * 100 for y in risk_par_cov_shr_train_cum_returns["y"]],  # Transformation en %
                mode='lines', 
                name="Train Set - Risk Parity - Covariance Shrinkage",
                line=dict(color=model_color, width=2),
                hovertemplate="<b>Train Set</b><br>%{x}<br>Return: %{y:.2f}%<extra></extra>"
            ))

            # Ajout du Test Set (ligne plus épaisse)
            fig.add_trace(go.Scatter(
                x=risk_par_cov_shr_test_cum_returns["x"], 
                y=[y * 100 for y in test_returns_shifted],  # Transformation en %
                mode='lines', 
                name="Test Set - Risk Parity - Covariance Shrinkage",
                line=dict(color=model_color, width=3.5),
                hovertemplate="<b>Test Set</b><br>%{x}<br>Return: %{y:.2f}%<extra></extra>"
            ))

            # Ajouter une ligne verticale pour la date de split
            fig.add_vline(x=test_start_date, line_dash="dot", line_color="purple")

            # Ajouter une annotation pour la Split Date
            fig.add_annotation(
                x=test_start_date, 
                y=1.1,  
                text=f"Split Date: {test_start_date.date()}", 
                showarrow=False,
                font=dict(color="white", size=12, family="Arial"),
                bgcolor="purple",
                opacity=0.8,
                xref="x",
                yref="paper",
                align="center"
            )

            # Mise en page finale
            fig.update_layout(
                title="Cumulative Returns (non-compounded) - Train & Test - Risk Parity - Covariance Shrinkage",
                xaxis_title="Observations",
                yaxis_title="Cumulative Returns (%)",
                yaxis_tickformat=".2f",
                legend_title="Portfolio",
                template="plotly_dark"
            )

            # Affichage du graphique combiné
            st.plotly_chart(fig, use_container_width=True)

            # Modifier le titre du graphique pour le Train Set
            fig_risk_par_cov_shr_pop_train_cum_returns = st.session_state["risk_par_cov_shr_pop_train"].plot_cumulative_returns()
            fig_risk_par_cov_shr_pop_train_cum_returns.update_layout(title="Cumulative Returns (non-compounded) - Train Set - Risk Parity - Covariance Shrinkage")
            st.plotly_chart(fig_risk_par_cov_shr_pop_train_cum_returns, use_container_width=True, key="risk_par_cov_shr_train_cum_returns")
            
            # Modifier le titre du graphique pour le Test Set
            fig_risk_par_cov_shr_pop_test_cum_returns = st.session_state["risk_par_cov_shr_pop_test"].plot_cumulative_returns()
            fig_risk_par_cov_shr_pop_test_cum_returns.update_layout(title="Cumulative Returns (non-compounded) - Test Set - Risk Parity - Covariance Shrinkage")
            st.plotly_chart(fig_risk_par_cov_shr_pop_test_cum_returns, use_container_width=True, key="risk_par_cov_shr_test_cum_returns")

            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

            #--------------------------------------
            # Summaries
            #--------------------------------------
            st.markdown("<h3 style='color: #2ECC71;'>Portfolio Summary</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h6 style='text-align: center;'>Train Set</h6>", unsafe_allow_html=True)
                st.write(st.session_state["risk_par_cov_shr_pop_train"].summary())
            with col2:
                st.markdown("<h6 style='text-align: center;'>Test Set</h6>", unsafe_allow_html=True)
                st.write(st.session_state["risk_par_cov_shr_pop_test"].summary())
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)