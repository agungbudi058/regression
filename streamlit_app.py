import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro

st.title("SPSS-style Regression Analysis (No sklearn) 📊")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(df.head())

    # Variable selection
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    target = st.selectbox("Dependent Variable", numeric_cols)
    features = st.multiselect("Independent Variables", [c for c in numeric_cols if c != target])

    if features and target:
        # Create formula for statsmodels
        formula = f"{target} ~ {' + '.join(features)}"
        
        # Add polynomial terms
        if st.checkbox("Add Polynomial Terms"):
            degree = st.slider("Polynomial Degree", 1, 3, 1)
            formula += f" + {' + '.join([f'np.power({f}, {d})' for f in features for d in range(2, degree+1)])}"

        # Train-test split (manual implementation)
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        np.random.seed(42)
        test_mask = np.random.rand(len(df)) < test_size
        train_df = df[~test_mask]
        test_df = df[test_mask]

        if st.button("Run Analysis"):
            # Fit OLS model
            model = smf.ols(formula, data=train_df).fit()
            
            # Full model summary
            st.subheader("Regression Summary")
            st.markdown(model.summary().as_html(), unsafe_allow_html=True)

            # ANOVA table
            st.subheader("ANOVA Table")
            anova = sm.stats.anova_lm(model, typ=2)
            st.dataframe(anova.style.format("{:.4f}"))

            # Coefficients table
            st.subheader("Coefficients")
            conf_int = model.conf_int()
            conf_int.columns = ["95% CI Lower", "95% CI Upper"]
            coeff_table = pd.DataFrame({
                "B": model.params,
                "SE": model.bse,
                "Beta": model.params / train_df[target].std(),
                "t": model.tvalues,
                "p": model.pvalues
            }).join(conf_int)
            st.dataframe(coeff_table.style.format("{:.4f}"))

            # Collinearity diagnostics
            st.subheader("Collinearity Statistics")
            X = pd.get_dummies(train_df[features], drop_first=True)
            X = sm.add_constant(X)
            vif_data = pd.DataFrame({
                "Variable": X.columns,
                "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            })
            st.dataframe(vif_data.style.format({"VIF": "{:.2f}"}))

            # Residual diagnostics
            st.subheader("Residual Statistics")
            residuals = test_df[target] - model.predict(test_df)
            resid_stats = pd.DataFrame({
                "Statistic": ["Mean", "Std. Deviation", "Durbin-Watson", "Shapiro-Wilk p-value"],
                "Value": [
                    np.mean(residuals),
                    np.std(residuals, ddof=1),
                    durbin_watson(residuals),
                    shapiro(residuals)[1]
                ]
                # 1. Linearity: Actual vs. Predicted Plot
            residuals = model.resid        # Residuals (errors)
            fitted_values = model.predict(X)  # Predicted values
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=fitted_values, y=y)
            plt.xlabel("Predicted Values")
            plt.ylabel("Actual Values")
            plt.title("Linearity Check: Actual vs. Predicted")
            plt.show()

            })
            st.dataframe(resid_stats.set_index("Statistic"))

else:
    st.info("Please upload a CSV file to begin")
