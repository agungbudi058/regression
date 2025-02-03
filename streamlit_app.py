import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_goldfeldquandt
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Complete Regression Analysis with Assumption Checks 📊")

# File upload
uploaded_file = st.file_uploader("Upload CSV file, no space in variable name", type=["csv"])

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

        # Train-test split
        test_size = st.slider("Test Size", 0.01, 0.5, 0.2)
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

            # Assumption Checks
            st.subheader("Assumption Checks")
            residuals = model.resid
            fitted_values = model.predict(train_df)

            # 1. Linearity: Actual vs. Predicted Plot
            st.write("**1. Linearity Check: Actual vs. Predicted**")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=fitted_values, y=train_df[target], ax=ax1)
            ax1.set_xlabel("Predicted Values")
            ax1.set_ylabel("Actual Values")
            st.pyplot(fig1)

            # 2. Homoscedasticity: Residuals vs. Predicted Plot
            st.write("**2. Homoscedasticity Check: Residuals vs. Predicted**")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=fitted_values, y=residuals, ax=ax2)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel("Predicted Values")
            ax2.set_ylabel("Residuals")
            st.pyplot(fig2)

            # Goldfeld-Quandt Test
            _, gq_p, _ = het_goldfeldquandt(residuals, model.model.exog)
            st.write(f"**Goldfeld-Quandt Test p-value:** {gq_p:.4f}")
            st.write("✅ Homoscedasticity holds (constant variance)" if gq_p > 0.05 
                    else "❌ Heteroscedasticity detected!")

            # 3. Normality of Residuals
            st.write("**3. Normality Check**")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sm.qqplot(residuals, line='s', ax=ax3)
            st.pyplot(fig3)

            shapiro_stat, shapiro_p = shapiro(residuals)
            st.write(f"**Shapiro-Wilk Test p-value:** {shapiro_p:.4f}")
            st.write("✅ Residuals are normal" if shapiro_p > 0.05 
                    else "❌ Residuals are NOT normal!")

            # 4. Multicollinearity
            st.write("**4. Multicollinearity Check**")
            X = pd.get_dummies(train_df[features], drop_first=True)
            X = sm.add_constant(X)
            vif_data = pd.DataFrame({
                "Feature": X.columns,
                "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            })
            st.dataframe(vif_data.style.format({"VIF": "{:.2f}"}))
            st.write("✅ VIF < 10: No multicollinearity" if all(vif_data["VIF"] < 10) 
                    else "❌ VIF ≥ 10: Severe multicollinearity detected!")

            # 5. Independence (Durbin-Watson)
            dw = durbin_watson(residuals)
            st.write(f"**Durbin-Watson Statistic:** {dw:.2f}")
            st.write("✅ No autocorrelation (1.5 < DW < 2.5)" if 1.5 < dw < 2.5 
                    else "❌ Autocorrelation detected!")

else:
    st.info("Please upload a CSV file to begin")

# Instructions
st.sidebar.markdown("""
**Assumption Checks Included:**
1. Linearity (Actual vs Predicted plot)
2. Homoscedasticity (Residual plot + Goldfeld-Quandt test)
3. Normality (Q-Q plot + Shapiro-Wilk test)
4. Multicollinearity (VIF)
5. Independence (Durbin-Watson)
""")
