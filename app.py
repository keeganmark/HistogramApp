import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ne 111 App", layout="wide")

# ----------------------------
# Utility functions (omitted for brevity, assume they are correct)
# ----------------------------
def safe_parse_numbers(text: str):
    tokens = text.replace(",", " ").replace("\n", " ").replace("\t", " ").split()
    vals = []
    for t in tokens:
        try:
            vals.append(float(t))
        except ValueError:
            continue
    return np.array(vals)

def param_names(dist):
    shapes = getattr(dist, "shapes", None)
    names = [s.strip() for s in shapes.split(",")] if shapes else []
    names += ["loc", "scale"]
    return names

def fit_distribution(dist, data):
    try:
        params = tuple(float(p) for p in dist.fit(data))
        return params, None
    except Exception as e:
        return None, str(e)

def pdf_from_params(dist, x, params):
    if params is None:
        return np.zeros_like(x)
    *shape_params, loc, scale = params if len(params) >= 2 else ([], 0, 1)
    try:
        return dist.pdf(x, *shape_params, loc=loc, scale=scale)
    except Exception:
        return np.zeros_like(x)

def histogram_data(data, bins, density=True, clip=True):
    if clip:
        lo, hi = np.percentile(data, [2.5, 97.5])
        data = data[(data >= lo) & (data <= hi)]
    # Note: density is False here for consistency, will apply scale factor later
    counts, edges = np.histogram(data, bins=bins, density=density) 
    centers = (edges[:-1] + edges[1:]) / 2
    return counts, centers, edges, data

# ----------------------------
# Distributions (omitted for brevity, assume they are correct)
# ----------------------------
DIST_DICT = {
    "Normal": stats.norm, "Log-Normal": stats.lognorm, "Gamma": stats.gamma, 
    "Beta": stats.beta, "Weibull Min": stats.weibull_min, "Weibull Max": stats.weibull_max, 
    "Exponential": stats.expon, "Pareto": stats.pareto, "Chi-Squared": stats.chi2, 
    "Student-t": stats.t, "Cauchy": stats.cauchy, "Laplace": stats.laplace, 
    "Rayleigh": stats.rayleigh, "Uniform": stats.uniform,
}

# ----------------------------
# Data Input (omitted for brevity, assumes previous fixes)
# ----------------------------
st.title("📈 Fitting Histograms to Statistical Data")

if "data" not in st.session_state:
    st.session_state.data = None

st.header("Load Data")
mode = st.selectbox("Input mode", ["Paste", "Upload CSV", "Generate sample"])

if mode == "Paste":
    raw = st.text_area("Paste numbers", height=180)
    if st.button("Load"):
        arr = safe_parse_numbers(raw)
        if len(arr) == 0:
            st.error("No numeric values found.")
        else:
            st.session_state.data = arr
            st.success(f"Loaded {len(arr)} values.")

elif mode == "Upload CSV":
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            st.error("CSV contains no numeric columns.")
        else:
            col = st.selectbox("Select column", numeric_cols)
            if st.button("Load column"):
                data = df[col].dropna().values
                st.session_state.data = data
                st.success(f"Loaded {len(data)} values.")

elif mode == "Generate sample":
    n = st.number_input("Sample size", min_value=10, max_value=20000, value=2000)

    if st.button("Generate"):
        d = np.concatenate([
               np.random.normal(0, 1, int(n * 0.75)),
               np.random.normal(4, 1.2, int(n * 0.25))
            ])
        st.session_state.data = d
        st.success(f"Generated {n} samples.")
# Stop if no data
data = st.session_state.data
if data is None:
    st.info("Please load data to proceed.")
    st.stop()
    
# Initial clip of data BEFORE sidebar to calculate intelligent bin default
lo, hi = np.percentile(data, [2.5, 97.5])
initial_clipped = data[(data >= lo) & (data <= hi)]

# --- NEW: Calculate intelligent default bin count ---
try:
    # Use 'auto' method (which uses Freedman-Diaconis or Sturges) to get a good starting bin count
    _, suggested_bins = np.histogram(initial_clipped, bins='auto')
    default_bins = len(suggested_bins) - 1 # np.histogram returns edges, need to subtract 1 for bin count
    # Clamp suggested bins between 10 and 120
    default_bins = max(10, min(120, default_bins))
except:
    default_bins = 40 # Fallback default
# -----------------------------------------------------

# ----------------------------
# Sidebar Options
# ----------------------------
with st.sidebar:
    st.header("Settings")
    
    # Use number_input with the calculated default_bins
    bins = st.number_input(
        "Histogram bins", 
        min_value=10, 
        max_value=120, 
        value=default_bins, # <--- Uses the calculated default
        step=5,
        help="Manually set the number of bars, or use the calculated default for optimal visualization."
    )
    bins = int(bins) # Ensure it's an integer
    
    density = st.checkbox("Normalize histogram", True, help="If unchecked, y-axis shows raw counts.")
    clip = st.checkbox("Clip outliers (2.5–97.5%)", True)
    compare_all = st.checkbox("Compare all distributions", False)

# ----------------------------
# Main Logic
# ----------------------------
counts, centers, edges, clipped = histogram_data(data, bins, density, clip)

# CALCULATE SCALE FACTOR and MAX HIST HEIGHT (from previous fixes)
bin_width = edges[1] - edges[0]
scale_factor = 1.0 if density else len(clipped) * bin_width
max_hist_height = np.max(counts)

# Tabs
tab1, tab2 = st.tabs(["Fit Distribution", "Compare Distributions"])

# =========================
# TAB 1: FITTING + METRICS
# =========================
fitted_params = None # ... (rest of the code follows the logic from the previous correct version)

with tab1:
    col1, col2 = st.columns([2, 3])

    # Auto-fit best distribution if requested (logic omitted for brevity)
    best_fit_name, best_fit_params, best_fit_metrics = None, None, None
    
    if st.button("Auto-fit best distribution"):
        # ... (Auto-fit loop using scale_factor and KS test)
        for name, dist in DIST_DICT.items():
            params, err = fit_distribution(dist, clipped)
            if params is None: continue
            
            pdf_est = pdf_from_params(dist, centers, params) * scale_factor
            mae = float(np.mean(np.abs(counts - pdf_est)))
            
            if best_fit_name is None or mae < best_fit_metrics["MAE"]:
                # ... update best fit metrics ...
                rmse = float(np.sqrt(np.mean((counts - pdf_est)**2)))
                try:
                    args = params[:-2] if len(params) > 2 else ()
                    loc, scale = params[-2], params[-1]
                    ks_stat, ks_p = stats.kstest(clipped, lambda x: dist.cdf(x, *args, loc=loc, scale=scale))
                except:
                    ks_stat, ks_p = np.nan, np.nan
                
                best_fit_name = name
                best_fit_params = params
                best_fit_metrics = {"MAE": mae, "RMSE": rmse, "KS stat": ks_stat, "KS p-value": ks_p}


        if best_fit_name:
            st.success(f"Best fit: {best_fit_name} (MAE={best_fit_metrics['MAE']:.5g})")
            fitted_params = best_fit_params
        else:
            st.warning("No successful fits.")

    # Dropdown to select distribution manually or view result of auto-fit
    with col1:
        st.subheader("Choose distribution")
        current_index = list(DIST_DICT.keys()).index(best_fit_name) if best_fit_name else 0
        dist_name = st.selectbox("Distribution", list(DIST_DICT.keys()), index=current_index)
        dist = DIST_DICT[dist_name]

        fit_mode = st.radio("Fitting mode", ["Automatic", "Manual"])
        fit_btn = st.button("Fit / Apply manual parameters")
        
        # ... Manual fitting logic ...
        if fit_mode == "Manual":
            st.subheader("Manual parameter tuning")
            default_params, _ = fit_distribution(dist, clipped)
            if not default_params: default_params = (0.0, 1.0)
            names = param_names(dist)
            
            param_vals = []
            cols = st.columns(2)
            for i, (pname, default) in enumerate(zip(names, default_params)):
                with cols[i % 2]:
                    # Dynamic slider range based on data stats
                    if pname == "loc":
                        lo, hi = np.min(clipped), np.max(clipped)
                        step = (hi-lo)/100
                    elif pname == "scale":
                        lo, hi = 1e-4, np.std(clipped)*3
                        step = (hi-lo)/100
                    else:
                        lo = default - abs(default)*5 - 2
                        hi = default + abs(default)*5 + 2
                        step = 0.01
                    
                    val = st.slider(pname, float(lo), float(hi), float(default), step=step)
                    param_vals.append(val)

            if fit_btn:
                fitted_params = tuple(param_vals)
        elif fit_mode == "Automatic" and fit_btn:
            fitted_params, _ = fit_distribution(dist, clipped)


    # Plotting (includes Y-axis fix)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Histogram
        ax.hist(clipped, bins=bins, alpha=0.35, color="#2b6ea3", edgecolor='white', density=density)
        
        # FIX: Set dynamic Y-limit for non-normalized graphs
        if not density and max_hist_height > 0:
            ax.set_ylim(0, max_hist_height * 1.05)
        
        if fitted_params:
            x = np.linspace(np.min(clipped), np.max(clipped), 500)
            pdf_vals = pdf_from_params(dist, x, fitted_params) * scale_factor
            
            ax.plot(x, pdf_vals, 'r-', lw=2, label=f"{dist_name} fit")
            ax.legend()
            
        ax.set_ylabel("Probability Density" if density else "Frequency (Count)")
        st.pyplot(fig)

        # ... Metrics calculation ...
        if fitted_params:
            pdf_on_bins = pdf_from_params(dist, centers, fitted_params) * scale_factor
            
            mae = float(np.mean(np.abs(counts - pdf_on_bins)))
            rmse = float(np.sqrt(np.mean((counts - pdf_on_bins) ** 2)))
            # ... KS test logic (omitted) ...
            try:
                args = fitted_params[:-2] if len(fitted_params) > 2 else ()
                loc, scale = fitted_params[-2], fitted_params[-1]
                ks_stat, ks_p = stats.kstest(clipped, lambda x: dist.cdf(x, *args, loc=loc, scale=scale))
            except:
                ks_stat, ks_p = np.nan, np.nan
            
            st.markdown("**Fit metrics:**")
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("MAE", f"{mae:.4g}")
            col_m2.metric("RMSE", f"{rmse:.4g}")
            st.caption("Lower MAE/RMSE is better.")
    with tab2:
        if not compare_all:
            st.info("Enable 'Compare all distributions' in the sidebar.")
        else:
            st.write("Fitting all distributions...")
            table = []
            progress_bar = st.progress(0)
            dist_items = list(DIST_DICT.items())
        
            for i, (name, dist) in enumerate(dist_items):
                params, err = fit_distribution(dist, clipped)
                if params is None: continue
            
            # Apply scale factor for metric calculation
                pdf_est = pdf_from_params(dist, centers, params) * scale_factor
            
                mae = float(np.mean(np.abs(counts - pdf_est)))
                rmse = float(np.sqrt(np.mean((counts - pdf_est)**2)))
            
            # KS Test (Always uses density logic internally, so we use CDF)
                try:
                # Helper for CDF args
                    args = params[:-2] if len(params) > 2 else ()
                    loc, scale = params[-2], params[-1]
                    ks_stat, ks_p = stats.kstest(clipped, lambda x: dist.cdf(x, *args, loc=loc, scale=scale))
                except:
                    ks_stat, ks_p = np.nan, np.nan
                
                table.append({
                    "Distribution": name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "KS p-value": ks_p,
                    "Parameters": str(np.round(params, 3))
                })
                progress_bar.progress((i + 1) / len(dist_items))

            df_results = pd.DataFrame(table)
            if not df_results.empty:
                df_results.sort_values("MAE", inplace=True)
                st.subheader("Distribution Fit Comparison (sorted by MAE)")
                st.dataframe(df_results.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "KS p-value": "{:.4f}"}), use_container_width=True)
            else:
                st.error("Could not fit any distributions.")