import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Distribution Fitting App", layout="wide")

# ----------------------------
# Utility functions
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
        # Fit distribution using MLE
        params = tuple(float(p) for p in dist.fit(data))
        return params, None
    except Exception as e:
        return None, str(e)

def pdf_from_params(dist, x, params):
    if params is None:
        return np.zeros_like(x)
    *shape_params, loc, scale = params if len(params) >= 2 else ([], 0, 1)
    try:
        # Avoid division by zero in scale
        scale = max(1e-9, scale) 
        return dist.pdf(x, *shape_params, loc=loc, scale=scale)
    except Exception:
        return np.zeros_like(x)

def histogram_data(data, bins, density=True, clip=True):
    if clip:
        # Clip to 2.5% and 97.5% percentiles
        lo, hi = np.percentile(data, [2.5, 97.5])
        data = data[(data >= lo) & (data <= hi)]
    # Use density=False to get raw counts, which we handle manually for scaling consistency
    counts, edges = np.histogram(data, bins=bins, density=density) 
    centers = (edges[:-1] + edges[1:]) / 2
    return counts, centers, edges, data

# --- Freedman-Diaconis Bin Calculation ---
def calculate_fd_bins(data):
    """Automatically calculates the optimal number of bins using the Freedman-Diaconis rule."""
    N = len(data)
    if N < 2:
        return 10
    
    # Calculate Interquartile Range (IQR)
    q1, q3 = np.percentile(data, [25, 75])
    IQR = q3 - q1
    
    data_range = np.max(data) - np.min(data)
    
    # Calculate bin width (h)
    if IQR == 0:
        # Fallback to Sturges' rule if all values are the same
        bin_width = data_range / (1 + np.log2(N)) if data_range > 0 else 1.0
    else:
        # Freedman-Diaconis formula for bin width
        bin_width = 2 * IQR / (N ** (1/3))

    if bin_width <= 1e-9 or data_range <= 1e-9:
        return 10
    
    # Calculate number of bins
    num_bins = int(np.ceil(data_range / bin_width))
    return max(10, num_bins)

# ----------------------------
# Distributions
# ----------------------------
DIST_DICT = {
    "Normal": stats.norm,
    "Log-Normal": stats.lognorm,
    "Gamma": stats.gamma,
    "Beta": stats.beta,
    "Weibull Min": stats.weibull_min,
    "Weibull Max": stats.weibull_max,
    "Exponential": stats.expon,
    "Pareto": stats.pareto,
    "Chi-Squared": stats.chi2,
    "Student-t": stats.t,
    "Cauchy": stats.cauchy,
    "Laplace": stats.laplace,
    "Rayleigh": stats.rayleigh,
    "Uniform": stats.uniform,
}

# ----------------------------
# Data Input & State Initialization
# ----------------------------

# 1. Custom Header Container (Aesthetic Improvement)
header_color = "#173e67" # Darker blue than histogram bars (#1f4e79)
st.markdown(
    f"""
    <style>
    .header-bar {{
        background-color: {header_color};
        padding: 15px 0 15px 0;
        margin: -20px -20px 20px -20px; /* Adjust margins to span wide */
        border-radius: 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .header-bar h1 {{
        color: white !important;
        padding-left: 20px; /* Align with content */
        margin: 0;
        font-size: 2em;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.markdown('<div class="header-bar"><h1>📈 Smart Distribution Fitting App</h1></div>', unsafe_allow_html=True)
    
# Removed the original st.title and replaced it with the custom container above

if "data" not in st.session_state:
    st.session_state.data = None
    
if "smart_bins" not in st.session_state:
    st.session_state.smart_bins = 50 

if "bin_key_counter" not in st.session_state:
    st.session_state.bin_key_counter = 0 

st.subheader("Load Data")
st.markdown("---") # Aesthetic separator

mode = st.selectbox("Input mode", ["Paste", "Upload CSV"])

if mode == "Paste":
    raw = st.text_area("Paste numbers", height=180)
    if st.button("Load"):
        arr = safe_parse_numbers(raw)
        if len(arr) == 0:
            st.error("No numeric values found.")
        else:
            st.session_state.data = arr
            st.success(f"Loaded {len(arr)} values.")
            
            # --- DYNAMIC BIN CALCULATION (Freedman-Diaconis Rule) ---
            st.session_state.smart_bins = calculate_fd_bins(st.session_state.data) 
            st.session_state.bin_key_counter += 1
            
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
                
                # --- DYNAMIC BIN CALCULATION (Freedman-Diaconis Rule) ---
                st.session_state.smart_bins = calculate_fd_bins(st.session_state.data)
                st.session_state.bin_key_counter += 1
                
# Stop if no data
data = st.session_state.data
if data is None:
    st.info("Please load data to proceed.")
    st.stop()
# Sidebar Options
with st.sidebar:
    st.header("Settings")
    st.info("**Check both boxes** below for the most accurate parameter estimates and visually clear fit for the central data.")
    bins = st.number_input(
        "Histogram bins", 
        min_value=10, 
        max_value=500, 
        value=st.session_state.smart_bins,
        key=f"dynamic_bins_{st.session_state.bin_key_counter}", 
        step=5,
        help="Manually set the number of bars. Automatically calculated using the Freedman-Diaconis rule."
    )
    bins = int(bins) 
    
    density = st.checkbox("Normalize histogram (Density)", True, help="If checked, y-axis shows probability density. This is required for a true PDF comparison.") 
    clip = st.checkbox("Clip outliers (2.5–97.5%)", True, help="Filter out extreme values before fitting and plotting for a better fit.") 

# Main Logic & Tabs
counts, centers, edges, clipped = histogram_data(data, bins, density, clip)

bin_width = edges[1] - edges[0]
# Use density=True as default now, so the scale factor logic must be updated
scale_factor = 1.0 if density else len(clipped) * bin_width

if 'mode_params' not in st.session_state:
    st.session_state.mode_params = {} 


tab1, tab2 = st.tabs(["Fit Distribution", "Compare Distributions"])

# TAB 1: FITTING + METRICS

fitted_params = None
best_fit_name, best_fit_params, best_fit_metrics = None, None, None

with tab1:
    col1, col2 = st.columns([2, 3])

    if st.button("Auto-fit best distribution"):
        best_mae = float('inf')
        
        for name, dist in DIST_DICT.items():
            params, err = fit_distribution(dist, clipped)
            if params is None: continue
            
            pdf_est = pdf_from_params(dist, centers, params) * scale_factor
            mae = float(np.mean(np.abs(counts - pdf_est)))
            
            if mae < best_mae:
                # RMSE and KS p-value are calculated internally for MAE comparison
                rmse = float(np.sqrt(np.mean((counts - pdf_est)**2)))
                try:
                    args = params[:-2] if len(params)>2 else ()
                    loc, scale = params[-2], params[-1]
                    ks_stat, ks_p = stats.kstest(clipped, lambda x: dist.cdf(x, *args, loc=loc, scale=scale))
                except:
                    ks_stat, ks_p = np.nan, np.nan
                    
                best_mae = mae
                best_fit_name = name
                best_fit_params = params
                best_fit_metrics = {"MAE": mae, "RMSE": rmse, "KS stat": ks_stat, "KS p-value": ks_p}

        if best_fit_name:
            st.success(f"Best fit: {best_fit_name} (MAE={best_fit_metrics['MAE']:.5g})")
            st.session_state.mode_params[(best_fit_name, "Automatic")] = best_fit_params
        else:
            st.warning("No successful fits.")

    with col1:
        st.subheader("Choose distribution")
        current_index = list(DIST_DICT.keys()).index(best_fit_name) if best_fit_name in DIST_DICT else 0
        dist_name = st.selectbox("Distribution", list(DIST_DICT.keys()), index=current_index)
        dist = DIST_DICT[dist_name]

        fit_mode = st.radio("Fitting mode", ["Automatic", "Manual"])
        fit_btn = st.button("Calculate Automatic Fit")
        
        current_auto_key = (dist_name, "Automatic")
        current_manual_key = (dist_name, "Manual")
        
        fitted_params_to_use = None
        names = param_names(dist)
        
        # --- MANUAL MODE LOGIC ---
        if fit_mode == "Manual":
            st.subheader("Manual parameter tuning")
            
            if current_manual_key not in st.session_state.mode_params:
                initial_params, _ = fit_distribution(dist, clipped)
                if not initial_params: initial_params = tuple([1.0] * len(names))
                st.session_state.mode_params[current_manual_key] = initial_params
            
            initial_params = st.session_state.mode_params[current_manual_key]

            param_vals = []
            cols = st.columns(2)
            
            for i, pname in enumerate(names):
                default_val = initial_params[i] if len(initial_params) > i else 1.0 # Ensure a fallback default
                initial_value = default_val 

                with cols[i % 2]:
                    # Dynamic slider range logic
                    if pname == "loc":
                        lo, hi = np.min(clipped) - 0.5 * np.ptp(clipped), np.max(clipped) + 0.5 * np.ptp(clipped)
                        step = (hi-lo)/100 or 0.01
                    elif pname == "scale":
                        std_clipped = np.std(clipped)
                        lo = 1e-9
                        hi = std_clipped * 5
                        if initial_value > hi: initial_value = std_clipped * 2 
                        step = (hi-lo)/100 or 0.01
                    else: # Shape parameters
                        # Ensure lo is not above hi if default is small/near zero
                        lo = initial_value - 5 
                        hi = initial_value + 5
                        lo, hi = min(lo, 0.01), max(hi, 5.0) # Ensure positive shapes are not negative
                        lo = min(lo, initial_value)
                        hi = max(hi, initial_value)
                        step = 0.01
                        
                    val = st.slider(pname, float(lo), float(hi), float(initial_value), step=step, key=f"manual_{dist_name}_{pname}")
                    param_vals.append(val)
            
            fitted_params_to_use = tuple(param_vals)
            st.session_state.mode_params[current_manual_key] = fitted_params_to_use

        # --- AUTOMATIC MODE LOGIC ---
        elif fit_mode == "Automatic":
            if fit_btn:
                auto_params, _ = fit_distribution(dist, clipped)
                if auto_params:
                    st.session_state.mode_params[current_auto_key] = auto_params
            
            if current_auto_key in st.session_state.mode_params:
                fitted_params_to_use = st.session_state.mode_params[current_auto_key]
            elif best_fit_params and dist_name == best_fit_name:
                 fitted_params_to_use = best_fit_params 

        fitted_params = fitted_params_to_use
        
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # --- STYLE CHANGE: Darker Blue and Black Outline ---
        ax.hist(clipped, bins=bins, alpha=0.7, color="#1f4e79", edgecolor='black', density=density)
        
        max_hist_height = np.max(counts) if len(counts)>0 else 0
        max_plot_height = max_hist_height
        
        if fitted_params:
            x = np.linspace(np.min(clipped), np.max(clipped), 500)
            pdf_vals = pdf_from_params(dist, x, fitted_params) * scale_factor
            
            max_pdf_height = np.max(pdf_vals) if len(pdf_vals) > 0 else 0
            max_plot_height = max(max_hist_height, max_pdf_height)

            ax.plot(x, pdf_vals, 'r-', lw=2, label=f"{dist_name} fit")
            ax.legend()
            
        if max_plot_height > 0:
            ax.set_ylim(0, max_plot_height * 1.05)
            
        ax.set_xlabel("Data Range Values") 
        ax.set_ylabel("Probability Density" if density else "Frequency (Count)")
        st.pyplot(fig)

        # Show metrics
        if fitted_params:
            pdf_on_bins = pdf_from_params(dist, centers, fitted_params) * scale_factor
            
            mae = float(np.mean(np.abs(counts - pdf_on_bins)))
            
            try:
                args = fitted_params[:-2] if len(fitted_params) > 2 else ()
                loc, scale = fitted_params[-2], fitted_params[-1]
                ks_stat, ks_p = stats.kstest(clipped, lambda x: dist.cdf(x, *args, loc=loc, scale=scale))
            except:
                ks_stat, ks_p = np.nan, np.nan
            
            st.markdown("**Fit metrics:**")
            # --- REMOVE RMSE AND KS P-VALUE ---
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("MAE (Avg Error)", f"{mae:.4g}")
            col_m2.metric("KS Stat (Max Diff)", f"{ks_stat:.4g}")
            st.caption("MAE measures average height difference. KS Stat measures maximum distribution difference.")

# ===========================
# TAB 2: COMPARE DISTRIBUTIONS (TABLE)
# ===========================
with tab2:
    st.write("Fitting all distributions...")
    table = []
    progress_bar = st.progress(0)
    dist_items = list(DIST_DICT.items())
        
    for i, (name, dist) in enumerate(dist_items):

        params, err = fit_distribution(dist, clipped)
        if params is None:
            continue
            
        pdf_est = pdf_from_params(dist, centers, params) * scale_factor

        # --- Calculate all metrics for the comparison table ---
        mae = float(np.mean(np.abs(counts - pdf_est)))
        rmse = float(np.sqrt(np.mean((counts - pdf_est)**2)))
            
        try:
            args = params[:-2] if len(params) > 2 else ()
            loc, scale = params[-2], params[-1]
            ks_stat, ks_p = stats.kstest(
                clipped,
                lambda x: dist.cdf(x, *args, loc=loc, scale=scale)
            )
        except:
            ks_stat, ks_p = np.nan, np.nan
                
        table.append({
            "Distribution": name,
            "MAE": mae,
            "RMSE": rmse,
            "KS stat": ks_stat,
            "KS p-value": ks_p,
            "Parameters": str(np.round(params, 3))
        })

        progress_bar.progress((i + 1) / len(dist_items))

    # --- Display results AFTER loop ---
    df_results = pd.DataFrame(table)

    if not df_results.empty:
        df_results.sort_values("MAE", inplace=True)
        st.subheader("Distribution Fit Comparison (sorted by MAE)")

        st.dataframe(
            df_results.style.format({
                "MAE": "{:.4f}",
                "RMSE": "{:.4f}",
                "KS p-value": "{:.4f}",
            }),
            use_container_width=True
        )
    else:
        st.error("Could not fit any distributions.")
