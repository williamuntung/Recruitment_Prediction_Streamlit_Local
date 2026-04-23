import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
from scipy import stats


st.set_page_config(
    page_title="PlaceIQ · Placement & Salary AI",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def load_models():
    models = {}
    try:
        models['placement'] = joblib.load('artifacts/placement_status_prediction.pkl')
    except Exception as e:
        models['placement'] = None
        models['placement_error'] = str(e)
    try:
        models['salary'] = joblib.load('artifacts/salary_prediction.pkl')
    except Exception as e:
        models['salary'] = None
        models['salary_error'] = str(e)
    return models


def load_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

        html, body { background: #0d0f1a !important; }

        .stApp,
        .stApp > div,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"],
        .main,
        .main > div {
            background: #0d0f1a !important;
            color: #e8eaf6 !important;
            font-family: 'DM Sans', sans-serif !important;
        }

        section[data-testid="stMain"] > div { background: #0d0f1a !important; }

        h1 {
            font-family: 'Syne', sans-serif !important;
            font-weight: 800 !important;
            font-size: 2.4rem !important;
            background: linear-gradient(135deg, #818cf8, #c084fc, #38bdf8);
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            text-align: center;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem !important;
        }
        h2, h3, h4 {
            font-family: 'Syne', sans-serif !important;
            color: #c7d2fe !important;
            background: none !important;
            -webkit-text-fill-color: #c7d2fe !important;
        }
        p, li, label { color: #cbd5e1; font-family: 'DM Sans', sans-serif; }

        .stMarkdown p,
        [data-testid="stMarkdownContainer"] p {
            color: #cbd5e1 !important;
            background: transparent !important;
        }
        [data-testid="stMarkdownContainer"] h4 {
            color: #c7d2fe !important;
            -webkit-text-fill-color: #c7d2fe !important;
            background: transparent !important;
        }

        .block-container {
            padding: 2.5rem 3rem 3rem;
            max-width: 960px;
            background: transparent !important;
        }

        /* ════════════════════
           SIDEBAR
        ════════════════════ */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div {
            background: #13162b !important;
            border-right: 1px solid rgba(99,102,241,0.2);
        }

        /* Section labels (bold markdown text) */
        [data-testid="stSidebar"] strong {
            color: #a5b4fc !important;
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }

        /* Widget labels */
        [data-testid="stSidebar"] label {
            color: #7c8db5 !important;
            font-size: 0.78rem !important;
            font-weight: 500 !important;
            background: transparent !important;
        }

        /* ── Radio option text — always white ──
           Targets every layer Streamlit might use */
        [data-testid="stSidebar"] [data-testid="stRadio"] label p,
        [data-testid="stSidebar"] [data-testid="stRadio"] label span,
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radio"] p,
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radio"] span,
        [data-testid="stSidebar"] [data-baseweb="radio"] p,
        [data-testid="stSidebar"] [data-baseweb="radio"] span,
        [data-testid="stSidebar"] div[role="radiogroup"] p,
        [data-testid="stSidebar"] div[role="radiogroup"] span {
            color: #f1f5f9 !important;
            -webkit-text-fill-color: #f1f5f9 !important;
            font-size: 0.85rem !important;
            font-weight: 400 !important;
            text-transform: none !important;
            letter-spacing: 0 !important;
            background: transparent !important;
        }

        /* Radio checked dot */
        [data-baseweb="radio"] [data-checked="true"] div { background: #818cf8 !important; }

        /* Sidebar headings */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {
            color: #818cf8 !important;
            -webkit-text-fill-color: #818cf8 !important;
            font-family: 'Syne', sans-serif !important;
            background: transparent !important;
        }

        /* Sidebar number inputs */
        [data-testid="stSidebar"] input[type="number"],
        [data-testid="stSidebar"] [data-baseweb="input"] input {
            background: #1a1f3a !important;
            color: #e8eaf6 !important;
            border: 1px solid rgba(99,102,241,0.2) !important;
            border-radius: 8px !important;
        }

        /* Sidebar selectbox */
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #1a1f3a !important;
            color: #e8eaf6 !important;
            border-color: rgba(99,102,241,0.2) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="popover"],
        [data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="menu"] {
            background: #1a1f3a !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] li,
        [data-testid="stSidebar"] [data-baseweb="select"] span {
            color: #e8eaf6 !important;
        }

        /* Mode selector radio (top of sidebar — also needs white text) */
        [data-testid="stSidebar"] > div:first-child [data-testid="stRadio"] label p,
        [data-testid="stSidebar"] > div:first-child [data-testid="stRadio"] label span {
            color: #f1f5f9 !important;
            -webkit-text-fill-color: #f1f5f9 !important;
        }

        /* ════════════════════
           MAIN WIDGETS
        ════════════════════ */
        input[type="number"], input[type="text"],
        [data-baseweb="input"] input,
        [data-baseweb="base-input"] input {
            background: #1a1f3a !important;
            color: #e8eaf6 !important;
            border: 1px solid rgba(99,102,241,0.2) !important;
            border-radius: 8px !important;
        }
        [data-baseweb="select"] > div { background: #1a1f3a !important; color: #e8eaf6 !important; }

        /* ════════════════════
           LAYOUT HELPERS
        ════════════════════ */
        .tagline {
            text-align: center;
            color: #64748b !important;
            font-size: 0.9rem;
            margin-bottom: 2.5rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            background: transparent !important;
        }

        .section-pill {
            display: inline-block;
            background: rgba(99,102,241,0.15);
            border: 1px solid rgba(99,102,241,0.35);
            border-radius: 999px;
            padding: 4px 16px;
            font-size: 0.78rem;
            color: #818cf8 !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 1.2rem;
        }

        .fancy-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
            margin: 2rem 0;
        }

        /* ════════════════════
           RESULT CARDS
        ════════════════════ */
        .verdict-placed {
            display: inline-flex; align-items: center; gap: 10px;
            background: rgba(52,211,153,0.12);
            border: 1px solid rgba(52,211,153,0.4);
            color: #34d399 !important;
            font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 700;
            border-radius: 14px; padding: 14px 28px;
        }
        .verdict-not-placed {
            display: inline-flex; align-items: center; gap: 10px;
            background: rgba(248,113,113,0.1);
            border: 1px solid rgba(248,113,113,0.35);
            color: #f87171 !important;
            font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 700;
            border-radius: 14px; padding: 14px 28px;
        }
        .verdict-salary {
            display: inline-flex; align-items: baseline; gap: 6px;
            color: #facc15 !important;
            font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800;
        }
        .verdict-salary span {
            font-size: 1.1rem; color: #94a3b8 !important;
            font-family: 'DM Sans', sans-serif; font-weight: 400;
        }

        .stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 1.4rem; }
        .stat-chip {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px; padding: 7px 14px;
            font-size: 0.76rem; color: #94a3b8 !important;
        }
        .stat-chip b { color: #c7d2fe !important; font-weight: 600; }

        /* Model status chips */
        .model-chip {
            display: flex; align-items: center; gap: 8px;
            padding: 6px 12px;
            background: #1a1f3a;
            border: 1px solid rgba(99,102,241,0.2);
            border-radius: 9px;
            margin-bottom: 6px;
        }
        .model-chip-label {
            font-size: 0.72rem; font-weight: 600; color: #94a3b8;
            text-transform: uppercase; letter-spacing: 0.06em;
        }
        .model-chip-status { font-size: 0.68rem; }

        /* ════════════════════
           BUTTON
        ════════════════════ */
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px;
            padding: 0.75em 1.5em;
            font-family: 'Syne', sans-serif !important;
            font-weight: 700 !important;
            font-size: 1rem;
            letter-spacing: 0.03em;
            box-shadow: 0 4px 20px rgba(99,102,241,0.3);
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #6d28d9, #4f46e5) !important;
            box-shadow: 0 6px 28px rgba(99,102,241,0.45);
            transform: translateY(-1px);
        }

        /* ════════════════════
           MISC
        ════════════════════ */
        .stSpinner > div { border-top-color: #818cf8 !important; }
        #MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(145deg, #1a1f3a, #141829) !important;
            border-radius: 18px !important;
        }
        </style>
    """, unsafe_allow_html=True)


def plot_placement_gauge(confidence, label):
    fig, ax = plt.subplots(figsize=(3.8, 2.4), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#141829")
    ax.set_facecolor("#141829")

    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#1e2540", linewidth=22, solid_capstyle="round")

    fill_theta = np.linspace(np.pi, np.pi - (np.pi * confidence), 300)
    color = "#34d399" if str(label).lower() in ["placed", "1", "yes"] else "#f87171"
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), color=color, linewidth=22,
            solid_capstyle="round", alpha=0.9)
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), color="white", linewidth=3,
            solid_capstyle="round", alpha=0.15)

    ax.text(0, 0.05, f"{int(confidence*100)}%", ha="center", va="center",
            fontsize=28, fontweight="bold", color="white", fontfamily="DejaVu Sans")
    ax.text(0, -0.28, "Confidence", ha="center", va="center", fontsize=9, color="#64748b")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, 1.3)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def plot_salary_distribution(salary):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_facecolor("#141829")
    ax.set_facecolor("#141829")

    mu, sigma = np.log(6.5), 0.55
    dist = stats.lognorm(s=sigma, scale=np.exp(mu))
    x = np.linspace(2, 30, 500)
    y = dist.pdf(x)

    salary_clamped = max(2.5, min(salary, 28.0))
    percentile = dist.cdf(salary_clamped) * 100

    x_left = x[x <= salary_clamped]
    ax.fill_between(x_left, dist.pdf(x_left), alpha=0.18, color="#facc15")
    ax.fill_between(x_left, dist.pdf(x_left), alpha=0.22, color="#f59e0b",
                    where=x_left >= salary_clamped * 0.7)
    x_right = x[x >= salary_clamped]
    ax.fill_between(x_right, dist.pdf(x_right), alpha=0.07, color="#94a3b8")
    ax.plot(x, y, color="#fbbf24", linewidth=2.2, zorder=4)

    needle_h = dist.pdf(salary_clamped)
    ax.vlines(salary_clamped, 0, needle_h * 1.08,
              colors="#facc15", linewidth=2, linestyle="--", zorder=5, alpha=0.9)
    ax.scatter([salary_clamped], [needle_h * 1.08], s=70,
               color="#facc15", zorder=6, edgecolors="white", linewidths=1.2)
    ax.text(salary_clamped, needle_h * 1.22, f"${salary_clamped:.1f} LPA",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#facc15", zorder=7)
    ax.text(salary_clamped, needle_h * 1.5, f"Top {100 - int(percentile)}%",
            ha="center", va="bottom", fontsize=8, color="#94a3b8",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1e2540",
                      ec=(250/255, 204/255, 21/255, 0.3), lw=0.8))

    ticks = [3, 6, 9, 12, 15, 20, 25]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"${t}L" for t in ticks], color="#475569", fontsize=7.5)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xlim(2, 28)
    ax.set_ylim(0, max(y) * 2.1)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#1e2540")
    ax.set_yticks([])
    ax.set_xlabel("Annual Package (LPA)", color="#475569", fontsize=8, labelpad=6)
    fig.tight_layout(pad=0.6)
    return fig, int(percentile)


def plot_skill_radar(data):
    categories = ["Coding", "Communication", "Aptitude", "Attendance", "CGPA"]
    values = [
        data["coding_skill_rating"] / 5,
        data["communication_skill_rating"] / 5,
        data["aptitude_skill_rating"] / 5,
        data["attendance_percentage"] / 100,
        data["cgpa"] / 10,
    ]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#141829")
    ax.set_facecolor("#141829")
    ax.plot(angles, values_plot, color="#818cf8", linewidth=2)
    ax.fill(angles, values_plot, color="#818cf8", alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="#a5b4fc", fontsize=8.5)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.grid(color="#1e2540", linewidth=1.2)
    ax.spines["polar"].set_color("#1e2540")
    return fig


def sidebar_input_form():
    st.sidebar.markdown("## ⚙️ Student Profile")
    st.sidebar.markdown("---")

    st.sidebar.markdown("**Demographics**")
    gender = st.sidebar.radio("Gender", ["Male", "Female"])
    branch = st.sidebar.selectbox("Branch", ["CSE", "ECE", "IT", "ME", "CE"])
    part_time_job = st.sidebar.radio("Part-time Job?", ["Yes", "No"])
    internet_access = st.sidebar.radio("Internet Access?", ["Yes", "No"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Background**")
    family_income_level = st.sidebar.selectbox("Family Income", ["Low", "Medium", "High"])
    city_tier = st.sidebar.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    extracurricular_involvement = st.sidebar.selectbox("Extracurricular", ["Low", "Medium", "High"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Academic**")
    cgpa = st.sidebar.number_input("CGPA", 0.0, 10.0, step=0.01)
    tenth_percentage = st.sidebar.number_input("10th %", 0.0, 100.0, step=0.1)
    twelfth_percentage = st.sidebar.number_input("12th %", 0.0, 100.0, step=0.1)
    backlogs = st.sidebar.number_input("Backlogs", 0, step=1)
    attendance_percentage = st.sidebar.number_input("Attendance %", 0.0, 100.0, step=0.1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Lifestyle**")
    study_hours_per_day = st.sidebar.number_input("Study Hours/Day", 0.0, 24.0, step=0.5)
    sleep_hours = st.sidebar.number_input("Sleep Hours/Day", 0.0, 12.0, step=0.5)
    stress_level = st.sidebar.slider("Stress Level", 1, 10, 5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Experience**")
    projects_completed = st.sidebar.number_input("Projects", 0, step=1)
    internships_completed = st.sidebar.number_input("Internships", 0, step=1)
    hackathons_participated = st.sidebar.number_input("Hackathons", 0, step=1)
    certifications_count = st.sidebar.number_input("Certifications", 0, step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Skill Ratings**")
    coding_skill_rating = st.sidebar.slider("Coding Skill", 1, 5, 3)
    communication_skill_rating = st.sidebar.slider("Communication", 1, 5, 3)
    aptitude_skill_rating = st.sidebar.slider("Aptitude", 1, 5, 3)

    return {
        'gender': gender,
        'branch': branch,
        'cgpa': float(cgpa),
        'tenth_percentage': float(tenth_percentage),
        'twelfth_percentage': float(twelfth_percentage),
        'backlogs': int(backlogs),
        'study_hours_per_day': float(study_hours_per_day),
        'attendance_percentage': float(attendance_percentage),
        'projects_completed': int(projects_completed),
        'internships_completed': int(internships_completed),
        'coding_skill_rating': int(coding_skill_rating),
        'communication_skill_rating': int(communication_skill_rating),
        'aptitude_skill_rating': int(aptitude_skill_rating),
        'hackathons_participated': int(hackathons_participated),
        'certifications_count': int(certifications_count),
        'sleep_hours': float(sleep_hours),
        'stress_level': int(stress_level),
        'part_time_job': part_time_job,
        'family_income_level': family_income_level,
        'city_tier': city_tier,
        'internet_access': internet_access,
        'extracurricular_involvement': extracurricular_involvement,
    }


def predict(model, data: dict):
    df = pd.DataFrame([data])
    df = df[model.feature_names_in_]
    return model.predict(df)[0]


def predict_proba(model, data: dict):
    try:
        df = pd.DataFrame([data])
        df = df[model.feature_names_in_]
        proba = model.predict_proba(df)[0]
        return float(max(proba))
    except Exception:
        return 0.75


def stat_chips_html(chips: list) -> str:
    items = "".join(
        f'<div class="stat-chip"><b>{k}</b>&nbsp;&nbsp;{v}</div>'
        for k, v in chips
    )
    return f'<div class="stat-row">{items}</div>'


def sidebar_model_status(models):
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="color:#475569;font-size:0.7rem;font-weight:700;'
        'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">Model Status</p>',
        unsafe_allow_html=True,
    )
    for label, key in [("Placement Model", "placement"), ("Salary Model", "salary")]:
        ok = models[key] is not None
        dot = "🟢" if ok else "🔴"
        status = "Loaded" if ok else "Not Found"
        color = "#34d399" if ok else "#f87171"
        st.sidebar.markdown(
            f'<div class="model-chip">'
            f'<span style="font-size:0.65rem">{dot}</span>'
            f'<div>'
            f'<div class="model-chip-label">{label}</div>'
            f'<div class="model-chip-status" style="color:{color}">{status}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    st.sidebar.markdown(
        '<p style="color:#334155;font-size:0.72rem;margin-top:8px;text-align:center">'
        'Place <code>.pkl</code> files in <code>artifacts/</code></p>',
        unsafe_allow_html=True,
    )


def main():
    load_css()
    models = load_models()

    # ── Sidebar nav ──
    st.sidebar.markdown("---")
    menu = st.sidebar.radio(
        "Mode",
        ["🎯  Placement Prediction", "💰  Salary Prediction"],
    )
    st.sidebar.markdown("---")

    data = sidebar_input_form()
    sidebar_model_status(models)

    # ── Hero ──
    st.markdown("<h1>PlaceIQ</h1>", unsafe_allow_html=True)
    st.markdown('<p class="tagline">Recruitment Prediction · AI-powered</p>', unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # PLACEMENT MODE
    if "Placement" in menu:
        st.markdown('<div class="section-pill">Placement Prediction</div>', unsafe_allow_html=True)
        st.markdown("#### Will this student get placed?")

        if st.button("🚀  Run Placement Analysis"):
            if models['placement'] is not None:
                with st.spinner("Running model inference..."):
                    result = predict(models['placement'], data)
                    confidence = predict_proba(models['placement'], data)

                is_placed = str(result).lower() in ["placed", "1", "yes"]
                verdict_cls = "verdict-placed" if is_placed else "verdict-not-placed"
                icon = "✅" if is_placed else "❌"

                with st.container(border=True):
                    col_verdict, col_gauge, col_radar = st.columns([1.5, 1.2, 1.2])

                    with col_verdict:
                        st.markdown(
                            f'<div class="{verdict_cls}">{icon}&nbsp;{result}</div>',
                            unsafe_allow_html=True,
                        )
                        chips = [
                            ("CGPA", f"{data['cgpa']:.1f}"),
                            ("Branch", data["branch"]),
                            ("Backlogs", data["backlogs"]),
                            ("Internships", data["internships_completed"]),
                            ("Projects", data["projects_completed"]),
                            ("Hackathons", data["hackathons_participated"]),
                            ("Certs", data["certifications_count"]),
                        ]
                        st.markdown(stat_chips_html(chips), unsafe_allow_html=True)

                    with col_gauge:
                        st.markdown(
                            "<p style='color:#64748b;font-size:0.75rem;text-transform:uppercase;"
                            "letter-spacing:0.07em;margin-bottom:2px'>Model confidence</p>",
                            unsafe_allow_html=True,
                        )
                        st.pyplot(plot_placement_gauge(confidence, result), use_container_width=True)

                    with col_radar:
                        st.markdown(
                            "<p style='color:#64748b;font-size:0.75rem;text-transform:uppercase;"
                            "letter-spacing:0.07em;margin-bottom:2px'>Skill profile</p>",
                            unsafe_allow_html=True,
                        )
                        st.pyplot(plot_skill_radar(data), use_container_width=True)

    # SALARY MODE
    elif "Salary" in menu:
        st.markdown('<div class="section-pill">Salary Prediction</div>', unsafe_allow_html=True)
        st.markdown("#### Estimated starting package (LPA)")

        if st.button("💰  Estimate Salary"):
            if models['salary'] is not None:
                with st.spinner("Crunching numbers..."):
                    result = predict(models['salary'], data)

                try:
                    salary_val = float(result)
                except Exception:
                    salary_val = 0.0

                low  = salary_val * 0.85
                high = salary_val * 1.15
                dist_fig, pct = plot_salary_distribution(salary_val)

                with st.container(border=True):
                    col_num, col_chart = st.columns([1, 1.8])

                    with col_num:
                        st.markdown(
                            f'<div class="verdict-salary">${salary_val:.1f}<span>LPA</span></div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<p style='color:#64748b;font-size:0.82rem;margin-top:8px;'>"
                            f"Likely range &nbsp;<b style='color:#c7d2fe;'>${low:.1f}L – ${high:.1f}L</b>"
                            f"</p>",
                            unsafe_allow_html=True,
                        )
                        chips = [
                            ("Branch", data["branch"]),
                            ("City", data["city_tier"]),
                            ("Coding", f"{data['coding_skill_rating']}/5"),
                            ("Aptitude", f"{data['aptitude_skill_rating']}/5"),
                            ("CGPA", f"{data['cgpa']:.1f}"),
                            ("Percentile", f"{pct}th"),
                        ]
                        st.markdown(stat_chips_html(chips), unsafe_allow_html=True)

                    with col_chart:
                        st.markdown(
                            "<p style='color:#64748b;font-size:0.75rem;text-transform:uppercase;"
                            "letter-spacing:0.07em;margin-bottom:2px'>Salary distribution</p>",
                            unsafe_allow_html=True,
                        )
                        st.pyplot(dist_fig, use_container_width=True)
                        st.markdown(
                            "<p style='color:#64748b;font-size:0.75rem;text-transform:uppercase;"
                            "letter-spacing:0.07em;margin:8px 0 2px'>Skill profile</p>",
                            unsafe_allow_html=True,
                        )
                        st.pyplot(plot_skill_radar(data), use_container_width=True)


if __name__ == "__main__":
    main()