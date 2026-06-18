import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from model import load_or_train_model, predict_match, get_team_stats, TEAM_LIST

st.set_page_config(
    page_title="2026 World Cup Predictor",
    page_icon="⚽",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-title  { font-size: 1rem; color: #6b7280; margin-top: 0.2rem; }
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
    }
    .metric-label { font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .win-card  { border-left: 4px solid #22c55e; }
    .draw-card { border-left: 4px solid #f59e0b; }
    .loss-card { border-left: 4px solid #ef4444; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #374151; margin: 1.5rem 0 0.8rem; }
    div[data-testid="stSelectbox"] label { font-weight: 600; font-size: 0.9rem; }

    /* Hide Streamlit header/menu/footer */
    #MainMenu { opacity: 0 !important; visibility: hidden !important; height: 0 !important; pointer-events: none !important; }
    header, header[role="banner"] { opacity: 0 !important; visibility: hidden !important; height: 0 !important; margin: 0 !important; padding: 0 !important; pointer-events: none !important; }
    footer, footer * { display: none !important; visibility: hidden !important; height: 0 !important; pointer-events: none !important; }

    /* Hide anchors to GitHub / Streamlit share */
    a[href*="github.com"], a[href*="streamlit.io"], a[href*="share.streamlit.io"] { display: none !important; visibility: hidden !important; pointer-events: none !important; }

    /* Hide any ancestor element that contains a GitHub link (modern browsers supporting :has()) */
    div:has(a[href*="github.com"]), section:has(a[href*="github.com"]), footer:has(a[href*="github.com"]) { display: none !important; visibility: hidden !important; }

    /* Fallback overlay to cover the bottom-right badge if it can't be targeted directly */
    #__hide_github_overlay__ { position: fixed; right: 0; bottom: 0; width: 320px; height: 36px; background: white; z-index: 9999; pointer-events: none; }

</style>
<div id="__hide_github_overlay__"></div>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">⚽ 2026 World Cup Match Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by historical FIFA data · XGBoost classifier</p>', unsafe_allow_html=True)
st.divider()

# ── Load model ─────────────────────────────────────────────────────────────────
with st.spinner("Loading model and historical data..."):
    model, df, rankings, feature_cols, label_encoder = load_or_train_model()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Match Settings")
    tournament_type = st.selectbox(
        "Tournament stage",
        ["Group Stage", "Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"],
    )
    neutral_venue = st.checkbox("Neutral venue", value=True)
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown(f"- **Algorithm:** XGBoost\n- **Training matches:** {len(df):,}\n- **Features:** {len(feature_cols)}")
    st.markdown("---")
    st.caption("Data: Kaggle International Football Results 1872–2024")

# ── Team selectors ─────────────────────────────────────────────────────────────
col1, col_vs, col2 = st.columns([5, 1, 5])

with col1:
    st.markdown('<p class="section-header">🏠 Home / Team 1</p>', unsafe_allow_html=True)
    home_team = st.selectbox("Select home team", TEAM_LIST, index=TEAM_LIST.index("Brazil"), key="home")

with col_vs:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color:#6b7280; padding-top:0.5rem'>VS</h2>", unsafe_allow_html=True)

with col2:
    st.markdown('<p class="section-header">✈️ Away / Team 2</p>', unsafe_allow_html=True)
    away_options = [t for t in TEAM_LIST if t != home_team]
    default_away = "Argentina" if "Argentina" in away_options else away_options[0]
    away_team = st.selectbox("Select away team", away_options, index=away_options.index(default_away), key="away")

# ── Predict ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔮 Predict Match Outcome", type="primary", use_container_width=True)

if predict_btn or True:  # auto-run on load
    result = predict_match(model, df, home_team, away_team, neutral_venue, label_encoder, rankings)

    if result is None:
        st.warning("Not enough historical data for this matchup. Try a different pair.")
    else:
        st.divider()

        # ── Probability cards ──────────────────────────────────────────────────
        st.markdown(f"### {home_team}  vs  {away_team}")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="metric-card win-card">
                <div class="metric-label">{home_team} Win</div>
                <div class="metric-value" style="color:#22c55e">{result['home_win']:.0%}</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card draw-card">
                <div class="metric-label">Draw</div>
                <div class="metric-value" style="color:#f59e0b">{result['draw']:.0%}</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card loss-card">
                <div class="metric-label">{away_team} Win</div>
                <div class="metric-value" style="color:#ef4444">{result['away_win']:.0%}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Outcome chart + H2H ───────────────────────────────────────────────
        col_chart, col_stats = st.columns([3, 2])

        with col_chart:
            fig = go.Figure(go.Bar(
                x=[result['home_win'], result['draw'], result['away_win']],
                y=[f"{home_team} Win", "Draw", f"{away_team} Win"],
                orientation='h',
                marker_color=['#22c55e', '#f59e0b', '#ef4444'],
                text=[f"{result['home_win']:.1%}", f"{result['draw']:.1%}", f"{result['away_win']:.1%}"],
                textposition='outside',
                textfont=dict(size=14, color='#374151'),
            ))
            fig.update_layout(
                title="Predicted Probabilities",
                xaxis=dict(range=[0, 1], tickformat='.0%', showgrid=True, gridcolor='#f1f5f9'),
                yaxis=dict(tickfont=dict(size=13)),
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=220,
                margin=dict(l=10, r=60, t=40, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")

        with col_stats:
            st.markdown('<p class="section-header">📈 Head-to-Head History</p>', unsafe_allow_html=True)
            h2h = result['h2h']
            total = sum(h2h.values())
            if total > 0:
                st.metric(f"{home_team} wins", f"{h2h['home_wins']}  ({h2h['home_wins']/total:.0%})")
                st.metric("Draws", f"{h2h['draws']}  ({h2h['draws']/total:.0%})")
                st.metric(f"{away_team} wins", f"{h2h['away_wins']}  ({h2h['away_wins']/total:.0%})")
                st.caption(f"Based on {total} historical meetings")
            else:
                st.info("No recorded head-to-head matches.")

        # ── Team form ──────────────────────────────────────────────────────────
        st.markdown('<p class="section-header">🔥 Recent Form (last 15 matches)</p>', unsafe_allow_html=True)
        f1, f2 = st.columns(2)

        home_stats = get_team_stats(df, home_team)
        away_stats = get_team_stats(df, away_team)

        def form_bar(stats, team_name, color):
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Wins",   x=["W"], y=[stats['wins']],   marker_color='#22c55e'))
            fig.add_trace(go.Bar(name="Draws",  x=["D"], y=[stats['draws']],  marker_color='#f59e0b'))
            fig.add_trace(go.Bar(name="Losses", x=["L"], y=[stats['losses']], marker_color='#ef4444'))
            fig.update_layout(
                title=dict(text=team_name, font=dict(size=13)),
                barmode='group',
                height=200,
                margin=dict(l=5, r=5, t=35, b=5),
                plot_bgcolor='white', paper_bgcolor='white',
                showlegend=True,
                legend=dict(orientation='h', y=-0.15, font=dict(size=11)),
                yaxis=dict(dtick=1, gridcolor='#f1f5f9'),
            )
            return fig

        with f1:
            st.plotly_chart(form_bar(home_stats, home_team, '#22c55e'), width="stretch")
            st.caption(f"Goals scored/conceded: {home_stats['gf']:.1f} / {home_stats['ga']:.1f} per game")

        with f2:
            st.plotly_chart(form_bar(away_stats, away_team, '#3b82f6'), width="stretch")
            st.caption(f"Goals scored/conceded: {away_stats['gf']:.1f} / {away_stats['ga']:.1f} per game")

        # ── Verdict ────────────────────────────────────────────────────────────
        best_prob = max(result['home_win'], result['draw'], result['away_win'])
        if best_prob == result['home_win']:
            verdict = f"🟢 **{home_team}** are favoured to win ({result['home_win']:.0%} probability)"
            verdict_color = "#f0fdf4"
        elif best_prob == result['away_win']:
            verdict = f"🔴 **{away_team}** are favoured to win ({result['away_win']:.0%} probability)"
            verdict_color = "#fff1f2"
        else:
            verdict = f"🟡 A **draw** is the most likely outcome ({result['draw']:.0%} probability)"
            verdict_color = "#fffbeb"

        st.markdown(f"""
        <div style="background:{verdict_color}; border-radius:10px; padding:1rem 1.5rem; margin-top:1rem;">
            <strong>Model verdict:</strong> {verdict}
        </div>""", unsafe_allow_html=True)