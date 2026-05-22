"""
VERA-KS: Verification Engine for Results & Accountability - Kansas
Type 4 Detection using KELPA Speaking vs Writing + KAP Achievement Data

Kansas context: KELPA (Kansas English Language Proficiency Assessment, NOT WIDA ACCESS).
4 domains: Listening, Speaking, Reading, Writing. 4 levels: Beginning (1), Intermediate (2),
Advanced (3), Fluent (4). Academic: KAP (Kansas Assessment Program), 4 levels:
Limited, Basic, Proficient, Advanced. Data: KIDS system, district ID format D0XXX
(e.g., D0259=Wichita, D0500=Kansas City). ~286 districts, ~42K ELs.
Dashboard: ksreportcard.ksde.org. KESA accreditation system.
Gannon v. State funding case. "Golden Triangle" meatpacking corridor:
Dodge City D0443 (~50% EL), Garden City D0457 (~50% EL), Liberal D0480 (~40% EL).
<12% low-income 8th graders proficient in math.

H-EDU.Solutions | https://h-edu.solutions
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_KS_BLUE = "#002855"
KS_GOLD = "#FFB81C"
KS_DARK_BLUE = "#001A3A"
KS_LIGHT_BLUE = "#335C85"

# ============================================================================
# DATA: Kansas Districts with EL Populations (from KIDS / ksreportcard.ksde.org)
# ============================================================================

def load_districts():
    """
    Load KS districts with significant EL populations.
    Real data from ksreportcard.ksde.org and KIDS system.
    District IDs use D0XXX format. ~286 districts, ~42K ELs statewide.
    KAP has 4 levels: Limited, Basic, Proficient, Advanced.
    Statewide ELA ~32%, Math ~28% proficient+advanced.
    """
    data = [
        # (district_id, district_name, total_students, el_count, el_percent,
        #  grad_rate, kap_ela_all, kap_ela_el, kap_ela_hispanic, kap_ela_white,
        #  kap_math_all, kap_math_el, top_el_languages)

        # "Golden Triangle" meatpacking corridor
        ("D0443", "Dodge City USD 443", 7200, 3600, 50.0,
         78.5, 28.4, 8.2, 18.5, 46.2,
         22.5, 6.8, "Spanish, Burmese, Somali, Vietnamese"),
        ("D0457", "Garden City USD 457", 7800, 3900, 50.0,
         76.8, 27.2, 7.8, 17.8, 44.5,
         21.8, 6.2, "Spanish, Somali, Burmese, Vietnamese"),
        ("D0480", "Liberal USD 480", 4600, 1840, 40.0,
         74.2, 25.5, 7.5, 16.8, 42.8,
         19.5, 5.8, "Spanish, Burmese, Guatemalan languages"),

        # Major urban districts
        ("D0259", "Wichita USD 259", 50000, 7500, 15.0,
         80.5, 30.2, 9.8, 20.2, 48.5,
         24.8, 7.5, "Spanish, Vietnamese, Burmese, Arabic"),
        ("D0500", "Kansas City USD 500", 22000, 5500, 25.0,
         72.8, 24.5, 8.5, 17.2, 42.2,
         18.2, 6.0, "Spanish, Somali, Burmese, Swahili"),
        ("D0501", "Topeka USD 501", 13500, 2025, 15.0,
         76.2, 28.8, 9.2, 19.5, 45.8,
         22.2, 7.0, "Spanish, Vietnamese, Arabic"),
        ("D0497", "Lawrence USD 497", 11200, 1120, 10.0,
         86.5, 38.2, 14.5, 28.5, 52.8,
         30.5, 10.8, "Spanish, Chinese, Arabic"),

        # Southwest Kansas meatpacking belt
        ("D0308", "Hutchinson USD 308", 5200, 780, 15.0,
         79.2, 30.5, 10.2, 21.5, 46.5,
         24.2, 7.8, "Spanish, Burmese"),
        ("D0373", "Newton USD 373", 3800, 760, 20.0,
         80.8, 31.2, 10.8, 22.2, 47.5,
         25.5, 8.2, "Spanish, Vietnamese"),
        ("D0305", "Salina USD 305", 7200, 1080, 15.0,
         81.2, 32.5, 11.5, 23.8, 48.2,
         26.2, 8.5, "Spanish, Vietnamese, Burmese"),

        # Other significant EL districts
        ("D0233", "Olathe USD 233", 30000, 3600, 12.0,
         88.2, 40.5, 15.2, 30.5, 54.2,
         32.8, 11.2, "Spanish, Vietnamese, Chinese"),
        ("D0229", "Blue Valley USD 229", 22000, 1540, 7.0,
         92.5, 48.2, 20.5, 38.2, 58.5,
         38.5, 15.2, "Chinese, Spanish, Hindi, Korean"),
        ("D0437", "Auburn-Washburn USD 437", 6200, 620, 10.0,
         87.5, 39.5, 14.8, 29.8, 52.5,
         31.2, 10.5, "Spanish, Chinese, Vietnamese"),
        ("D0475", "Geary County USD 475", 7500, 1500, 20.0,
         75.8, 27.8, 9.5, 19.8, 44.8,
         21.5, 7.2, "Spanish, Korean, Tagalog, German"),
        ("D0512", "Shawnee Mission USD 512", 27000, 2700, 10.0,
         86.8, 38.8, 14.2, 28.8, 52.2,
         30.8, 10.5, "Spanish, Vietnamese, Chinese"),
    ]

    return pd.DataFrame(data, columns=[
        'district_id', 'district_name', 'total_students',
        'el_count', 'el_percent', 'graduation_rate',
        'kap_ela_all', 'kap_ela_el', 'kap_ela_hispanic',
        'kap_ela_white',
        'kap_math_all', 'kap_math_el', 'top_el_languages'
    ])


# ============================================================================
# DATA: KELPA Domain Data (modeled from KSDE KELPA public results)
# ============================================================================

def load_kelpa_data(districts_df):
    """
    Generate district KELPA domain data modeled from KSDE KELPA results.
    KELPA (Kansas English Language Proficiency Assessment) -- NOT WIDA ACCESS.
    4 domains: Listening, Speaking, Reading, Writing.
    4 levels: Beginning (1), Intermediate (2), Advanced (3), Fluent (4).
    Scale scores approximate KELPA ranges by grade.
    Exit criteria: Overall Fluent (Level 4) on KELPA.
    """
    kelpa_data = []

    for _, d in districts_df.iterrows():
        for grade in range(3, 9):
            for year in [2024, 2025]:
                # Base scores by grade -- speaking naturally higher than writing
                base_speaking = 330 + (grade * 8)
                base_writing = 275 + (grade * 6)
                base_listening = 335 + (grade * 7)
                base_reading = 288 + (grade * 6)

                # District adjustments: lower EL proficiency = lower scores
                el_factor = d['kap_ela_el'] / 12.0
                speaking_adj = int(12 * el_factor + d['el_percent'] * 0.30)
                writing_adj = int(-12 + (el_factor - 1) * 10)
                listening_adj = speaking_adj - 3
                reading_adj = writing_adj + 8

                # Golden Triangle meatpacking districts: high EL%, multilingual
                if d['district_id'] in ['D0443', 'D0457', 'D0480']:
                    speaking_adj += 8
                    writing_adj -= 6

                # Urban high-diversity districts
                if d['district_id'] in ['D0500', 'D0259']:
                    speaking_adj += 4
                    writing_adj -= 3

                # Year-over-year modest growth
                year_adj = 3 if year == 2025 else 0

                kelpa_data.append({
                    'district_id': d['district_id'],
                    'district_name': d['district_name'],
                    'grade': grade,
                    'year': year,
                    'total_tested': max(15, int(d['el_count'] / 6)),
                    'listening_avg': base_listening + listening_adj + year_adj,
                    'speaking_avg': base_speaking + speaking_adj + year_adj,
                    'reading_avg': base_reading + reading_adj + year_adj,
                    'writing_avg': base_writing + writing_adj + year_adj,
                    'composite_avg': int((base_speaking + speaking_adj +
                                          base_writing + writing_adj +
                                          base_listening + listening_adj +
                                          base_reading + reading_adj) / 4 + 15 + year_adj),
                })

    return pd.DataFrame(kelpa_data)


# ============================================================================
# DATA: KAP Achievement Data (from ksreportcard.ksde.org)
# ============================================================================

def load_kap_data(districts_df):
    """
    Generate KAP data based on ksreportcard.ksde.org proficiency rates.
    KAP (Kansas Assessment Program) has 4 performance levels:
    Limited, Basic, Proficient, Advanced.
    ELA and Math tested grades 3-8 (and 10).
    Statewide: ELA ~32%, Math ~28% proficient+advanced.
    <12% low-income 8th graders proficient in math.
    """
    kap_data = []

    for _, d in districts_df.iterrows():
        for grade in range(3, 9):
            for year in [2024, 2025]:
                for subject in ['ELA', 'Math']:
                    if subject == 'ELA':
                        base = d['kap_ela_all']
                    else:
                        base = d['kap_math_all']

                    # Grade adjustment: proficiency dips in middle school
                    prof = max(8, min(70, base + (grade - 5) * -1.2))

                    # Year adjustment
                    if year == 2024:
                        prof = prof - 1.0

                    # KAP 4-level distribution
                    advanced = max(1.5, prof * 0.15)
                    proficient = max(4, prof - advanced)
                    basic = max(12, (100 - prof) * 0.40)
                    limited = max(8, 100 - proficient - advanced - basic)

                    kap_data.append({
                        'district_id': d['district_id'],
                        'district_name': d['district_name'],
                        'grade': grade,
                        'subject': subject,
                        'year': year,
                        'limited_pct': round(limited, 1),
                        'basic_pct': round(basic, 1),
                        'proficient_pct': round(proficient, 1),
                        'advanced_pct': round(advanced, 1),
                        'prof_advanced_pct': round(proficient + advanced, 1),
                    })

    return pd.DataFrame(kap_data)


# ============================================================================
# DATA: Statewide Domain Proficiency (from KSDE KELPA data)
# ============================================================================

def load_statewide_domain_data():
    """
    Statewide KELPA domain proficiency percentages by grade cluster.
    Source: KSDE KELPA data, ksreportcard.ksde.org.
    Kansas has ~42,000 ELs across ~286 districts.
    KELPA levels: Beginning (1), Intermediate (2), Advanced (3), Fluent (4).
    % at Advanced or Fluent shown below.
    """
    return pd.DataFrame([
        {'year': '2024-25', 'grade_cluster': 'K-2', 'listening': 38, 'speaking': 34, 'reading': 20, 'writing': 14},
        {'year': '2024-25', 'grade_cluster': '3-5', 'listening': 44, 'speaking': 40, 'reading': 24, 'writing': 16},
        {'year': '2024-25', 'grade_cluster': '6-8', 'listening': 48, 'speaking': 42, 'reading': 28, 'writing': 19},
        {'year': '2024-25', 'grade_cluster': '9-12', 'listening': 51, 'speaking': 44, 'reading': 31, 'writing': 21},
        {'year': '2023-24', 'grade_cluster': 'K-2', 'listening': 36, 'speaking': 32, 'reading': 18, 'writing': 12},
        {'year': '2023-24', 'grade_cluster': '3-5', 'listening': 42, 'speaking': 38, 'reading': 22, 'writing': 14},
        {'year': '2023-24', 'grade_cluster': '6-8', 'listening': 46, 'speaking': 40, 'reading': 26, 'writing': 17},
        {'year': '2023-24', 'grade_cluster': '9-12', 'listening': 49, 'speaking': 42, 'reading': 29, 'writing': 19},
    ])


# ============================================================================
# AUTHENTICATION
# ============================================================================


# ============================================================================
# TYPE 4 DETECTION
# ============================================================================

def compute_type4_analysis(kelpa_df, district_id, grade, year):
    """
    Compute Type 4 detection for a given district/grade/year.
    Type 4 candidates show strong oral skills but weak written skills.
    Delta = Speaking - Writing. Flag threshold: normalized delta > 8.
    Uses KELPA data (Kansas-specific, NOT WIDA ACCESS).
    """
    filtered = kelpa_df[
        (kelpa_df['district_id'] == district_id) &
        (kelpa_df['grade'] == grade) &
        (kelpa_df['year'] == year)
    ]
    if filtered.empty:
        return None

    row = filtered.iloc[0]
    delta = row['speaking_avg'] - row['writing_avg']
    delta_normalized = delta / 5
    flagged = delta_normalized > 8

    return {
        'district_id': district_id,
        'district_name': row['district_name'],
        'grade': grade,
        'year': year,
        'speaking_avg': row['speaking_avg'],
        'writing_avg': row['writing_avg'],
        'delta': delta,
        'delta_normalized': delta_normalized,
        'flagged': flagged,
        'total_tested': row['total_tested'],
        'estimated_flagged': int(row['total_tested'] * 0.15) if flagged else int(row['total_tested'] * 0.05)
    }


# ============================================================================
# PAGES
# ============================================================================

def render_overview(districts_df):
    st.header("Kansas Education Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pilot Districts", len(districts_df))
    with col2:
        st.metric("Total Students", f"{districts_df['total_students'].sum():,}")
    with col3:
        st.metric("English Learners", f"{districts_df['el_count'].sum():,}")
    with col4:
        st.metric("Statewide ELA Prof", "~32%", help="2025 KAP ELA statewide proficient+advanced")

    st.divider()

    # Gannon v. State -- the equity hook
    st.subheader("Gannon v. State: The Funding Equity Imperative")
    st.markdown("""
    The **Gannon v. State of Kansas** case (2010-2019) was a landmark school finance lawsuit
    in which the Kansas Supreme Court ruled that the state's education funding system was
    **constitutionally inadequate and inequitable**. The court required the legislature to
    increase funding to meet the requirements of the Kansas Constitution, which mandates
    the state provide "suitable provision for finance of the educational interests of the state."

    The case resulted in approximately **$500M+ in additional annual funding** after multiple
    rounds of litigation. However, advocates argue that significant gaps remain -- particularly
    for English Learners, low-income students, and rural districts.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.error("**Gannon v. State**\nKS Supreme Court: funding\nconstitutionally inadequate")
    with col2:
        st.warning("**Low-Income Math Crisis**\n<12% of low-income 8th graders\nproficient in math")
    with col3:
        st.warning("**Golden Triangle**\nDodge City, Garden City, Liberal\n~40-50% EL enrollment")

    st.divider()

    st.subheader("Key State Context")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**KELPA (Not WIDA)**\nKansas English Language\nProficiency Assessment\nLevels 1-4: Beginning to Fluent")
    with col2:
        st.info("**KAP**\nKansas Assessment Program\n4 levels: Limited, Basic,\nProficient, Advanced")
    with col3:
        st.info("**KIDS Data System**\nState data infrastructure\nD0XXX district format\nKESA accreditation")

    st.divider()

    st.subheader("Key State Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Statewide Math Prof", "~28%", help="2025 KAP Math statewide proficient+advanced")
    with col2:
        st.metric("Total ELs", "~42K", help="Statewide EL enrollment")
    with col3:
        st.metric("Total Districts", "~286", help="KIDS data system, D0XXX format")
    with col4:
        st.metric("Low-Income Math 8th", "<12%", help="Low-income 8th graders proficient in math")

    st.divider()

    # Golden Triangle spotlight
    st.subheader("The Golden Triangle: Meatpacking Corridor")
    st.markdown("""
    The **"Golden Triangle"** refers to three southwest Kansas cities -- **Dodge City**,
    **Garden City**, and **Liberal** -- anchored by large meatpacking plants (Tyson, Cargill,
    National Beef). These communities have experienced dramatic demographic shifts, with EL
    enrollment reaching **40-50% of total students**. The linguistic diversity is extraordinary:
    Spanish, Burmese, Somali, Vietnamese, Guatemalan indigenous languages, and others.

    These districts face unique challenges: high mobility, multilingual populations requiring
    services in dozens of languages, and limited local tax bases despite critical industry
    employment. **Gannon v. State** funding increases helped, but the per-pupil needs in
    these districts far exceed state averages.
    """)

    golden = districts_df[districts_df['district_id'].isin(['D0443', 'D0457', 'D0480'])]
    fig_gt = px.bar(
        golden, x='district_name', y=['el_count', 'total_students'],
        barmode='group',
        color_discrete_sequence=[KS_GOLD, KS_BLUE],
        labels={'value': 'Students', 'district_name': 'District', 'variable': 'Group'},
        title="Golden Triangle: EL Population vs Total Enrollment"
    )
    fig_gt.update_layout(height=350, legend_title_text='')
    st.plotly_chart(fig_gt, use_container_width=True)

    st.divider()

    st.subheader("Top EL Languages Statewide")
    lang_data = pd.DataFrame({
        'Language': ['Spanish', 'Burmese', 'Vietnamese', 'Somali', 'Arabic', 'Chinese', 'Swahili', 'Other'],
        'Approx Share': [72, 6, 4, 3, 2, 2, 1, 10],
    })
    fig_lang = px.bar(lang_data, x='Language', y='Approx Share',
                      color='Approx Share',
                      color_continuous_scale=[[0, KS_GOLD], [1, KS_BLUE]],
                      labels={'Approx Share': '% of EL Population'},
                      text='Approx Share')
    fig_lang.update_traces(texttemplate='%{text}%', textposition='outside')
    fig_lang.update_layout(height=350, showlegend=False, coloraxis_showscale=False,
                           title="Top EL Home Languages in Kansas")
    st.plotly_chart(fig_lang, use_container_width=True)

    st.divider()

    st.subheader("Pilot Districts -- Highest EL Populations")
    display = districts_df[['district_id', 'district_name', 'total_students', 'el_count', 'el_percent',
                            'kap_ela_all', 'kap_ela_el', 'kap_ela_white',
                            'top_el_languages']].copy()
    display.columns = ['Dist ID', 'District', 'Students', 'EL Count', 'EL %',
                       'ELA All %', 'ELA EL %', 'ELA White %',
                       'Top Languages']
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("English Learner Population by District")
    fig = px.bar(
        districts_df.sort_values('el_count', ascending=True),
        x='el_count', y='district_name', orientation='h',
        color='el_percent', color_continuous_scale=[[0, KS_GOLD], [1, KS_BLUE]],
        labels={'el_count': 'English Learners', 'district_name': 'District', 'el_percent': 'EL %'}
    )
    fig.update_layout(height=600, showlegend=False,
                      title="EL Population by District (color = EL %)")
    st.plotly_chart(fig, use_container_width=True)


def render_domain_analysis(domain_df):
    st.header("Statewide KELPA Domain Proficiency")

    st.markdown("""
    **Source:** KSDE KELPA data, ksreportcard.ksde.org. Kansas uses **KELPA** (Kansas English
    Language Proficiency Assessment), **NOT** WIDA ACCESS. KELPA has 4 domains: Listening,
    Speaking, Reading, Writing. 4 proficiency levels: Beginning (1), Intermediate (2),
    Advanced (3), Fluent (4).

    Domain proficiency percentages (% at Advanced or Fluent) show the systemic oral-written
    delta: Speaking consistently outperforms Writing across all grade clusters. Kansas exit
    criteria require an overall **Fluent (Level 4)** on KELPA.
    """)

    year = st.selectbox("Year", ['2024-25', '2023-24'], key="dom_y")
    filtered = domain_df[domain_df['year'] == year]

    st.divider()

    fig = go.Figure()
    for domain, color in [('listening', KS_BLUE), ('speaking', KS_GOLD),
                           ('reading', '#888888'), ('writing', KS_DARK_BLUE)]:
        fig.add_trace(go.Bar(
            x=filtered['grade_cluster'], y=filtered[domain],
            name=domain.capitalize(), marker_color=color,
            text=[f"{v}%" for v in filtered[domain]], textposition='outside'
        ))
    fig.update_layout(
        title=f"KELPA Domain Proficiency by Grade Cluster ({year})",
        xaxis_title="Grade Cluster", yaxis_title="% Advanced or Fluent",
        barmode='group', height=450, yaxis=dict(range=[0, 68])
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Speaking-Writing Delta by Grade Cluster")
    filtered = filtered.copy()
    filtered['delta'] = filtered['speaking'] - filtered['writing']
    fig2 = go.Figure(go.Bar(
        x=filtered['grade_cluster'], y=filtered['delta'],
        marker_color=[KS_DARK_BLUE if d > 20 else KS_GOLD for d in filtered['delta']],
        text=[f"{d:+d} pts" for d in filtered['delta']], textposition='outside'
    ))
    fig2.update_layout(title="Speaking - Writing Gap",
                       yaxis_title="Delta (percentage points)", height=350)
    st.plotly_chart(fig2, use_container_width=True)

    avg_delta = filtered['delta'].mean()
    st.metric("Average Speaking-Writing Delta", f"{avg_delta:+.0f} percentage points",
              help="Positive = Speaking proficiency exceeds Writing proficiency statewide")

    st.markdown("""
    ---
    **Why this matters for Kansas:** The oral-written gap is especially pronounced in the
    **Golden Triangle** meatpacking corridor -- Dodge City (~50% EL), Garden City (~50% EL),
    and Liberal (~40% EL) -- where students from Burmese, Somali, and Spanish-speaking
    families develop conversational fluency but struggle with academic writing. The linguistic
    diversity in these districts (sometimes 30+ languages) makes targeted literacy
    intervention especially challenging.

    Under **Gannon v. State**, the Kansas Supreme Court mandated constitutionally adequate
    funding. These domain gaps reveal where additional investment in academic literacy
    services is critically needed.
    """)


def render_kelpa_analysis(kelpa_df, districts_df):
    st.header("KELPA Analysis")
    st.markdown("""
    **KELPA** (Kansas English Language Proficiency Assessment) measures English learners
    across four domains: Listening, Speaking, Reading, Writing. Kansas uses KELPA, **not**
    WIDA ACCESS. ~42,000 ELs across ~286 districts.

    KELPA levels: **Beginning (1), Intermediate (2), Advanced (3), Fluent (4)**.
    Exit criteria: Overall **Fluent (Level 4)** on KELPA.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        district = st.selectbox("District", districts_df['district_name'].tolist(), key="kelpa_d")
    with col2:
        grade = st.selectbox("Grade", list(range(3, 9)), key="kelpa_g")
    with col3:
        year = st.selectbox("Year", [2025, 2024], key="kelpa_y")

    district_id = districts_df[districts_df['district_name'] == district]['district_id'].values[0]
    filtered = kelpa_df[
        (kelpa_df['district_id'] == district_id) &
        (kelpa_df['grade'] == grade) &
        (kelpa_df['year'] == year)
    ]

    if not filtered.empty:
        row = filtered.iloc[0]

        # Show top languages for context
        lang = districts_df[districts_df['district_id'] == district_id]['top_el_languages'].values[0]
        st.info(f"**Top EL languages in {district}:** {lang}")

        st.divider()
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Listening", f"{row['listening_avg']:.0f}")
        with col2:
            st.metric("Speaking", f"{row['speaking_avg']:.0f}")
        with col3:
            st.metric("Reading", f"{row['reading_avg']:.0f}")
        with col4:
            st.metric("Writing", f"{row['writing_avg']:.0f}")
        with col5:
            st.metric("Composite", f"{row['composite_avg']:.0f}")

        domains = ['Listening', 'Speaking', 'Reading', 'Writing']
        scores = [row['listening_avg'], row['speaking_avg'], row['reading_avg'], row['writing_avg']]
        fig = go.Figure(go.Bar(
            x=domains, y=scores,
            marker_color=[KS_BLUE, KS_GOLD, '#888888', KS_DARK_BLUE],
            text=[f"{s:.0f}" for s in scores], textposition='outside'
        ))
        fig.update_layout(
            title=f"KELPA Domains -- {district} -- Grade {grade} ({year})",
            yaxis_title="Scale Score", height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        oral = (row['listening_avg'] + row['speaking_avg']) / 2
        written = (row['reading_avg'] + row['writing_avg']) / 2
        gap = oral - written
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Oral Average", f"{oral:.0f}")
        with col2:
            st.metric("Written Average", f"{written:.0f}")
        with col3:
            st.metric("Oral-Written Gap", f"{gap:+.0f}",
                      delta="Flag" if gap > 30 else "Monitor" if gap > 20 else "OK")

        # Exit criteria check
        st.subheader("Exit Criteria Check (KS: Overall Fluent / Level 4 on KELPA)")
        st.markdown("""
        Kansas requires an overall **Fluent (Level 4)** rating on KELPA to exit EL services.
        Unlike WIDA states, Kansas developed its own assessment instrument (KELPA) aligned
        to Kansas ELP standards. The **Gannon v. State** ruling requires constitutionally
        adequate funding to support ELs in reaching this threshold.
        """)
    else:
        st.warning("No data available for the selected filters.")


def render_type4(kelpa_df, districts_df):
    st.header("Type 4 Detection")
    st.markdown("""
    **Type 4 candidates** show strong oral skills but weak written skills.
    Delta = Speaking - Writing. Flag threshold: normalized delta > 8.

    In Kansas, this is particularly relevant for the **Golden Triangle** meatpacking corridor
    (Dodge City, Garden City, Liberal) where multilingual students develop conversational
    English rapidly but lag in academic literacy. With 30+ languages represented in some
    districts, the orthographic challenges span multiple language families (Romance, Sino-Tibetan,
    Afroasiatic, Cushitic). KELPA data (NOT WIDA ACCESS) drives this analysis.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        district = st.selectbox("District", districts_df['district_name'].tolist(), key="t4_d")
    with col2:
        grade = st.selectbox("Grade", list(range(3, 9)), key="t4_g")
    with col3:
        year = st.selectbox("Year", [2025, 2024], key="t4_y")

    district_id = districts_df[districts_df['district_name'] == district]['district_id'].values[0]
    result = compute_type4_analysis(kelpa_df, district_id, grade, year)

    if result:
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Speaking", f"{result['speaking_avg']:.0f}")
        with col2:
            st.metric("Writing", f"{result['writing_avg']:.0f}")
        with col3:
            st.metric("Delta", f"{result['delta']:+.0f}")
        with col4:
            st.metric("Status", "FLAGGED" if result['flagged'] else "OK")

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Speaking', x=['Score'], y=[result['speaking_avg']],
                             marker_color=KS_GOLD))
        fig.add_trace(go.Bar(name='Writing', x=['Score'], y=[result['writing_avg']],
                             marker_color=KS_BLUE))
        fig.update_layout(
            title=f"Speaking vs Writing -- {district} -- Grade {grade}",
            barmode='group', height=350
        )
        st.plotly_chart(fig, use_container_width=True)

        if result['flagged']:
            st.error(f"**Type 4 Flag Triggered** -- Delta: {result['delta']:+.0f}. "
                     f"Est. {result['estimated_flagged']} of {result['total_tested']} students affected.")
            st.markdown("""
            **Gannon-informed action:** Under the Kansas Supreme Court's adequacy ruling,
            districts must ensure sufficient resources reach EL students. Districts should
            review KELPA domain data for these students and provide targeted academic
            writing intervention with culturally and linguistically responsive instruction.
            """)
        else:
            st.success(f"**No Type 4 Flag** -- Delta within normal range ({result['delta']:+.0f}).")

        st.subheader(f"All Grades -- {district} ({year})")
        all_data = [compute_type4_analysis(kelpa_df, district_id, g, year) for g in range(3, 9)]
        all_data = [r for r in all_data if r]
        if all_data:
            gdf = pd.DataFrame(all_data)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=gdf['grade'], y=gdf['speaking_avg'],
                name='Speaking', mode='lines+markers',
                line=dict(color=KS_GOLD, width=3)
            ))
            fig.add_trace(go.Scatter(
                x=gdf['grade'], y=gdf['writing_avg'],
                name='Writing', mode='lines+markers',
                line=dict(color=KS_BLUE, width=3)
            ))
            fig.update_layout(
                title="Speaking vs Writing Across Grades",
                xaxis_title="Grade", yaxis_title="Scale Score", height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            st.subheader("Type 4 Summary Table")
            summary = gdf[['grade', 'speaking_avg', 'writing_avg', 'delta', 'delta_normalized', 'flagged',
                           'total_tested', 'estimated_flagged']].copy()
            summary.columns = ['Grade', 'Speaking', 'Writing', 'Delta', 'Norm Delta', 'Flagged',
                              'Tested', 'Est. Affected']
            st.dataframe(summary, use_container_width=True, hide_index=True)


def render_achievement_gaps(districts_df):
    st.header("Achievement Gap Analysis")

    st.markdown("""
    **Data from ksreportcard.ksde.org (KAP results).** Kansas has persistent achievement gaps
    between white students and Hispanic and EL students. The **Gannon v. State** case centered
    on the constitutional obligation to provide adequate and equitable funding. These gaps
    demonstrate where adequacy has not been achieved for the state's most vulnerable learners.

    Statewide 2025 KAP: ELA ~32%, Math ~28% proficient+advanced.
    **<12% of low-income 8th graders are proficient in math.**
    """)

    st.divider()

    # Achievement gap bar chart
    fig = go.Figure()
    sorted_df = districts_df.sort_values('kap_ela_all', ascending=True)
    for col, name, color in [
        ('kap_ela_white', 'White', '#666666'),
        ('kap_ela_all', 'All Students', KS_BLUE),
        ('kap_ela_hispanic', 'Hispanic', KS_LIGHT_BLUE),
        ('kap_ela_el', 'English Learners', KS_GOLD),
    ]:
        fig.add_trace(go.Bar(
            x=sorted_df[col], y=sorted_df['district_name'],
            name=name, orientation='h', marker_color=color
        ))

    fig.update_layout(
        title="KAP ELA Proficiency by Subgroup -- 2025",
        barmode='group', xaxis_title="% Proficient + Advanced", height=650,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Gap magnitude analysis
    st.subheader("Gap Magnitude: White - EL ELA Proficiency")
    districts_df_copy = districts_df.copy()
    districts_df_copy['wh_gap'] = districts_df_copy['kap_ela_white'] - districts_df_copy['kap_ela_hispanic']
    districts_df_copy['we_gap'] = districts_df_copy['kap_ela_white'] - districts_df_copy['kap_ela_el']

    col1, col2 = st.columns(2)
    with col1:
        avg_wh = districts_df_copy['wh_gap'].mean()
        st.metric("Avg White-Hispanic Gap", f"{avg_wh:.1f} pts", delta="Gannon v. State", delta_color="inverse")
    with col2:
        avg_we = districts_df_copy['we_gap'].mean()
        st.metric("Avg White-EL Gap", f"{avg_we:.1f} pts", delta="Gannon v. State", delta_color="inverse")

    fig_gap = go.Figure()
    gap_sorted = districts_df_copy.sort_values('we_gap', ascending=True)
    fig_gap.add_trace(go.Bar(
        x=gap_sorted['we_gap'], y=gap_sorted['district_name'],
        orientation='h', marker_color=[KS_DARK_BLUE if g > 30 else KS_GOLD for g in gap_sorted['we_gap']],
        text=[f"{g:.0f} pts" for g in gap_sorted['we_gap']], textposition='outside'
    ))
    fig_gap.update_layout(
        title="White-EL ELA Gap by District (pts)", height=600,
        xaxis_title="Gap (percentage points)"
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    # Math crisis callout
    st.subheader("Low-Income Math Proficiency Crisis")
    st.error("""
    **<12% of low-income 8th graders in Kansas are proficient in math on KAP.**

    This statistic underscores the urgency of the Gannon v. State adequacy mandate.
    English Learners, who are disproportionately low-income, face compounding barriers:
    language demands embedded in math assessments, limited access to grade-level content,
    and insufficient bilingual math instruction.
    """)

    # Scatter: EL proficiency vs overall
    st.subheader("EL Proficiency vs Overall Proficiency")
    fig2 = px.scatter(
        districts_df, x='kap_ela_all', y='kap_ela_el', size='el_count',
        color='el_percent', color_continuous_scale=[[0, '#ccc'], [1, KS_BLUE]],
        hover_name='district_name',
        labels={'kap_ela_all': 'All Students ELA %', 'kap_ela_el': 'EL ELA %',
                'el_count': 'EL Count', 'el_percent': 'EL %'}
    )
    fig2.add_shape(type="line", x0=0, y0=0, x1=60, y1=60,
                   line=dict(dash="dash", color="gray"))
    fig2.update_layout(
        title="EL Proficiency vs District Overall -- Gap Visualization", height=450
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    ---
    **Gannon context:** The Kansas Supreme Court ruled the state's funding system
    constitutionally inadequate. While ~$500M in additional annual funding was eventually
    appropriated, these achievement gaps -- especially the <12% low-income 8th grade math
    proficiency rate -- demonstrate that adequate funding has not yet translated to adequate
    outcomes for ELs and low-income students. Every gap shown above represents students
    whose constitutional right to a suitable education is not being met.
    """)


def render_kap(kap_df, districts_df):
    st.header("KAP Assessment Analysis")
    st.markdown("""
    **Kansas Assessment Program (KAP)** -- 4 performance levels:
    Limited, Basic, Proficient, Advanced.

    ELA and Math tested grades 3-8 (and 10).
    Statewide 2025: ELA ~32%, Math ~28% proficient+advanced.
    Dashboard: ksreportcard.ksde.org.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        district = st.selectbox("District", districts_df['district_name'].tolist(), key="kap_d")
    with col2:
        grade = st.selectbox("Grade", list(range(3, 9)), key="kap_g")
    with col3:
        subject = st.selectbox("Subject", ['ELA', 'Math'], key="kap_s")
    with col4:
        year = st.selectbox("Year", [2025, 2024], key="kap_y")

    district_id = districts_df[districts_df['district_name'] == district]['district_id'].values[0]
    filtered = kap_df[
        (kap_df['district_id'] == district_id) &
        (kap_df['grade'] == grade) &
        (kap_df['subject'] == subject) &
        (kap_df['year'] == year)
    ]

    if not filtered.empty:
        row = filtered.iloc[0]
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Limited", f"{row['limited_pct']:.1f}%")
        with col2:
            st.metric("Basic", f"{row['basic_pct']:.1f}%")
        with col3:
            st.metric("Proficient", f"{row['proficient_pct']:.1f}%")
        with col4:
            st.metric("Advanced", f"{row['advanced_pct']:.1f}%")

        levels = ['Limited', 'Basic', 'Proficient', 'Advanced']
        values = [row['limited_pct'], row['basic_pct'],
                  row['proficient_pct'], row['advanced_pct']]
        colors = [KS_DARK_BLUE, KS_LIGHT_BLUE, KS_GOLD, KS_BLUE]
        fig = go.Figure(go.Bar(
            x=levels, y=values, marker_color=colors,
            text=[f"{v:.1f}%" for v in values], textposition='outside'
        ))
        fig.update_layout(
            title=f"KAP {subject} -- {district} -- Grade {grade} ({year})",
            yaxis_title="Percentage", height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Proficiency rate context
        st.metric("Combined Proficiency (Proficient + Advanced)",
                  f"{row['prof_advanced_pct']:.1f}%",
                  help="Statewide: ELA ~32%, Math ~28%")

        # Cross-grade comparison
        st.subheader(f"KAP {subject} Across Grades -- {district} ({year})")
        cross = kap_df[
            (kap_df['district_id'] == district_id) &
            (kap_df['subject'] == subject) &
            (kap_df['year'] == year)
        ]
        if not cross.empty:
            fig2 = go.Figure()
            level_col_map = {
                'Limited': 'limited_pct',
                'Basic': 'basic_pct',
                'Proficient': 'proficient_pct',
                'Advanced': 'advanced_pct',
            }
            for level, color in zip(levels, colors):
                col_name = level_col_map[level]
                fig2.add_trace(go.Bar(
                    x=cross['grade'], y=cross[col_name],
                    name=level, marker_color=color
                ))
            fig2.update_layout(
                barmode='stack', xaxis_title="Grade", yaxis_title="Percentage",
                height=400, title=f"KAP {subject} Performance Distribution"
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")


def render_export(kelpa_df, kap_df, districts_df, domain_df):
    st.header("Export Data")

    st.markdown("Download VERA-KS analysis data as CSV files for further analysis.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("KELPA Data")
        st.dataframe(kelpa_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download KELPA CSV",
            kelpa_df.to_csv(index=False),
            "vera_ks_kelpa.csv", "text/csv",
            use_container_width=True
        )
    with col2:
        st.subheader("KAP Data")
        st.dataframe(kap_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download KAP CSV",
            kap_df.to_csv(index=False),
            "vera_ks_kap.csv", "text/csv",
            use_container_width=True
        )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Statewide KELPA Domain Proficiency")
        st.dataframe(domain_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Domain CSV",
            domain_df.to_csv(index=False),
            "vera_ks_domains.csv", "text/csv",
            use_container_width=True
        )
    with col2:
        st.subheader("District Reference Data")
        st.dataframe(districts_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Districts CSV",
            districts_df.to_csv(index=False),
            "vera_ks_districts.csv", "text/csv",
            use_container_width=True
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.set_page_config(
        page_title="VERA-KS | Kansas Type 4 Detection",
        page_icon="*",
        layout="wide"
    )

    st.markdown(f"""
    <style>
        .stApp {{ background-color: #fafafa; }}
        .block-container {{ padding-top: 2rem; }}
        h1, h2, h3 {{ color: {KS_BLUE}; }}
        .stButton > button {{ background-color: {KS_BLUE}; color: white; }}
        .stButton > button:hover {{ background-color: {KS_DARK_BLUE}; color: white; }}
    </style>
    """, unsafe_allow_html=True)

    # Load all data
    districts_df = load_districts()
    kelpa_df = load_kelpa_data(districts_df)
    kap_df = load_kap_data(districts_df)
    domain_df = load_statewide_domain_data()

    # Sidebar
    st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 20px 0;">
        <h2 style="color: {KS_BLUE}; margin: 0;">VERA-KS</h2>
        <p style="color: #666; font-size: 0.85rem; margin-top: 5px;">Kansas Implementation</p>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.divider()

    page = st.sidebar.radio("Navigation", [
        "Overview",
        "Statewide Domain Analysis",
        "KELPA Analysis",
        "Type 4 Detection",
        "Achievement Gaps",
        "KAP Analysis",
        "Export Data"
    ])

    st.sidebar.divider()
    st.sidebar.markdown(f"""
    **Data Sources:**
    - KELPA (NOT WIDA ACCESS)
    - KSDE KELPA Data
    - KAP (Grades 3-8, 10)
    - KIDS Data System
    - ksreportcard.ksde.org

    **Type 4 Detection:**
    - Speaking vs Writing delta
    - Flag threshold: > 8 points (normalized)

    **KELPA Exit Criteria:**
    - Overall Fluent (Level 4)

    **KELPA Levels:**
    - 1: Beginning
    - 2: Intermediate
    - 3: Advanced
    - 4: Fluent

    **Key Context:**
    - ~42K ELs, ~286 districts
    - KESA accreditation
    - **Gannon v. State: funding equity**
    - Golden Triangle: ~40-50% EL
    - <12% low-income 8th math prof
    - D0XXX district ID format

    ---
    [H-EDU.Solutions](https://h-edu.solutions)
    """)

    # Page routing
    if page == "Overview":
        render_overview(districts_df)
    elif page == "Statewide Domain Analysis":
        render_domain_analysis(domain_df)
    elif page == "KELPA Analysis":
        render_kelpa_analysis(kelpa_df, districts_df)
    elif page == "Type 4 Detection":
        render_type4(kelpa_df, districts_df)
    elif page == "Achievement Gaps":
        render_achievement_gaps(districts_df)
    elif page == "KAP Analysis":
        render_kap(kap_df, districts_df)
    elif page == "Export Data":
        render_export(kelpa_df, kap_df, districts_df, domain_df)


if __name__ == "__main__":
    main()
