import time
from pathlib import Path

from streamlit_autorefresh import st_autorefresh

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud

# ── Paths & constants ─────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent / 'data' / 'survey_data.xlsx'

TEAMS = ['Innovation', 'Outcomes', 'Knowledge']

DARK_BG  = '#0B0F1A'
LIGHT_BG = '#E8F7F5'   # soft teal-mint, drawn from the QR card gradient

WORD_DARK = {
    'Innovation': ['#1B4F8A', '#2E86DE', '#74C0FC', '#BDE0FE'],
    'Outcomes':   ['#6B21A8', '#A855F7', '#D8B4FE', '#F3E8FF'],
    'Knowledge':  ['#166534', '#22C55E', '#86EFAC', '#DCFCE7'],
}
WORD_LIGHT = {
    'Innovation': ['#0D47A1', '#1565C0', '#1976D2', '#1E88E5'],
    'Outcomes':   ['#4A148C', '#6A1B9A', '#7B1FA2', '#8E24AA'],
    'Knowledge':  ['#1B5E20', '#2E7D32', '#388E3C', '#43A047'],
}
TEAM_TINTS = {
    'Innovation': '#DBF0F5',   # teal-washed blue
    'Outcomes':   '#E8EEF8',   # teal-washed lavender
    'Knowledge':  '#D8F5F0',   # teal-washed green
}
TEAM_ACCENTS = {
    'Innovation': '#2E86DE',
    'Outcomes':   '#A855F7',
    'Knowledge':  '#22C55E',
}

# ── Helper functions ──────────────────────────────────────────────────────────

def make_colormap(colors):
    return LinearSegmentedColormap.from_list('custom', colors)


def color_func_for(palette):
    cmap = make_colormap(palette)
    def cf(word, font_size, position, orientation, random_state=None, **kwargs):
        r, g, b, _ = cmap(random_state.uniform(0.2, 1.0))
        return f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
    return cf


def circular_mask(size=700, padding=12):
    y, x = np.ogrid[:size, :size]
    outside = (x - size//2)**2 + (y - size//2)**2 > (size//2 - padding)**2
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[outside] = 255
    return arr


def make_wc_dark(freq_dict, team, size=700, padding=12):
    return WordCloud(
        width=size, height=size,
        background_color=DARK_BG,
        mask=circular_mask(size, padding),
        max_words=50, prefer_horizontal=0.85,
        relative_scaling=0.55, min_font_size=16, max_font_size=120,
        color_func=color_func_for(WORD_DARK[team]),
        collocations=False, repeat=False,
    ).generate_from_frequencies(freq_dict)


def make_wc_light(freq_dict, team, size=700, padding=12):
    return WordCloud(
        width=size, height=size,
        background_color=None, mode='RGBA',
        mask=circular_mask(size, padding),
        max_words=50, prefer_horizontal=0.85,
        relative_scaling=0.55, min_font_size=16, max_font_size=120,
        color_func=color_func_for(WORD_LIGHT[team]),
        collocations=False, repeat=False,
    ).generate_from_frequencies(freq_dict)


def add_glow(ax, color):
    for r, a in [(0.52, 0.07), (0.51, 0.12), (0.50, 0.17)]:
        ax.add_patch(mpatches.Circle(
            (0.5, 0.5), r, transform=ax.transAxes,
            color=color, alpha=a, linewidth=0, zorder=1,
        ))


def make_radial_bg_pil(hex_color, size=700, padding=12, max_alpha=0.22):
    """PIL RGBA image: team tint at center fading to transparent at circle edge."""
    from matplotlib.colors import to_rgb
    from PIL import Image as PILImage
    r, g, b = to_rgb(hex_color)
    cx = cy = size // 2
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    circle_r = cx - padding     # matches circular_mask radius
    t = np.clip(dist / circle_r, 0, 1)
    alpha = ((1.0 - t) ** 2 * max_alpha * 255).clip(0, 255).astype(np.uint8)
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[..., 0] = int(r * 255)
    arr[..., 1] = int(g * 255)
    arr[..., 2] = int(b * 255)
    arr[..., 3] = alpha
    return PILImage.fromarray(arr, mode='RGBA')


def composite_wc(wc, hex_color, padding=12):
    """Alpha-composite the word cloud over a radial gradient background."""
    from PIL import Image as PILImage
    bg = make_radial_bg_pil(hex_color, size=wc.width, padding=padding)
    fg = wc.to_image().convert('RGBA')   # force RGBA regardless of wc mode
    return np.array(PILImage.alpha_composite(bg, fg))


def _reveal(state_key):
    st.session_state[state_key] = True


def reveal_placeholder(btn_key, state_key, height_px):
    """Button vertically centred in reserved blank space."""
    half = height_px // 2
    st.markdown(f'<div style="height:{half}px"></div>', unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        st.button('Reveal', key=btn_key, use_container_width=True,
                  on_click=_reveal, args=(state_key,))
    st.markdown(f'<div style="height:{half}px"></div>', unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────

Q_TEAM = 'Please select your team'
Q1_COL = 'If your department were a TV show, it would be'
Q2_COL = 'If your department had a theme song, it would be:'
Q3_COL = 'If the three departments formed a superhero team, what would our team name be?'

@st.cache_data(ttl=30)
def load_data():
    df = pd.read_excel(DATA_PATH, sheet_name='Sheet1')
    df = df.rename(columns={Q_TEAM: 'Team'})
    df['Q1_label'] = (
        df[Q1_COL].str.strip().str.lower().str.split().str.join(' ').str.title()
    )
    df['Q2_label'] = (
        df[Q2_COL].str.strip().str.lower().str.split().str.join(' ').str.title()
    )
    return df

# ── Figure builders ───────────────────────────────────────────────────────────

_WC_SIZE       = 700
_WC_PADDING    = 50   # words are constrained this far from the circle edge
_OUTLINE_PAD   = 12   # outline sits this far from the circle edge (gap = _WC_PADDING - _OUTLINE_PAD)


def build_triple_fig(df, col, dark):
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor('none')

    for ax, team in zip(axes, TEAMS):
        sub  = df[df['Team'] == team]
        freq = {k: v * 100 for k, v in sub[col].value_counts().items()}

        if not freq:
            ax.axis('off')
            continue

        ax.set_facecolor('none')
        if dark:
            wc = make_wc_dark(freq, team, size=_WC_SIZE, padding=_WC_PADDING)
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            add_glow(ax, TEAM_ACCENTS[team])
        else:
            wc = make_wc_light(freq, team, size=_WC_SIZE, padding=_WC_PADDING)
            ax.imshow(composite_wc(wc, TEAM_ACCENTS[team], _OUTLINE_PAD), interpolation='bilinear')
            ax.axis('off')

        # Faint circle outline near the outer edge
        outline = mpatches.Circle(
            (_WC_SIZE / 2, _WC_SIZE / 2), _WC_SIZE / 2 - _OUTLINE_PAD,
            fill=False,
            edgecolor=TEAM_ACCENTS[team],
            linewidth=1.0,
            alpha=0.12,
            zorder=5,
        )
        ax.add_patch(outline)

        ax.set_title(team, fontsize=15, fontweight='bold',
                     color=TEAM_ACCENTS[team], pad=10)

    plt.tight_layout(pad=2)
    return fig


def build_q3_fig(df, dark):

    raw = df[Q3_COL].dropna().str.strip()
    raw = raw[raw != '']
    if raw.empty:
        return None
    # Normalise case + whitespace so minor spelling differences group together
    raw = raw.str.lower().str.split().str.join(' ').str.title()
    q3_freq = raw.value_counts().to_dict()

    if dark:
        tri = make_colormap([
            '#2E86DE', '#74C0FC', '#A855F7', '#D8B4FE',
            '#22C55E', '#86EFAC', '#2E86DE',
        ])
        def tricolor(word, font_size, position, orientation, random_state=None, **kwargs):
            r, g, b, _ = tri(random_state.uniform(0, 1))
            return f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'

        wc = WordCloud(
            width=900, height=900,
            background_color=DARK_BG,
            mask=circular_mask(900),
            max_words=80, prefer_horizontal=0.75,
            relative_scaling=0.55, min_font_size=18, max_font_size=120,
            color_func=tricolor,
            collocations=False, repeat=False,
        ).generate_from_frequencies(q3_freq)
    else:
        tri = make_colormap([
            '#0D47A1', '#1976D2', '#6A1B9A', '#8E24AA',
            '#1B5E20', '#388E3C', '#0D47A1',
        ])
        def tricolor(word, font_size, position, orientation, random_state=None, **kwargs):
            r, g, b, _ = tri(random_state.uniform(0, 1))
            return f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'

        wc = WordCloud(
            width=900, height=900,
            background_color=None, mode='RGBA',
            mask=circular_mask(900),
            max_words=80, prefer_horizontal=0.75,
            relative_scaling=0.55, min_font_size=18, max_font_size=120,
            color_func=tricolor,
            collocations=False, repeat=False,
        ).generate_from_frequencies(q3_freq)

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    if dark:
        ax.imshow(wc, interpolation='bilinear')
        for radius, alpha, color in [
            (0.515, 0.07, '#2E86DE'),
            (0.508, 0.10, '#A855F7'),
            (0.500, 0.13, '#22C55E'),
        ]:
            ax.add_patch(mpatches.Circle(
                (0.5, 0.5), radius, transform=ax.transAxes,
                color=color, alpha=alpha, linewidth=0, zorder=0,
            ))
    else:
        ax.imshow(wc, interpolation='bilinear')

    ax.axis('off')
    plt.tight_layout()
    return fig

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='Survey Word Clouds',
    layout='wide',
    page_icon='☁️',
    initial_sidebar_state='collapsed',
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Page background — very light teal gradient echoing the QR card */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(150deg, #E6F7F6 0%, #EFF9F8 35%, #F5FBFA 70%, #F9FDFC 100%);
        background-attachment: fixed;
    }
    [data-testid="stHeader"] { background: transparent; }

    /* Hide Streamlit's own toolbar, deploy and manage-app controls */
    [data-testid="stToolbar"]          { display: none !important; }
    [data-testid="stDecoration"]       { display: none !important; }
    [data-testid="manage-app-button"]  { display: none !important; }
    .stDeployButton                    { display: none !important; }
    #MainMenu                          { display: none !important; }
    footer                             { display: none !important; }

    /* Sidebar tinted to match */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #D8F2F0 0%, #E6F7F5 100%);
    }

    /* Section headers */
    h2 { color: #0D5C5A !important; margin-top: 10rem !important; }

    /* Metadata line */
    .stMarkdown div[style*="text-align:right"] { color: #1B8B85 !important; }

    /* Hide disabled (just-clicked) reveal buttons during rerun */
    section[data-testid="stMain"] div[data-testid="stButton"] > button:disabled {
        display: none !important;
    }

    /* Reveal buttons — scoped to main content only, not sidebar or toolbar */
    section[data-testid="stMain"] div[data-testid="stButton"] > button {
        background-color: #1B9E9A;
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        border-radius: 0.5rem;
    }
    section[data-testid="stMain"] div[data-testid="stButton"] > button:hover {
        background-color: #157A77;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header image ──────────────────────────────────────────────────────────────
QR_PATH = Path(__file__).parent / 'QRCode for Horizontal Retreat - Icebreakers.png'
_, qr_col, _ = st.columns([1, 3, 1])
with qr_col:
    st.image(str(QR_PATH), use_container_width=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title('Settings')
    dark     = st.toggle('Dark theme', value=False)
    interval = st.slider('Refresh every (s)', min_value=10, max_value=300, value=30, step=10)

    if st.button('Refresh now', use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption('Word clouds regenerate automatically when new responses land in the Excel file.')

# ── Load data ─────────────────────────────────────────────────────────────────

df   = load_data()
n    = len(df)
last = time.strftime('%H:%M:%S')

st.markdown(
    f"<div style='text-align:right; color:gray; font-size:0.85rem;'>"
    f"Last updated: <b>{last}</b> &nbsp;|&nbsp; <b>{n}</b> responses"
    f"</div>",
    unsafe_allow_html=True,
)

# ── Sections ──────────────────────────────────────────────────────────────────

for key in ('show_q1', 'show_q2', 'show_q3'):
    if key not in st.session_state:
        st.session_state[key] = False

# Q1
st.markdown('## If your department were a TV show, it would be…')
if not st.session_state.show_q1:
    reveal_placeholder('btn_q1', 'show_q1', 460)
else:
    fig = build_triple_fig(df, 'Q1_label', dark)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# Q2
st.markdown('## If your department had a theme song, it would be…')
if not st.session_state.show_q2:
    reveal_placeholder('btn_q2', 'show_q2', 460)
else:
    fig = build_triple_fig(df, 'Q2_label', dark)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# Q3
st.markdown('## If the three departments formed a superhero team… what would our team name be?')
if not st.session_state.show_q3:
    reveal_placeholder('btn_q3', 'show_q3', 520)
else:
    fig = build_q3_fig(df, dark)
    if fig:
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info('No Q3 responses yet.')

# ── Auto-refresh ──────────────────────────────────────────────────────────────
# Browser-side timer triggers a rerun after `interval` seconds.
# The script itself completes immediately, so no widgets are left in a faded state.

st_autorefresh(interval=interval * 1000, key='data_refresh')
