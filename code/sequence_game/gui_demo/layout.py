"""Dash layout for the local GUI demo."""

from __future__ import annotations

from dash import dcc, html
import dash_cytoscape as cyto

_XAI_MODELS = [
    "llama3.2:3b",
    "llama3.1:8b",
    "qwen3:8b",
    "huihui_ai/qwen3-abliterated:8b",
    "qwen3:14b",
    "phi4:14b",
    "huihui_ai/qwen3-abliterated:14b",
    "gemma4:26b",
    "gemma4:31b",
    "huihui_ai/qwen3-abliterated:32b",
    "qwq:32b",
    "huihui_ai/deepseek-r1-abliterated:70b",
]

from .demo_runner import (
    ACTION_SURFACE_MODELS,
    ALICE_ALGO_OPTIONS,
    BLOCKED_ATTACK_TOOLTIP,
    DISABLED_ATTACKS,
    EVE_ALGO_OPTIONS,
    REPEATER_ATTACK_OPTIONS,
)
from .graph_view import CYTOSCAPE_STYLESHEET

CONTROL_MODEL_OPTIONS = [
    {"label": value["label"], "value": key}
    for key, value in ACTION_SURFACE_MODELS.items()
] + [{"label": "repeater-backed availability/quality demo", "value": "repeater_availability_quality"}]
EVE_ALGO_DROPDOWN_OPTIONS = [
    {"label": v["label"], "value": k} for k, v in EVE_ALGO_OPTIONS.items()
]
ALICE_ALGO_DROPDOWN_OPTIONS = [
    {"label": v["label"], "value": k} for k, v in ALICE_ALGO_OPTIONS.items()
]
REPEATER_ATTACK_DROPDOWN_OPTIONS = [
    {"label": value["label"], "value": key}
    for key, value in REPEATER_ATTACK_OPTIONS.items()
]

_THEME_OPTIONS = [
    {"label": "🌙 Midnight", "value": "midnight"},
    {"label": "🌒 Dark",     "value": "dark"},
    {"label": "☀ Light",    "value": "light"},
    {"label": "◉ Accessible HC", "value": "hc"},
    {"label": "⚙ System",   "value": "system"},
]


def _field(label, child, class_name: str = "field"):
    return html.Label([html.Span(label, className="field-label"), child], className=class_name)


def build_layout():
    return html.Div([
        # ── Stores & intervals ────────────────────────────────────────────────
        dcc.Store(id="topology-store"),
        dcc.Store(id="selection-store", data={"alice": None, "bob": None}),
        dcc.Store(id="routes-store", data=[]),
        dcc.Store(id="run-store"),
        dcc.Store(id="replay-index-store", data=0),
        dcc.Store(id="xai-store"),
        dcc.Store(id="xai-context-store"),   # last explanation for mitigation
        dcc.Store(id="xai-cache-store", data={}),  # keyed cache of explanations
        dcc.Store(id="theme-store", data="midnight"),
        dcc.Store(
            id="attack-bundle-store",
            data=ACTION_SURFACE_MODELS["binary_collision"]["attack_kinds"],
        ),
        dcc.Interval(id="replay-timer", interval=700, disabled=True),
        dcc.Interval(id="progress-ticker", interval=400, disabled=False),

        # ── Top bar ───────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.H1("Quantum-network strategy demo", className="app-title"),
                html.Div("scope-labelled control GUI; not a scientific result",
                         className="app-subtitle"),
            ], className="title-block"),
            html.Div([
                html.Button("Generate topology", id="generate-btn", n_clicks=0,
                            className="primary-btn"),
                _field("seed", dcc.Input(
                    id="seed-input", type="number", value=1, min=0, step=1,
                    debounce=True, persistence=True), "top-field"),
                _field("nodes", dcc.Input(
                    id="num-nodes-input", type="number", value=8, min=2, max=50, step=1,
                    debounce=True, persistence=True), "top-field"),
                _field("edge p", dcc.Input(
                    id="edge-prob-input", type="number", value=0.35, min=0, max=1, step=0.01,
                    debounce=True, persistence=True), "top-field"),
                _field("lengths", dcc.Dropdown(
                    id="edge-length-mode", value="unit", clearable=False, persistence=True,
                    options=[{"label": "unit", "value": "unit"},
                             {"label": "random length", "value": "random"}],
                ), "top-field"),
                html.Button("Reset", id="reset-btn", n_clicks=0),
                _field("theme", dcc.Dropdown(
                    id="theme-select",
                    options=_THEME_OPTIONS,
                    value="midnight",
                    clearable=False,
                    persistence=True,
                    className="theme-select",
                ), "top-field"),
            ], className="top-controls"),
        ], className="top-bar"),

        # ── Main content ──────────────────────────────────────────────────────
        html.Div([
            html.Aside([
                html.Div(id="warning-box", className="warning-box"),
                html.Div([
                    html.H2("Strategy controls", className="card-title"),
                    _field("Eve algorithm", dcc.Dropdown(
                        id="eve-algo-select", value="q_learning", clearable=False,
                        persistence=True, options=EVE_ALGO_DROPDOWN_OPTIONS,
                    )),
                    _field("Alice algorithm", dcc.Dropdown(
                        id="alice-algo-select", value="cautious_greedy", clearable=False,
                        persistence=True, options=ALICE_ALGO_DROPDOWN_OPTIONS,
                    )),
                    _field("model", dcc.Dropdown(
                        id="mode-select", value="binary_collision", clearable=False,
                        persistence=True, options=CONTROL_MODEL_OPTIONS,
                    )),
                    html.Div([
                        _field("steps", dcc.Dropdown(
                            id="steps-select", value=3000, clearable=False,
                            options=[{"label": str(v), "value": v}
                                     for v in (500, 1000, 3000, 5000)],
                        )),
                        _field("replay window", dcc.Input(
                            id="final-window-input", type="number", value=20,
                            min=1, max=100, step=1, debounce=True)),
                    ], className="two-col"),
                    html.Div([
                        _field("avoid horizon", dcc.Input(
                            id="avoidance-horizon-input", type="number", value=2,
                            min=0, max=20, step=1, debounce=True)),
                        _field("attack cost", dcc.Input(
                            id="attack-cost-input", type="number", value=0.05,
                            min=0, max=1, step=0.01, debounce=True)),
                    ], className="two-col"),
                ], id="strategy-card", className="panel-card"),

                html.Div([
                    html.H2("Action surface", className="card-title"),
                    html.Div("The selected model fixes Eve's action bundle and reward target.",
                             className="card-note"),
                    html.Div(
                        f"Unsupported attacks remain blocked: {BLOCKED_ATTACK_TOOLTIP}",
                        className="card-note blocked-note",
                    ),
                    html.Div(id="attack-bundle-status", className="bundle-status"),
                ], id="attack-bundle-card", className="panel-card"),

                html.Div([
                    html.H2("Repeater-backed path", className="card-title"),
                    html.Div(
                        "Visible only in repeater mode: select one simple "
                        "Alice-repeaters-Bob path. Arbitrary full-graph execution remains blocked.",
                        className="card-note",
                    ),
                    _field("selected repeater path", dcc.Dropdown(
                        id="selected-route-dropdown", clearable=False)),
                    _field("repeater attack", dcc.Dropdown(
                        id="repeater-attack", value="repeater_memory_measure_X",
                        clearable=False, options=REPEATER_ATTACK_DROPDOWN_OPTIONS,
                    )),
                ], id="repeater-controls", className="panel-card repeater-controls"),

                html.Div([
                    html.H2("Disabled / blocked", className="card-title"),
                    html.Ul([
                        html.Li(item["label"], title=item["tooltip"])
                        for item in DISABLED_ATTACKS
                    ]),
                ], className="panel-card disabled-attacks"),

                html.Div(id="run-preview", className="run-preview"),
                # Progress display OUTSIDE dcc.Loading so it's never masked by the spinner
                html.Div(id="run-progress-display", className="run-progress-display"),
                dcc.Loading(
                    html.Div([
                        html.Button("Run", id="run-btn", n_clicks=0, className="run-btn"),
                        html.Div(id="run-status"),
                    ]),
                    type="circle",
                    color="var(--accent, #41d19a)",
                    delay_show=200,
                ),
            ], className="left-panel"),

            html.Main([
                # ── Network ────────────────────────────────────────────────────
                html.Div([
                    html.Div([
                        html.H2("Network", className="card-title"),
                        html.Span(id="scope-label", className="scope-label"),
                    ], className="card-header"),
                    html.Div("Click a node once for Alice, then another for Bob.",
                             className="hint"),
                    cyto.Cytoscape(
                        id="graph-view", elements=[], layout={"name": "preset"},
                        stylesheet=CYTOSCAPE_STYLESHEET,
                        style={"width": "100%", "height": "100%"},
                        minZoom=0.6, maxZoom=2.0,
                    ),
                ], className="cp-network", id="cp-network"),

                html.Div([html.Div(className="drag-grip")],
                         className="drag-handle", id="drag-handle-1"),

                html.Div([
                    # Steps ─────────────────────────────────────────────────────
                    html.Div([
                        html.Div([
                            html.H2("Steps", className="card-title",
                                    style={"display": "inline", "marginRight": "8px"}),
                            html.Span(id="replay-step-label",
                                      style={"fontSize": "11px", "color": "var(--muted)"}),
                        ]),
                        html.Div([
                            html.Button("◀◀", id="replay-prev", n_clicks=0,
                                        title="previous step"),
                            html.Button("▶ play", id="replay-toggle", n_clicks=0,
                                        title="play / pause"),
                            html.Button("▶▶", id="replay-next", n_clicks=0,
                                        title="next step"),
                            html.Div(
                                dcc.Slider(
                                    id="replay-slider", min=0, max=0, step=1, value=0,
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                                className="replay-slider-wrapper",
                            ),
                        ], className="replay-strip"),
                    ], className="cp-steps", id="cp-steps"),

                    # XAI ────────────────────────────────────────────────────────
                    html.Div([
                        # Main header row
                        html.Div([
                            html.H2("XAI", className="card-title",
                                    style={"display": "inline", "marginRight": "12px"}),
                            dcc.Dropdown(
                                id="xai-model-select",
                                options=[{"label": m, "value": m} for m in _XAI_MODELS],
                                value="llama3.1:8b",
                                clearable=False,
                                style={"width": "160px", "display": "inline-block",
                                       "verticalAlign": "middle"},
                            ),
                            html.Button("Explain step", id="xai-explain-btn", n_clicks=0,
                                        className="primary-btn",
                                        style={"padding": "5px 12px", "fontSize": "12px",
                                               "marginLeft": "8px"}),
                            html.Button("Strategy", id="xai-strategy-btn", n_clicks=0,
                                        className="secondary-btn",
                                        style={"padding": "5px 12px", "fontSize": "12px",
                                               "marginLeft": "4px"}),
                            html.Button("↺", id="xai-regen-btn", n_clicks=0,
                                        className="secondary-btn", disabled=True,
                                        title="Regenerate (ignore cached explanation)",
                                        style={"padding": "5px 8px", "fontSize": "12px",
                                               "marginLeft": "4px"}),
                            html.Span(id="xai-scores",
                                      style={"marginLeft": "12px", "fontSize": "12px",
                                             "color": "var(--muted)"}),
                        ], style={"display": "flex", "alignItems": "center",
                                  "gap": "4px", "flexWrap": "wrap", "marginBottom": "6px"}),
                        # Mitigation row
                        html.Div([
                            html.Button("Mitigate", id="xai-mitigate-btn", n_clicks=0,
                                        className="mitigation-btn", disabled=True,
                                        title="Get defensive mitigation advice for this attack",
                                        style={"padding": "4px 12px", "fontSize": "11px"}),
                            html.Span("Run an explanation first.",
                                      id="xai-mitigate-hint",
                                      style={"marginLeft": "8px", "fontSize": "11px",
                                             "color": "var(--muted)"}),
                        ], style={"marginBottom": "8px"}),
                        html.Div(
                            id="xai-output",
                            children=[dcc.Markdown(
                                "Run a simulation, scrub to a step, then click **Explain step**.",
                                className="muted",
                            )],
                            className="xai-output-body",
                        ),
                    ], className="cp-xai", id="cp-xai"),
                ], className="cp-bottom", id="cp-bottom"),
            ], className="center-panel"),

            html.Aside([
                html.Div(id="eve-telemetry", className="telemetry-card"),
                html.Div(id="alice-telemetry", className="telemetry-card"),
            ], className="right-panel"),
        ], className="content-grid"),

        html.Div([
            html.Div(id="topology-summary", className="detail-card"),
            html.Div(id="routes-summary", className="detail-card"),
            html.Div(id="step-details", className="detail-card"),
            html.Div(id="metrics-summary", className="detail-card"),
        ], className="details-grid"),
    ], className="app-shell")


APP_CSS = """
/* ══════════════════════════════════════════════════════════════════════
   5-theme system via data-theme attribute on <html>.
   Themes: midnight (default) | dark | light | hc | system (JS-handled)
   ══════════════════════════════════════════════════════════════════════ */

/* ── 1. Midnight (deep navy dark — default) ───────────────────────── */
:root {
  color-scheme: dark;
  --bg:              #06080d;
  --panel:           #111722;
  --panel-soft:      #151d2a;
  --panel-strong:    #1b2533;
  --topbar-bg:       #090d14;
  --border:          #2b3647;
  --border-strong:   #3f5068;
  --text:            #d7deea;
  --muted:           #9aa8bb;
  --accent:          #41d19a;
  --accent-dim:      #12382c;
  --accent-border:   #1f8d68;
  --accent-text:     #d8ffee;
  --blue:            #5b9dff;
  --red:             #ff6262;
  --amber:           #f7c766;
  --secondary-bg:    #1a1f2e;
  --secondary-border:#3a5080;
  --secondary-text:  #b0ccee;
  --mitigation-bg:   #2d1e00;
  --mitigation-border:#8c5e00;
  --mitigation-text: #ffe0a0;
  --input-bg:        #151d2a;
  --shadow:          rgba(0,0,0,0.25);
}

/* ── 2. Dark (GitHub Dark-like medium dark) ───────────────────────── */
[data-theme="dark"] {
  color-scheme: dark;
  --bg:              #0d1117;
  --panel:           #161b22;
  --panel-soft:      #1c2128;
  --panel-strong:    #22272e;
  --topbar-bg:       #010409;
  --border:          #30363d;
  --border-strong:   #484f58;
  --text:            #e6edf3;
  --muted:           #8b949e;
  --accent:          #3fb950;
  --accent-dim:      #0d2e16;
  --accent-border:   #238636;
  --accent-text:     #d2fce8;
  --blue:            #58a6ff;
  --red:             #f85149;
  --amber:           #d29922;
  --secondary-bg:    #1c2128;
  --secondary-border:#373e47;
  --secondary-text:  #8b949e;
  --mitigation-bg:   #2a1800;
  --mitigation-border:#9e6a03;
  --mitigation-text: #f0c000;
  --input-bg:        #1c2128;
  --shadow:          rgba(0,0,0,0.3);
}

/* ── 3. Light (clean GitHub-Light-like) ──────────────────────────── */
[data-theme="light"] {
  color-scheme: light;
  --bg:              #f6f8fa;
  --panel:           #ffffff;
  --panel-soft:      #f6f8fa;
  --panel-strong:    #eaeef2;
  --topbar-bg:       #24292f;
  --border:          #d0d7de;
  --border-strong:   #8c959f;
  --text:            #1f2328;
  --muted:           #59636e;
  --accent:          #1a7f37;
  --accent-dim:      #dafbe1;
  --accent-border:   #1a7f37;
  --accent-text:     #0a2e12;
  --blue:            #0969da;
  --red:             #cf222e;
  --amber:           #9a6700;
  --secondary-bg:    #eaeef2;
  --secondary-border:#8c959f;
  --secondary-text:  #1f2328;
  --mitigation-bg:   #fff8c5;
  --mitigation-border:#9a6700;
  --mitigation-text: #4a3000;
  --input-bg:        #ffffff;
  --shadow:          rgba(31,35,40,0.12);
}
/* Light mode topbar keeps dark nav feel; text inside inverted */
[data-theme="light"] .top-bar { color: #f0f6fc; }
[data-theme="light"] .top-bar .field-label { color: #b0bac3; }
[data-theme="light"] .app-title { color: #f0f6fc; }
[data-theme="light"] .app-subtitle { color: #9aa8bb; }

/* ── 4. HC — Accessible, colorblind-safe (Okabe-Ito + high contrast) */
/* Deuteranopia/Protanopia safe: no red/green ambiguity.             */
/* Black-on-white with Okabe-Ito accent palette (orange, blue, etc.) */
[data-theme="hc"] {
  color-scheme: light;
  --bg:              #ffffff;
  --panel:           #f5f5f5;
  --panel-soft:      #eeeeee;
  --panel-strong:    #dcdcdc;
  --topbar-bg:       #1a1a1a;
  --border:          #888888;
  --border-strong:   #333333;
  --text:            #000000;
  --muted:           #444444;
  /* Okabe-Ito palette */
  --accent:          #009E73;  /* bluish green */
  --accent-dim:      #d0f4e8;
  --accent-border:   #007a58;
  --accent-text:     #003320;
  --blue:            #0072B2;  /* blue */
  --red:             #D55E00;  /* vermilion (NOT red — avoids R/G confusion) */
  --amber:           #E69F00;  /* orange */
  --secondary-bg:    #e4eefa;
  --secondary-border:#0072B2;
  --secondary-text:  #003366;
  --mitigation-bg:   #fff4d0;
  --mitigation-border:#E69F00;
  --mitigation-text: #4a2a00;
  --input-bg:        #ffffff;
  --shadow:          rgba(0,0,0,0.18);
}
[data-theme="hc"] .top-bar { color: #f0f0f0; }
[data-theme="hc"] .top-bar .field-label { color: #b0b0b0; }
[data-theme="hc"] .app-title { color: #f0f0f0; }
[data-theme="hc"] .app-subtitle { color: #aaaaaa; }

/* ── Base ─────────────────────────────────────────────────────────── */
body {
  margin: 0;
  font-family: "Hack", "Hack Nerd Font Mono", ui-monospace, SFMono-Regular,
               Menlo, Consolas, monospace;
  background: var(--bg);
  color: var(--text);
}
.app-shell { min-height: 100vh; }

/* ── Top bar ──────────────────────────────────────────────────────── */
.top-bar {
  display: grid;
  grid-template-columns: minmax(280px, 1fr) auto;
  gap: 18px;
  align-items: end;
  padding: 12px 16px;
  background: var(--topbar-bg);
  border-bottom: 1px solid var(--border);
}
.app-title    { margin: 0; font-size: 19px; font-weight: 700; }
.app-subtitle { margin-top: 4px; color: var(--muted); font-size: 12px; }
.top-controls {
  display: grid;
  grid-template-columns: 155px 100px 100px 115px 165px 78px 160px;
  gap: 9px;
  align-items: end;
}
/* Theme selector: compact inside the control bar */
.theme-select.dash-dropdown { min-height: unset; }
.theme-select .Select-control { padding: 2px 6px !important; min-height: unset !important; }
.theme-select .Select-value { line-height: 1.6 !important; }

/* ── Content grid ─────────────────────────────────────────────────── */
.content-grid {
  display: grid;
  grid-template-columns: 310px minmax(470px, 1fr) 390px;
  gap: 12px;
  height: calc(100vh - 78px);
  min-height: 610px;
  padding: 12px;
  box-sizing: border-box;
}
.left-panel, .center-panel, .right-panel { min-width: 0; min-height: 0; }
.left-panel { display: flex; flex-direction: column; gap: 10px; overflow: auto; }

/* ── Center panel ─────────────────────────────────────────────────── */
.center-panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  box-shadow: 0 12px 30px var(--shadow);
  display: flex; flex-direction: column;
  overflow: hidden; min-width: 0; min-height: 0;
}
.cp-network {
  padding: 12px; overflow: hidden;
  flex: 3; min-height: 100px;
  box-sizing: border-box; display: flex; flex-direction: column;
}
.cp-bottom { flex: 2; display: flex; flex-direction: column; overflow: hidden; min-height: 0; }
.cp-steps  { padding: 10px 12px; flex-shrink: 0; }
.cp-xai    { padding: 12px; flex: 1; overflow: auto; min-height: 120px; }
.xai-output-body { white-space: pre-wrap; line-height: 1.55; font-size: 13px; }
/* Markdown inside XAI output */
.xai-output-body p    { margin: 0 0 6px; }
.xai-output-body ul   { margin: 4px 0 6px; padding-left: 20px; }
.xai-output-body li   { margin-bottom: 2px; }
.xai-output-body code { background: var(--panel-strong); padding: 1px 4px; border-radius: 3px; font-size: 12px; }
.xai-output-body strong { color: var(--text); }

/* ── Drag handle ──────────────────────────────────────────────────── */
.drag-handle {
  height: 8px; background: var(--border); cursor: row-resize;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; user-select: none; transition: background 0.15s;
}
.drag-handle:hover, .drag-handle.dragging { background: var(--border-strong); }
.drag-grip { width: 28px; height: 2px; border-radius: 1px; background: var(--muted); position: relative; }
.drag-grip::before, .drag-grip::after {
  content: ''; position: absolute; left: 0; width: 100%;
  height: 2px; background: var(--muted); border-radius: 1px;
}
.drag-grip::before { top: -3px; }
.drag-grip::after  { top:  3px; }

/* ── Right panel ──────────────────────────────────────────────────── */
.right-panel { display: grid; grid-template-rows: 1fr 1fr; gap: 12px; overflow: hidden; }

/* ── Cards ────────────────────────────────────────────────────────── */
.panel-card, .graph-card, .telemetry-card, .detail-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  box-shadow: 0 12px 30px var(--shadow);
}
.panel-card  { padding: 12px; }
.graph-card  { width: 100%; min-width: 0; padding: 12px; display: flex; flex-direction: column; }
.telemetry-card { overflow: auto; padding: 12px; }
.card-header { display: flex; justify-content: space-between; gap: 10px; align-items: center; }
.card-title  { margin: 0 0 9px; font-size: 14px; font-weight: 700; color: var(--text); }
.card-note, .hint, .muted, .bundle-status {
  color: var(--muted); font-size: 12px; line-height: 1.35;
}
.field, .top-field { display: block; margin: 8px 0; font-size: 12px; }
.top-field   { margin: 0; }
.field-label { display: block; color: var(--muted); font-weight: 700; margin-bottom: 4px; }
.two-col     { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }

/* ── Inputs ───────────────────────────────────────────────────────── */
/* ── Dash 4.x dcc.Input structure ──────────────────────────────────
   type="number" wraps: .dash-input-container > .dash-input-element
                                                + .dash-input-stepper × 2
   type="text"   renders: <input class="dash-input"> (no container)
   We target both structures plus the plain <input> element.           */

/* Container (wraps element + stepper buttons) */
.dash-input-container {
  background: var(--input-bg) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 6px !important;
  box-sizing: border-box !important;
  display: flex !important;
  align-items: stretch !important;
  width: 100% !important;
}
/* Actual <input> inside the container (text, number) */
.dash-input-element {
  background: transparent !important;
  color: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 8px 9px !important;
  font-family: inherit !important;
  font-size: inherit !important;
  flex: 1 !important;
  min-width: 0 !important;
}
.dash-input-element::placeholder { color: var(--muted) !important; opacity: 1 !important; }
/* + / − stepper buttons */
.dash-input-stepper {
  background: var(--panel-strong) !important;
  color: var(--text) !important;
  border: none !important;
  border-left: 1px solid var(--border) !important;
  border-radius: 0 !important;
  padding: 0 7px !important;
  cursor: pointer !important;
  font-size: 13px !important;
  line-height: 1 !important;
  white-space: nowrap !important;
  flex-shrink: 0 !important;
}
.dash-input-stepper:last-child { border-radius: 0 5px 5px 0 !important; }
.dash-input-stepper:hover { background: var(--border) !important; }
.dash-input-stepper:disabled { opacity: 0.4 !important; cursor: not-allowed !important; }

/* Fallback: plain <input> and legacy .dash-input (non-container path) */
input,
.dash-input,
input.dash-input {
  box-sizing: border-box;
  border: 1px solid var(--border-strong) !important;
  border-radius: 6px !important;
  padding: 8px 9px !important;
  font-family: inherit !important;
  color: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important;
  background: var(--input-bg) !important;
  background-color: var(--input-bg) !important;
  width: 100%;
}
input::placeholder,
.dash-input::placeholder { color: var(--muted) !important; opacity: 1 !important; }
/* Hide browser-native spinners (Dash 4.x custom steppers replace them) */
input[type=number]::-webkit-outer-spin-button,
input[type=number]::-webkit-inner-spin-button { -webkit-appearance: none !important; margin: 0; }
input[type=number],
.dash-input[type=number],
.dash-input-element { -moz-appearance: textfield !important; }

/* ── Dropdowns (react-select v2 via Dash) ─────────────────────────── */
.dash-dropdown,
.Select,
.Select-control,
.Select-menu,
.Select-menu-outer,
.Select-value,
.Select-placeholder,
.Select-input {
  background: var(--input-bg) !important;
  color: var(--text) !important;
  border-color: var(--border-strong) !important;
}
.Select-control { border-radius: 6px !important; box-shadow: none !important; }
.Select-value-label, .Select-placeholder { color: var(--text) !important; }
/* Search input inside dropdown */
.Select-input > input,
.is-searchable .Select-input > input {
  background: transparent !important;
  color: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important;
  caret-color: var(--accent) !important;
  opacity: 1 !important;
}
.Select-input > input::placeholder { color: var(--muted) !important; opacity: 1 !important; }
/* Option list */
.is-open .Select-menu-outer,
.Select-menu-outer {
  background: var(--input-bg) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 6px !important;
  overflow: hidden;
}
.Select-option {
  background: var(--input-bg) !important;
  color: var(--text) !important;
}
.VirtualizedSelectOption {
  background: var(--input-bg) !important;
  color: var(--text) !important;
}
.VirtualizedSelectFocusedOption,
.Select-option.is-focused {
  background: var(--panel-strong) !important;
  color: var(--text) !important;
}
.Select-option.is-selected {
  background: var(--accent-dim) !important;
  color: var(--accent-text) !important;
}
.Select-arrow { border-top-color: var(--muted) !important; }
.Select-clear-zone { color: var(--muted) !important; }
/* Autofill override (WebKit) */
input:-webkit-autofill {
  -webkit-box-shadow: 0 0 0 30px var(--input-bg) inset !important;
  -webkit-text-fill-color: var(--text) !important;
}

/* ── List elements (browser default is white) ────────────────────── */
ul, li { background: transparent !important; }

/* ── Buttons ──────────────────────────────────────────────────────── */
button {
  border: 1px solid var(--border-strong);
  border-radius: 6px;
  background: var(--panel-soft);
  color: var(--text);
  padding: 8px 10px;
  cursor: pointer;
  font-family: inherit;
  white-space: nowrap;
  overflow: hidden;
}
button:hover:not(:disabled) { border-color: var(--blue); background: var(--panel-strong); }
button:disabled              { opacity: 0.45; cursor: not-allowed; }
.primary-btn, .run-btn {
  background: var(--accent-dim);
  border-color: var(--accent-border);
  color: var(--accent-text);
  font-weight: 700;
}
.primary-btn:hover:not(:disabled), .run-btn:hover:not(:disabled) {
  filter: brightness(1.12);
  border-color: var(--accent);
}
.secondary-btn {
  background: var(--secondary-bg);
  border-color: var(--secondary-border);
  color: var(--secondary-text);
  font-weight: 600;
}
.secondary-btn:hover:not(:disabled) { border-color: var(--blue); background: var(--panel-strong); }
.mitigation-btn {
  background: var(--mitigation-bg);
  border-color: var(--mitigation-border);
  color: var(--mitigation-text);
  font-weight: 600;
}
.mitigation-btn:hover:not(:disabled) { filter: brightness(1.12); }
.run-btn { width: 100%; padding: 11px; }

/* ── Warning / status ─────────────────────────────────────────────── */
.warning-box {
  min-height: 24px;
  color: #ffe4aa; background: #2a2110; border: 1px solid #755b20;
  padding: 8px; border-radius: 8px; font-size: 12px;
}
[data-theme="light"] .warning-box,
[data-theme="hc"]    .warning-box {
  color: #6b3c00; background: #fff8e0; border-color: #cc9900;
}
.warning-box ul { margin: 0; padding-left: 18px; }
.blocked-note { margin-top: 8px; }
.repeater-controls { background: var(--panel-strong); }
.disabled-attacks ul { margin: 0; padding-left: 18px; color: var(--muted); font-size: 12px; }
.run-preview { font-size: 11px; color: var(--muted); padding: 4px 0; font-style: italic; }
.run-progress-display {
  font-size: 11px; color: var(--accent); min-height: 18px; padding: 3px 0 2px;
  font-family: inherit;
}
.run-status { min-height: 32px; color: var(--muted); font-size: 12px; line-height: 1.35; }
.status-error, .status-ok { border-radius: 8px; padding: 8px; display: block; }
.status-error { color: #ffd6d6; background: #3a171a; border: 1px solid #7b3035; }
[data-theme="light"] .status-error,
[data-theme="hc"]    .status-error { color: #800000; background: #ffe8e8; border-color: var(--red); }
.status-ok { color: var(--accent-text); background: var(--accent-dim); border: 1px solid var(--accent-border); }
.scope-label {
  color: var(--accent-text); background: var(--accent-dim);
  border: 1px solid var(--accent-border); padding: 4px 8px;
  border-radius: 999px; font-size: 11px; white-space: nowrap;
}

/* ── Replay strip ─────────────────────────────────────────────────── */
.replay-strip {
  display: grid;
  grid-template-columns: 44px 92px 44px 1fr;
  gap: 14px;
  align-items: center;
  margin-top: 8px;
}
/* Fixed-height buttons prevent ⏸ pause from wrapping */
.replay-strip button {
  padding: 0 6px;
  height: 28px;
  font-size: 11px;
  display: flex;
  align-items: center;
  justify-content: center;
}
/* Slider wrapper: ensures rc-slider sits at vertical midpoint */
.replay-slider-wrapper {
  display: flex;
  align-items: center;
  height: 28px;     /* match button height so slider is centered in row */
  padding: 0 2px;
}
.replay-slider-wrapper .dash-slider,
.replay-slider-wrapper .rc-slider { flex: 1; margin: 0; }

/* ── RC Slider ────────────────────────────────────────────────────── */
.rc-slider { padding: 0 !important; }
.rc-slider-rail {
  background-color: var(--border-strong) !important;
  height: 4px !important; border-radius: 2px !important;
}
.rc-slider-track {
  background-color: var(--accent) !important;
  height: 4px !important; border-radius: 2px !important;
}
.rc-slider-dot { border-color: var(--border-strong) !important; background: var(--panel-soft) !important; }
.rc-slider-handle {
  background-color: var(--panel) !important;
  border: 2px solid var(--accent) !important;
  width: 14px !important; height: 14px !important;
  margin-top: -5px !important;
  box-shadow: 0 0 0 2px rgba(65,209,154,0.18) !important;
  opacity: 1 !important;
}
.rc-slider-handle:hover, .rc-slider-handle:active { border-color: var(--blue) !important; }
/* Dash 4.x slider tooltip uses .tooltip / .tooltip-bounding-box (not rc-slider-tooltip-inner) */
.tooltip-bounding-box { position: relative !important; }
.tooltip,
.tooltip.always-visible {
  background: var(--panel-strong) !important;
  color: var(--text) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 4px !important;
  padding: 2px 6px !important;
  font-size: 11px !important;
  font-family: inherit !important;
  white-space: nowrap !important;
}
/* Legacy rc-slider tooltip (kept for backward compat) */
.rc-slider-tooltip-inner {
  background: var(--panel-strong) !important; color: var(--text) !important;
  border: 1px solid var(--border-strong) !important;
}

/* ── Data tables ──────────────────────────────────────────────────── */
.telemetry-title    { margin: 0 0 8px; font-size: 15px; }
.telemetry-subtitle { color: var(--muted); font-size: 11px; margin-bottom: 8px; }
.data-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.data-table th { text-align: left; background: var(--panel-strong); color: var(--text); font-weight: 700; }
.data-table th, .data-table td { border: 1px solid var(--border); padding: 5px 6px; vertical-align: top; }

/* ── Attention animation ──────────────────────────────────────────── */
@keyframes attentionBlink {
  0%, 100% { border-color: var(--red); box-shadow: 0 0 0 1px rgba(255,98,98,0.18); }
  50%       { border-color: #7b3035;   box-shadow: none; }
}
.attention-card {
  border-style: dashed !important; border-color: var(--red) !important;
  outline: 1px dotted var(--red); animation: attentionBlink 1.25s steps(2, end) infinite;
}
.attention-input .Select-control,
.attention-input input { border-style: dashed !important; border-color: var(--red) !important; }

/* ── Details grid ─────────────────────────────────────────────────── */
.details-grid {
  display: grid; grid-template-columns: repeat(4, minmax(240px, 1fr));
  gap: 12px; padding: 0 12px 14px;
}
.detail-card { padding: 12px; max-height: 360px; overflow: auto; }
.mini-heading { margin: 10px 0 5px; font-size: 12px; color: var(--text); }

/* ── Responsive fallback ──────────────────────────────────────────── */
@media (max-width: 1180px) {
  .top-bar, .content-grid, .details-grid { display: block; height: auto; }
  .top-controls, .right-panel { display: grid; margin-top: 10px; }
  .content-grid > * { margin-bottom: 12px; }
}
"""
