"""Dash entry point for the local quantum-network strategy GUI demo."""

from __future__ import annotations

import argparse

from dash import Dash

from .callbacks import register_callbacks
from .layout import APP_CSS, build_layout


def create_app() -> Dash:
    """Build the Dash app without starting the development server."""

    app = Dash(
        __name__,
        title="Quantum-network strategy demo",
        suppress_callback_exceptions=False,
    )
    app.index_string = f"""<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>{APP_CSS}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
        <script>
        (function () {{
            var MIN_NET = 100, MIN_XAI = 120;

            function setup() {{
                var panel  = document.querySelector('.center-panel');
                var handle = document.getElementById('drag-handle-1');
                var net    = document.getElementById('cp-network');
                var bottom = document.getElementById('cp-bottom');
                var stp    = document.getElementById('cp-steps');
                var xai    = document.getElementById('cp-xai');
                if (!panel || !handle || !net || !bottom || !stp || !xai) {{
                    setTimeout(setup, 200);
                    return;
                }}

                /* Pin network to 60% of panel height; bottom takes the rest (flex:1) */
                var panelH = panel.offsetHeight;
                net.style.height = Math.round(panelH * 0.6) + 'px';
                net.style.flex   = 'none';
                bottom.style.flex = '1';

                var startY = 0, startNetH = 0;

                handle.addEventListener('mousedown', function (e) {{
                    startY    = e.clientY;
                    startNetH = net.offsetHeight;
                    handle.classList.add('dragging');
                    document.addEventListener('mousemove', onMove);
                    document.addEventListener('mouseup',   onUp);
                    e.preventDefault();
                }});

                function onMove(e) {{
                    var dy       = e.clientY - startY;
                    var panelH   = panel.offsetHeight;
                    var handleH  = handle.offsetHeight;
                    var stpH     = stp.offsetHeight;   /* static — never changes */
                    var minBot   = stpH + MIN_XAI;
                    var maxNetH  = panelH - handleH - minBot;
                    net.style.height = Math.min(maxNetH, Math.max(MIN_NET, startNetH + dy)) + 'px';
                }}

                function onUp() {{
                    handle.classList.remove('dragging');
                    document.removeEventListener('mousemove', onMove);
                    document.removeEventListener('mouseup',   onUp);
                }}
            }}

            setTimeout(setup, 400);
        }})();
        </script>
    </body>
</html>
"""
    app.layout = build_layout()
    register_callbacks(app)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Dash GUI demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8050, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    create_app().run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
