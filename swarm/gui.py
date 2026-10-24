"""
Desktop GUI for the agent swarm (PyQt6).

Launch with:

    python -m swarm gui      (or double-click start_gui.bat)

It drives the exact same `Orchestrator` the CLI uses - no second code path -
so a run started here can be resumed from the CLI and vice versa.

Architecture
------------
* The orchestrator runs on a QThread (`RunWorker`); the UI thread never blocks.
* Log lines are pushed through a Qt signal (queued connection across threads).
* Live state (board, leaderboard, squad table, artefacts) is *polled* by a
  QTimer every 400 ms. Every shared structure in `swarm.state` is already
  lock-protected, so polling is safe and needs no extra plumbing.
* Charts are hand-painted with QPainter - no matplotlib, keeping the project
  standard-library-plus-Qt.

Pages: Run · Squads · Leaderboard · Board · Artefacts · History · Settings
"""
from __future__ import annotations

import html
import json
import os
import subprocess
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QObject, QThread, QTimer, QRectF, QSize, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QKeySequence, QPainter, QPen, \
    QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .config import (
    DAYTONA_API_KEY,
    DAYTONA_SANDBOX_ID,
    DAYTONA_SANDBOX_NAME,
    NVIDIA_API_KEY,
    SANDBOX_WORKSPACE,
    SETTINGS,
    Scale,
    Settings,
)
from .orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Palette / stylesheet
# ---------------------------------------------------------------------------
C = {
    "bg": "#0b0e14",
    "side": "#0e1220",
    "panel": "#131824",
    "line": "#1f2634",
    "fg": "#e6ebf5",
    "mut": "#8a94a8",
    "acc": "#5aa9ff",
    "good": "#3ddc97",
    "mid": "#f5c451",
    "low": "#ff6b6b",
    "violet": "#a970ff",
}

QSS = f"""
QMainWindow, QWidget#root {{
  background: {C['bg']}; color: {C['fg']};
  font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 13px;
}}
QFrame#sidebar {{ background: {C['side']}; border-right: 1px solid {C['line']}; }}
QLabel#brand {{ font-size: 15px; font-weight: 700; letter-spacing: 1.5px; color: {C['fg']}; }}
QLabel#brandsub {{ font-size: 10px; color: {C['mut']}; letter-spacing: 1px; }}
QPushButton#nav {{
  background: transparent; border: none; border-radius: 8px;
  color: {C['mut']}; text-align: left; padding: 9px 12px; font-size: 13px;
}}
QPushButton#nav:hover {{ background: #161c2c; color: {C['fg']}; }}
QPushButton#nav:checked {{ background: #18233a; color: {C['acc']}; font-weight: 600; }}
QFrame#card {{
  background: {C['panel']}; border: 1px solid {C['line']}; border-radius: 12px;
}}
QFrame#headerbar {{ background: transparent; border-bottom: 1px solid {C['line']}; }}
QLabel#h1 {{ font-size: 17px; font-weight: 700; color: {C['fg']}; }}
QLabel#sub {{ font-size: 12px; color: {C['mut']}; }}
QLabel#stat {{ font-size: 22px; font-weight: 700; color: {C['acc']}; }}
QLabel#statlabel {{ font-size: 10px; color: {C['mut']}; letter-spacing: 1px; }}
QLabel#lbl {{ color: {C['mut']}; font-size: 11px; letter-spacing: .6px; }}
QLineEdit, QPlainTextEdit, QSpinBox, QComboBox {{
  background: #0a0e18; border: 1px solid {C['line']}; border-radius: 7px;
  padding: 6px 9px; color: {C['fg']}; selection-background-color: {C['acc']};
  selection-color: #08111f;
}}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QComboBox:focus {{
  border-color: {C['acc']};
}}
QLineEdit:disabled, QSpinBox:disabled {{ color: {C['mut']}; }}
QSpinBox::up-button, QSpinBox::down-button {{ background: transparent; width: 16px; border: none; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: none; border-left: 4px solid transparent;
  border-right: 4px solid transparent; border-top: 5px solid {C['mut']}; margin-right: 8px; }}
QComboBox QAbstractItemView {{
  background: #0a0e18; border: 1px solid {C['line']}; selection-background-color: #18233a;
  outline: none; padding: 4px;
}}
QPushButton {{
  background: #1a2132; border: 1px solid {C['line']}; border-radius: 7px;
  padding: 7px 14px; color: {C['fg']}; font-weight: 600;
}}
QPushButton:hover {{ background: #212a40; border-color: #2b3752; }}
QPushButton:pressed {{ background: #171e2e; }}
QPushButton:disabled {{ color: #4d566b; background: #141a28; border-color: #1a2130; }}
QPushButton#primary {{ background: {C['acc']}; color: #07101f; border: none; }}
QPushButton#primary:hover {{ background: #7cbcff; }}
QPushButton#primary:disabled {{ background: #26324a; color: #6a7690; }}
QPushButton#danger {{ background: {C['low']}; color: #1a0606; border: none; }}
QPushButton#danger:hover {{ background: #ff8585; }}
QPushButton#danger:disabled {{ background: #3a2027; color: #7b5b60; }}
QPushButton#ghost {{ background: transparent; border: 1px solid {C['line']}; color: {C['mut']}; }}
QPushButton#ghost:hover {{ color: {C['fg']}; border-color: {C['acc']}; }}
QTableWidget, QTableView {{
  background: #0a0e18; border: 1px solid {C['line']}; border-radius: 8px;
  gridline-color: #161c2a; alternate-background-color: #0d1220;
  selection-background-color: #18233a; selection-color: {C['fg']};
}}
QHeaderView::section {{
  background: {C['panel']}; color: {C['mut']}; border: none;
  border-bottom: 1px solid {C['line']}; border-right: 1px solid #171d2b;
  padding: 7px 9px; font-size: 10.5px; letter-spacing: .8px; font-weight: 600;
}}
QTableWidget::item {{ padding: 4px; }}
QProgressBar {{
  background: #0a0e18; border: 1px solid {C['line']}; border-radius: 6px;
  text-align: center; color: {C['fg']}; font-size: 11px; min-height: 14px;
}}
QProgressBar::chunk {{ background: {C['good']}; border-radius: 5px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #232c42; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #232c42; border-radius: 5px; min-width: 30px; }}
QCheckBox {{ color: {C['fg']}; spacing: 7px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {C['line']};
  border-radius: 4px; background: #0a0e18; }}
QCheckBox::indicator:checked {{ background: {C['acc']}; border-color: {C['acc']}; }}
QComboBox#filter {{ padding: 4px 8px; }}
QScrollArea {{ border: none; background: transparent; }}
"""


def _mono() -> QFont:
    f = QFont()
    f.setFamilies(["Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono"])
    f.setPointSize(10)
    return f


def _score_color(v: float) -> str:
    return C["good"] if v >= 8 else (C["mid"] if v >= 6 else C["low"])


def _open_path(path: str) -> None:
    """Open a file or folder with the OS default handler (cross-platform)."""
    if hasattr(os, "startfile"):
        os.startfile(path)          # type: ignore[attr-defined]  # Windows
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def _icon() -> QIcon:
    """Small programmatic app icon: a node graph glyph."""
    pm = QPixmap(64, 64)
    pm.fill(QColor(C["bg"]))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(C["acc"]), 3))
    pts = [(14, 46), (32, 16), (50, 46)]
    p.drawLine(pts[0][0], pts[0][1], pts[1][0], pts[1][1])
    p.drawLine(pts[1][0], pts[1][1], pts[2][0], pts[2][1])
    p.drawLine(pts[0][0], pts[0][1], pts[2][0], pts[2][1])
    for i, (x, y) in enumerate(pts):
        p.setBrush(QColor(C["good"] if i == 1 else C["violet"]))
        p.setPen(QPen(QColor(C["bg"]), 2))
        p.drawEllipse(QRectF(x - 7, y - 7, 14, 14))
    p.end()
    return QIcon(pm)


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------
class StatCard(QFrame):
    def __init__(self, label: str, value: str = "0"):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)
        self.value = QLabel(value)
        self.value.setObjectName("stat")
        self.cap = QLabel(label.upper())
        self.cap.setObjectName("statlabel")
        lay.addWidget(self.value)
        lay.addWidget(self.cap)

    def set(self, value: str, color: Optional[str] = None) -> None:
        if self.value.text() != value:
            self.value.setText(value)
        if color:
            self.value.setStyleSheet(f"color: {color};")


class BarChart(QWidget):
    """Hand-painted horizontal bar chart (no third-party plotting libs)."""

    def __init__(self, empty: str = "no data yet", height: int = 170):
        super().__init__()
        self._data: List[Tuple[str, float, str]] = []
        self._empty = empty
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(self, data: List[Tuple[str, float, str]]) -> None:
        if data != self._data:
            self._data = data
            self.update()

    def paintEvent(self, _ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self._data:
            p.setPen(QColor(C["mut"]))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._empty)
            p.end()
            return

        w = self.width()
        h = self.height()
        rows = len(self._data)
        label_w = min(150, max(70, int(w * 0.28)))
        row_h = max(14, min(30, h // max(1, rows)))
        top = max(4, (h - row_h * rows) // 2)
        peak = max([v for _, v, _ in self._data] + [1e-9])
        font = _mono()
        font.setPointSize(8)
        p.setFont(font)

        for i, (label, value, color) in enumerate(self._data):
            y = top + i * row_h
            bar_h = max(6, int(row_h * 0.55))
            p.setPen(QColor(C["mut"]))
            p.drawText(QRectF(0, y, label_w - 8, row_h),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                       label[:22])
            track = QRectF(label_w, y + (row_h - bar_h) / 2, max(10, w - label_w - 58), bar_h)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#111726"))
            p.drawRoundedRect(track, 3, 3)
            frac = max(0.02, min(1.0, value / peak))
            fill = QRectF(track.x(), track.y(), track.width() * frac, track.height())
            p.setBrush(QColor(color))
            p.drawRoundedRect(fill, 3, 3)
            p.setPen(QColor(C["fg"]))
            p.drawText(QRectF(w - 54, y, 50, row_h),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                       f"{value:.1f}")
        p.end()


class Console(QPlainTextEdit):
    """Colour-coded, capped, auto-scrolling run console."""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setFont(_mono())
        self.document().setMaximumBlockCount(4000)
        self.setStyleSheet(
            f"QPlainTextEdit {{ background:#070a11; border:1px solid {C['line']};"
            f" border-radius:10px; padding:8px; }}"
        )
        self._follow = True

    def set_follow(self, on: bool) -> None:
        self._follow = on

    def write_line(self, text: str, color: Optional[str] = None) -> None:
        t = text.rstrip("\n")
        if not t:
            return
        if color is None:
            low = t.lower()
            if t.lstrip().startswith("!") or "error" in low or "fail" in low:
                color = C["low"]
            elif "verified" in low or "done in" in low or "ready" in low:
                color = C["good"]
            elif "budget" in low or "warn" in low:
                color = C["mid"]
            else:
                color = "#b9c3d6"
        self.appendHtml(f'<span style="color:{color}">{html.escape(t)}</span>')
        if self._follow:
            bar = self.verticalScrollBar()
            bar.setValue(bar.maximum())


# ---------------------------------------------------------------------------
# Orchestrator adapter + worker thread
# ---------------------------------------------------------------------------
class _LogBus(QObject):
    """Bridge from the orchestrator's log calls to Qt slots.

    `Orchestrator` is a plain Python class, so it cannot own a `pyqtSignal`
    itself; this tiny QObject lives in the GUI thread and forwards. Emitting
    from the worker thread is safe - Qt queues the delivery automatically.
    """

    line = pyqtSignal(str)


class LiveOrchestrator(Orchestrator):
    """Orchestrator whose console output is a Qt signal instead of stdout."""

    def __init__(self, settings=None, out_dir: Optional[str] = None):
        self.bus = _LogBus()          # before super(): __init__ may log
        super().__init__(settings, out_dir)

    @property
    def line(self):
        """The `line(str)` signal - kept named `line` for call-site symmetry."""
        return self.bus.line

    def _log(self, msg: str) -> None:  # noqa: D102
        self.bus.line.emit(str(msg))


class RunWorker(QThread):
    line = pyqtSignal(str)
    done = pyqtSignal(bool, str)            # success, message
    outputs = pyqtSignal(str, str, str)     # json, txt, html

    def __init__(self, orch: LiveOrchestrator, start_squad: int, out_dir: str,
                 parent=None):
        super().__init__(parent)
        self.orch = orch
        self.start_squad = start_squad
        self.out_dir = out_dir

    def run(self) -> None:  # noqa: D102
        t0 = time.time()
        success, message = True, ""
        try:
            self.orch.run(start_squad=self.start_squad)
            if self.orch._stop.is_set():
                success, message = False, "stopped by user"
        except Exception:
            success = False
            message = f"{traceback.format_exc(limit=4)}"
            self.line.emit(message)

        # Always write the artefacts, even after a crash or Ctrl-C style stop.
        paths = ("", "", "")
        try:
            paths = self._write_outputs(t0)
        except Exception as e:
            self.line.emit(f"! could not write outputs: {type(e).__name__}: {e}")

        self.outputs.emit(*paths)
        self.done.emit(success, message or "run complete")

    # -- reports ----------------------------------------------------------
    def _write_outputs(self, t0: float) -> Tuple[str, str, str]:
        report = self.orch.final_report()
        os.makedirs(self.out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        json_path = os.path.join(self.out_dir, f"swarm-report-{stamp}.json")
        txt_path = os.path.join(self.out_dir, f"swarm-report-{stamp}.txt")
        html_path = os.path.join(self.out_dir, f"swarm-dashboard-{stamp}.html")

        exported = self.orch.export_report()
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(exported, fh, indent=2, ensure_ascii=False)
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(report)

        try:
            from .dashboard import write as write_dashboard
            write_dashboard(exported, html_path)
        except Exception as e:
            self.line.emit(f"! dashboard failed: {e}")
            html_path = ""

        self.line.emit(f"\nwall time : {time.time() - t0:.1f}s")
        self.line.emit(f"json report: {json_path}")
        self.line.emit(f"txt report : {txt_path}")
        if html_path:
            self.line.emit(f"dashboard  : {html_path}")
        self.line.emit(f"checkpoint : {self.orch.checkpoint_path}")
        return json_path, txt_path, html_path


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str = "") -> QVBoxLayout:
    lay = QVBoxLayout()
    lay.setSpacing(2)
    h = QLabel(title)
    h.setObjectName("h1")
    lay.addWidget(h)
    if subtitle:
        s = QLabel(subtitle)
        s.setObjectName("sub")
        lay.addWidget(s)
    return lay


def labeled_field(label: str, widget: QWidget) -> QVBoxLayout:
    lay = QVBoxLayout()
    lay.setSpacing(4)
    l = QLabel(label.upper())
    l.setObjectName("lbl")
    lay.addWidget(l)
    lay.addWidget(widget)
    return lay


def make_table(columns: List[str],
               stretch_col: Optional[int] = None) -> QTableWidget:
    """Build a themed table.

    `stretch_col` names the one column that absorbs spare width; without it
    the last column does (Qt's default convenience flag), which makes short
    trailing columns like "State" or "Reviews" balloon out.
    """
    t = QTableWidget()
    t.setColumnCount(len(columns))
    t.setHorizontalHeaderLabels(columns)
    t.setAlternatingRowColors(True)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(26)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    if stretch_col is None:
        t.horizontalHeader().setStretchLastSection(True)
    else:
        t.horizontalHeader().setStretchLastSection(False)
        t.horizontalHeader().setSectionResizeMode(
            stretch_col, QHeaderView.ResizeMode.Stretch)
    t.setShowGrid(False)
    return t


def stat_row(cards: List[StatCard]) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(12)
    for c in cards:
        row.addWidget(c, 1)
    return row


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    POLL_MS = 400

    def __init__(self):
        super().__init__()
        # Apply the theme here (not only in main()) so every entry point -
        # including embedders and tests - gets the same look.
        app = QApplication.instance()
        if isinstance(app, QApplication) and not app.styleSheet():
            app.setStyleSheet(QSS)
        self.setWindowTitle("Agent Swarm — multi-agent orchestration console")
        self.setWindowIcon(_icon())
        self.resize(1460, 920)
        self.setMinimumSize(QSize(1120, 720))

        self.orch: Optional[LiveOrchestrator] = None
        self.worker: Optional[RunWorker] = None
        self.running = False
        self._run_started = 0.0
        self._board_seen = 0
        self._squads_sig = None
        self._leaders_sig = None
        self._art_seen = None
        self._last_dashboard = ""
        self._board_rows: List[Dict] = []
        self._proc: Optional[subprocess.Popen] = None

        self._build_ui()

        self.poll = QTimer(self)
        self.poll.setInterval(self.POLL_MS)
        self.poll.timeout.connect(self._tick)
        self.poll.start()

        self.clock = QTimer(self)
        self.clock.setInterval(1000)
        self.clock.timeout.connect(self._tick_clock)
        self.clock.start()

        self.refresh_history()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # -- sidebar ------------------------------------------------------
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(216)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(14, 18, 14, 16)
        sl.setSpacing(4)

        brand_box = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(_icon().pixmap(30, 30))
        txt = QVBoxLayout()
        txt.setSpacing(0)
        b = QLabel("AGENT SWARM")
        b.setObjectName("brand")
        sub = QLabel("ORCHESTRATION CONSOLE")
        sub.setObjectName("brandsub")
        txt.addWidget(b)
        txt.addWidget(sub)
        brand_box.addWidget(logo)
        brand_box.addLayout(txt)
        brand_box.addStretch(1)
        sl.addLayout(brand_box)
        sl.addSpacing(22)

        self.stack = QStackedWidget()
        pages = [
            ("▶   Run", self._page_run),
            ("▦   Squads", self._page_squads),
            ("★   Leaderboard", self._page_leaderboard),
            ("💬  Board", self._page_board),
            ("📦  Artefacts", self._page_artifacts),
            ("🕘  History", self._page_history),
            ("⚙   Settings", self._page_settings),
        ]
        self.nav_buttons = []
        for i, (label, builder) in enumerate(pages):
            btn = QPushButton(label)
            btn.setObjectName("nav")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, idx=i: self.goto(idx))
            sl.addWidget(btn)
            self.nav_buttons.append(btn)
            page = builder()
            self.stack.addWidget(page)
        sl.addStretch(1)

        self.pill = QLabel("IDLE")
        self.pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pill.setFixedHeight(26)
        self.pill.setStyleSheet(
            f"background:#18233a; color:{C['mut']}; border-radius:13px;"
            " font-size:10px; letter-spacing:1.4px; font-weight:700;"
        )
        sl.addWidget(self.pill)
        ver = QLabel(f"PyQt6 · Python {sys.version.split()[0]}")
        ver.setObjectName("brandsub")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sl.addWidget(ver)

        outer.addWidget(side)
        outer.addWidget(self.stack, 1)
        self.goto(0)

        # keyboard shortcuts (documented as tooltips on the buttons)
        start_sc = QShortcut(QKeySequence("F5"), self)
        start_sc.activated.connect(self.start_run)
        stop_sc = QShortcut(QKeySequence("Shift+F5"), self)
        stop_sc.activated.connect(self.stop_run)
        dash_sc = QShortcut(QKeySequence("Ctrl+D"), self)
        dash_sc.activated.connect(self.open_dashboard)

    def goto(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)
        for i, b in enumerate(self.nav_buttons):
            b.setChecked(i == idx)
        if idx == 5:
            self.refresh_history()

    # ------------------------------------------------------------- RUN page
    def _page_run(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(26, 20, 26, 20)
        root.setSpacing(14)

        head = QHBoxLayout()
        hb = QVBoxLayout()
        hb.setSpacing(0)
        t = QLabel("Run")
        t.setObjectName("h1")
        s = QLabel("Configure a swarm run, watch it live, resume it later.")
        s.setObjectName("sub")
        hb.addWidget(t)
        hb.addWidget(s)
        head.addLayout(hb)
        head.addStretch(1)

        self.btn_start = QPushButton("Start run")
        self.btn_start.setObjectName("primary")
        self.btn_start.setFixedWidth(132)
        self.btn_start.setToolTip("Start the swarm run  (F5)")
        self.btn_start.clicked.connect(self.start_run)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.setFixedWidth(96)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip(
            "Ask the run to finish the current squad and stop  (Shift+F5)")
        self.btn_stop.clicked.connect(self.stop_run)
        self.btn_dash = QPushButton("Open dashboard")
        self.btn_dash.setObjectName("ghost")
        self.btn_dash.setEnabled(False)
        self.btn_dash.setToolTip(
            "Open the HTML dashboard in your browser  (Ctrl+D)")
        self.btn_dash.clicked.connect(self.open_dashboard)
        self.btn_out = QPushButton("Open out folder")
        self.btn_out.setObjectName("ghost")
        self.btn_out.setToolTip("Open the output folder in the file manager")
        self.btn_out.clicked.connect(self.open_out_folder)
        for b in (self.btn_start, self.btn_stop, self.btn_dash, self.btn_out):
            head.addWidget(b)
        root.addLayout(head)

        # stats
        self.st_squads = StatCard("squads")
        self.st_tokens = StatCard("tokens")
        self.st_calls = StatCard("llm calls")
        self.st_posts = StatCard("board posts")
        self.st_latency = StatCard("avg latency")
        root.addLayout(stat_row([self.st_squads, self.st_tokens, self.st_calls,
                                 self.st_posts, self.st_latency]))

        # progress + elapsed
        prog = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(16)
        self.phase = QLabel("idle")
        self.phase.setObjectName("sub")
        self.phase.setMinimumWidth(340)
        self.elapsed = QLabel("00:00")
        self.elapsed.setObjectName("sub")
        prog.addWidget(self.progress, 1)
        prog.addWidget(self.phase, 2)
        prog.addWidget(self.elapsed)
        root.addLayout(prog)

        # config card (scrollable)
        card = QFrame()
        card.setObjectName("card")
        form = QVBoxLayout(card)
        form.setContentsMargins(16, 14, 16, 14)
        form.setSpacing(10)

        self.goal = QPlainTextEdit(SETTINGS.goal)
        self.goal.setMinimumHeight(58)
        self.goal.setMaximumHeight(92)
        fl = QLabel("GOAL")
        fl.setObjectName("lbl")
        form.addWidget(fl)
        form.addWidget(self.goal)

        grid = QGridLayout()
        grid.setSpacing(10)
        # defaults come from swarm/config.py so the GUI can never drift from
        # the CLI defaults or the README flag table
        base, defaults = Scale(), Settings()
        self.f_squads = self._spin(1, 500, base.squads)
        self.f_agents = self._spin(1, 20, base.agents_per_squad)
        self.f_curators = self._spin(1, 50, base.curators)
        self.f_conc = self._spin(1, 64, base.concurrency)
        self.f_timeout = self._spin(5, 600, base.llm_timeout)
        self.f_reviews = self._spin(0, 10, defaults.peer_reviews_per_work)
        self.f_disc = self._spin(0, 10, defaults.discussants)
        self.f_budget = self._spin(0, 2_000_000_000, defaults.max_total_tokens)
        self.f_budget.setGroupSeparatorShown(True)
        self.f_ctx = self._spin(0, 500, defaults.board_context_limit)
        self.f_prefix = QLineEdit(defaults.run_prefix)
        fields = [
            ("squads", self.f_squads), ("agents / squad", self.f_agents),
            ("curators", self.f_curators), ("concurrency", self.f_conc),
            ("llm timeout (s)", self.f_timeout), ("peer reviews", self.f_reviews),
            ("discussants", self.f_disc), ("token budget (0 = ∞)", self.f_budget),
            ("board context", self.f_ctx), ("run prefix", self.f_prefix),
        ]
        for i, (label, widget) in enumerate(fields):
            grid.addLayout(labeled_field(label, widget), i // 5, i % 5)
        form.addLayout(grid)

        opts = QHBoxLayout()
        self.f_offline = QCheckBox("offline (no tokens)")
        self.f_nosbx = QCheckBox("no sandbox (dry run)")
        self.f_offline.toggled.connect(self._toggle_mode)
        self.f_nosbx.toggled.connect(self._toggle_mode)
        opts.addWidget(self.f_offline)
        opts.addWidget(self.f_nosbx)
        opts.addSpacing(18)
        self.mode_hint = QLabel("")
        self.mode_hint.setObjectName("sub")
        opts.addWidget(self.mode_hint, 1)
        form.addLayout(opts)

        # output dir
        od = QHBoxLayout()
        self._out_edit = QLineEdit(os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "out")))
        self._out_edit.setPlaceholderText("directory for reports, checkpoints and dashboards")
        self._out_edit.textChanged.connect(self.refresh_history)
        out_browse = QPushButton("Browse…")
        out_browse.setObjectName("ghost")
        out_browse.clicked.connect(self._browse_out)
        od.addLayout(labeled_field("output dir", self._out_edit), 1)
        od.addWidget(out_browse)
        form.addLayout(od)

        res = QHBoxLayout()
        self.f_resume = QLineEdit()
        self.f_resume.setPlaceholderText("optional: out/_checkpoint.json to continue a previous run")
        browse = QPushButton("Browse…")
        browse.setObjectName("ghost")
        browse.clicked.connect(self._browse_resume)
        clear = QPushButton("Clear")
        clear.setObjectName("ghost")
        clear.clicked.connect(lambda: self.f_resume.clear())
        res.addLayout(labeled_field("resume from", self.f_resume), 1)
        res.addWidget(browse)
        res.addWidget(clear)
        form.addLayout(res)

        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(card)
        # the form sizes itself to its content (so the resume row is never
        # clipped) but is capped so the console always keeps room
        sc.setMaximumHeight(470)
        root.addWidget(sc)

        # console
        self.console = Console()
        cl = QHBoxLayout()
        ct = QLabel("CONSOLE")
        ct.setObjectName("lbl")
        cl.addWidget(ct)
        cl.addStretch(1)
        follow = QCheckBox("follow")
        follow.setChecked(True)
        follow.toggled.connect(self.console.set_follow)
        clear_c = QPushButton("clear")
        clear_c.setObjectName("ghost")
        clear_c.clicked.connect(self.console.clear)
        cl.addWidget(follow)
        cl.addWidget(clear_c)
        root.addLayout(cl)
        root.addWidget(self.console, 1)
        self.console.write_line("Ready. Pick a mode and press Start run.", C["mut"])
        self._toggle_mode()
        return w

    @staticmethod
    def _spin(lo: int, hi: int, val: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    def _toggle_mode(self) -> None:
        if self.f_offline.isChecked() and self.f_nosbx.isChecked():
            self.mode_hint.setText("pipeline dry-run — free, nothing is executed")
            self.mode_hint.setStyleSheet(f"color:{C['good']};")
        elif self.f_offline.isChecked():
            self.mode_hint.setText("offline LLM stubs — free, sandbox still used")
            self.mode_hint.setStyleSheet(f"color:{C['mid']};")
        elif self.f_nosbx.isChecked():
            self.mode_hint.setText("live LLM, no sandbox writes")
            self.mode_hint.setStyleSheet(f"color:{C['mid']};")
        else:
            self.mode_hint.setText("FULL LIVE RUN — spends NVIDIA tokens + sandbox")
            self.mode_hint.setStyleSheet(f"color:{C['low']};")

    def _browse_resume(self) -> None:
        d = os.path.abspath(self.out_dir())
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a checkpoint or report", d,
            "Swarm JSON (*.json);;All files (*)")
        if path:
            self.f_resume.setText(path)

    def _browse_out(self) -> None:
        d = os.path.abspath(self.out_dir())
        path = QFileDialog.getExistingDirectory(self, "Choose the output directory", d)
        if path:
            self._out_edit.setText(path)

    # ---------------------------------------------------------- other pages
    def _page_squads(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(26, 20, 26, 20)
        root.setSpacing(12)
        root.addLayout(page_header("Squads",
                           "Each squad = one role, five agents, one curator-council prompt."))

        self.tbl_squads = make_table(
            ["#", "Role", "Dur", "Agents", "Avg", "Curator winner", "Prompt scores",
             "Verification", "State"], stretch_col=7)
        root.addWidget(self.tbl_squads, 3)

        chart_box = QFrame()
        chart_box.setObjectName("card")
        chart_box.setFixedHeight(236)
        cl = QVBoxLayout(chart_box)
        cl.setContentsMargins(14, 10, 14, 10)
        lbl = QLabel("SQUAD DURATION (s)")
        lbl.setObjectName("lbl")
        cl.addWidget(lbl)
        self.chart_squads = BarChart("waiting for the first squad…")
        cl.addWidget(self.chart_squads)
        root.addWidget(chart_box, 1)
        return w

    def _page_leaderboard(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(26, 20, 26, 20)
        root.setSpacing(12)
        root.addLayout(page_header("Leaderboard",
                           "Peer-voted reputation: every agent scores its squadmates 0–10."))
        self.tbl_leaders = make_table(
            ["#", "Agent", "Reputation", "Avg", "Reviews"], stretch_col=1)
        root.addWidget(self.tbl_leaders, 3)

        box = QFrame()
        box.setObjectName("card")
        box.setFixedHeight(236)
        bl = QVBoxLayout(box)
        bl.setContentsMargins(14, 10, 14, 10)
        l = QLabel("TOP REPUTATION")
        l.setObjectName("lbl")
        bl.addWidget(l)
        self.chart_leaders = BarChart("no peer votes yet")
        bl.addWidget(self.chart_leaders)
        root.addWidget(box, 1)
        return w

    def _page_board(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(26, 20, 26, 20)
        root.setSpacing(12)
        root.addLayout(page_header("Discussion board",
                           "The short digest every agent can see — the swarm's shared memory."))

        top = QHBoxLayout()
        top.addStretch(1)
        self.board_filter = QComboBox()
        self.board_filter.setObjectName("filter")
        self._board_kinds = {"digest", "discussion", "curator", "verification",
                             "system", "vote"}
        self.board_filter.addItems(
            ["all kinds", "digest", "discussion", "curator", "verification",
             "system", "vote"])
        self.board_filter.currentIndexChanged.connect(self._render_board)
        top.addWidget(QLabel("filter"))
        top.addWidget(self.board_filter)
        root.addLayout(top)

        self.tbl_board = make_table(["#", "Squad", "Kind", "Author", "Role", "Text"],
                                    stretch_col=5)
        root.addWidget(self.tbl_board, 1)
        return w

    def _page_artifacts(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(26, 20, 26, 20)
        root.setSpacing(12)
        root.addLayout(page_header("Artefacts",
                           "Files the swarm wrote into the shared Daytona sandbox."))

        self.run_dir_lbl = QLabel("run dir: (not started)")
        self.run_dir_lbl.setObjectName("sub")
        root.addWidget(self.run_dir_lbl)

        self.tbl_art = make_table(["Artefact", "Path"], stretch_col=1)
        root.addWidget(self.tbl_art, 1)

        row = QHBoxLayout()
        copy = QPushButton("Copy run dir")
        copy.setObjectName("ghost")
        copy.clicked.connect(self._copy_run_dir)
        openb = QPushButton("Open out folder")
        openb.setObjectName("ghost")
        openb.clicked.connect(self.open_out_folder)
        row.addWidget(copy)
        row.addWidget(openb)
        row.addStretch(1)
        root.addLayout(row)
        return w

    def _page_history(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(26, 20, 26, 20)
        root.setSpacing(12)
        root.addLayout(page_header("History",
                           "Every report this project has produced. Double-click to open its dashboard."))

        row = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_history)
        openb = QPushButton("Open out folder")
        openb.setObjectName("ghost")
        openb.clicked.connect(self.open_out_folder)
        row.addWidget(refresh)
        row.addWidget(openb)
        row.addStretch(1)
        root.addLayout(row)

        self.tbl_hist = make_table(
            ["Report", "When", "Squads", "Tokens", "Board", "Artefacts"],
            stretch_col=0)
        self.tbl_hist.cellDoubleClicked.connect(self._open_history_row)
        root.addWidget(self.tbl_hist, 1)

        self.hist_hint = QLabel("Tip: use “Resume from” on the Run page with one of these files.")
        self.hist_hint.setObjectName("sub")
        root.addWidget(self.hist_hint)
        return w

    def _page_settings(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(26, 20, 26, 20)
        root.setSpacing(12)
        root.addLayout(page_header("Settings",
                           "Read-only view of what the swarm is configured to use."))

        grid = QGridLayout()
        grid.setSpacing(12)

        # credentials
        cred = QFrame()
        cred.setObjectName("card")
        cl = QVBoxLayout(cred)
        cl.setContentsMargins(16, 14, 16, 14)
        cl.setSpacing(8)
        ct = QLabel("CREDENTIALS")
        ct.setObjectName("lbl")
        cl.addWidget(ct)
        self.key_nvidia = QLineEdit()
        self.key_nvidia.setReadOnly(True)
        self.key_nvidia.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_daytona = QLineEdit()
        self.key_daytona.setReadOnly(True)
        self.key_daytona.setEchoMode(QLineEdit.EchoMode.Password)
        self.btn_reveal = QPushButton("Reveal")
        self.btn_reveal.setObjectName("ghost")
        self.btn_reveal.setCheckable(True)
        self.btn_reveal.toggled.connect(self._toggle_keys)
        cl.addLayout(labeled_field("nvidia api key", self.key_nvidia))
        cl.addLayout(labeled_field("daytona api key", self.key_daytona))
        cl.addWidget(self.btn_reveal)
        note = QLabel("Set NVIDIA_API_KEY / DAYTONA_API_KEY in the environment to override.")
        note.setObjectName("sub")
        note.setWordWrap(True)
        cl.addWidget(note)
        self.key_nvidia.setText(NVIDIA_API_KEY)
        self.key_daytona.setText(DAYTONA_API_KEY)

        # sandbox
        sb = QFrame()
        sb.setObjectName("card")
        sl = QVBoxLayout(sb)
        sl.setContentsMargins(16, 14, 16, 14)
        sl.setSpacing(8)
        st = QLabel("SANDBOX")
        st.setObjectName("lbl")
        sl.addWidget(st)
        for name, value in (("name", DAYTONA_SANDBOX_NAME),
                            ("id", DAYTONA_SANDBOX_ID),
                            ("workspace", SANDBOX_WORKSPACE),
                            ("python", sys.version.split()[0]),
                            ("qt / pyqt", f"{self._qt_version()} / {self._pyqt_version()}")):
            r = QHBoxLayout()
            k = QLabel(name)
            k.setObjectName("sub")
            v = QLineEdit(value)
            v.setReadOnly(True)
            r.addWidget(k, 1)
            r.addWidget(v, 2)
            sl.addLayout(r)
        sl.addStretch(1)

        grid.addWidget(cred, 0, 0)
        grid.addWidget(sb, 0, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        root.addLayout(grid)

        tests = QFrame()
        tests.setObjectName("card")
        tl = QVBoxLayout(tests)
        tl.setContentsMargins(16, 14, 16, 14)
        tt = QLabel("MAINTENANCE")
        tt.setObjectName("lbl")
        tl.addWidget(tt)
        btns = QHBoxLayout()
        b1 = QPushButton("Run offline test suite")
        b1.clicked.connect(self.run_tests)
        b2 = QPushButton("Open out folder")
        b2.setObjectName("ghost")
        b2.clicked.connect(self.open_out_folder)
        b3 = QPushButton("Run `swarm probe`")
        b3.setObjectName("ghost")
        b3.clicked.connect(self.run_probe)
        for b in (b1, b2, b3):
            btns.addWidget(b)
        btns.addStretch(1)
        tl.addLayout(btns)
        tl.addWidget(QLabel("Output appears on the Run page console."))
        root.addWidget(tests)
        root.addStretch(1)
        return w

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _qt_version() -> str:
        from PyQt6.QtCore import QT_VERSION_STR
        return QT_VERSION_STR

    @staticmethod
    def _pyqt_version() -> str:
        from PyQt6.QtCore import PYQT_VERSION_STR
        return PYQT_VERSION_STR

    def _toggle_keys(self, on: bool) -> None:
        mode = (QLineEdit.EchoMode.Normal if on
                else QLineEdit.EchoMode.Password)
        self.key_nvidia.setEchoMode(mode)
        self.key_daytona.setEchoMode(mode)
        self.btn_reveal.setText("Hide" if on else "Reveal")

    def out_dir(self) -> str:
        text = (self._find_out_edit().text() or "").strip()
        return os.path.abspath(text or os.path.join(os.path.dirname(__file__), "..", "out"))

    def _find_out_edit(self) -> QLineEdit:
        return self._out_edit

    def open_out_folder(self) -> None:
        d = self.out_dir()
        os.makedirs(d, exist_ok=True)
        _open_path(d)

    def _copy_run_dir(self) -> None:
        if self.orch:
            QApplication.clipboard().setText(self.orch.run_dir)
            self.console.write_line(f"copied: {self.orch.run_dir}", C["acc"])

    def open_dashboard(self) -> None:
        if not self._last_dashboard or not os.path.exists(self._last_dashboard):
            QMessageBox.information(self, "No dashboard",
                                    "Run something first — the dashboard is written at the end.")
            return
        _open_path(self._last_dashboard)

    def run_tests(self) -> None:
        self.goto(0)
        self.console.write_line("\n$ python -m unittest discover -s . -p \"test_*.py\"", C["acc"])
        self._spawn([sys.executable, "-m", "unittest", "discover", "-s", ".",
                     "-p", "test_*.py"])

    def run_probe(self) -> None:
        self.goto(0)
        self.console.write_line("\n$ python -m swarm probe", C["acc"])
        self._spawn([sys.executable, "-m", "swarm", "probe"])

    def _spawn(self, cmd: List[str]) -> None:
        if self._proc and self._proc.poll() is None:
            QMessageBox.warning(self, "Busy", "A command is already running.")
            return
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", bufsize=1,
            )
        except Exception as e:
            self.console.write_line(f"! spawn failed: {e}", C["low"])
            return

        proc = self._proc

        class Pump(QThread):
            line = pyqtSignal(str)

            def run(inner) -> None:
                assert proc.stdout is not None
                for ln in proc.stdout:
                    inner.line.emit(ln.rstrip("\n"))
                proc.wait()

        self._pump = Pump()
        self._pump.line.connect(lambda t: self.console.write_line(t))
        self._pump.finished.connect(lambda: self.console.write_line(
            f"[exit {proc.returncode}]", C["good"] if proc.returncode == 0 else C["low"]))
        self._pump.start()

    # ------------------------------------------------------------ run flow
    def start_run(self) -> None:
        if self.running:
            return
        if self.worker is not None and self.worker.isRunning():
            # the previous worker is still winding down (it emits `done`
            # before its thread actually exits) - never swap it out live
            return

        out_dir = self.out_dir()
        scale = Scale(
            squads=self.f_squads.value(),
            agents_per_squad=self.f_agents.value(),
            curators=self.f_curators.value(),
            concurrency=self.f_conc.value(),
            llm_timeout=self.f_timeout.value(),
        )
        settings = Settings(
            scale=scale,
            goal=self.goal.toPlainText().strip() or SETTINGS.goal,
            use_sandbox=not self.f_nosbx.isChecked(),
            live_llm=not self.f_offline.isChecked(),
            verbose=True,
            board_context_limit=self.f_ctx.value(),
            peer_reviews_per_work=self.f_reviews.value(),
            discussants=self.f_disc.value(),
            max_total_tokens=self.f_budget.value(),
            run_prefix=self.f_prefix.text().strip() or "run",
        )

        orch = LiveOrchestrator(settings, out_dir=out_dir)
        orch.line.connect(self.console.write_line)

        start_squad = 0
        resume = self.f_resume.text().strip()
        if resume:
            from . import cli as cli_mod
            original = cli_mod._print
            cli_mod._print = lambda *a: self.console.write_line(
                " ".join(str(x) for x in a), C["acc"])
            try:
                start_squad = cli_mod._resume(orch, resume)
            finally:
                cli_mod._print = original
            if start_squad is None:
                QMessageBox.warning(self, "Resume failed",
                                    "That file could not be used as a checkpoint.")
                return
            if start_squad >= scale.squads:
                QMessageBox.information(
                    self, "Nothing to do",
                    f"Every squad in that checkpoint is complete ({start_squad}/"
                    f"{scale.squads}). Raise the squad count to continue past it.")
                return

        self.orch = orch
        # reset the views so a second run does not inherit the first run's rows
        self._board_seen = 0
        self._board_rows = []
        self._squads_sig = None
        self._leaders_sig = None
        self._art_seen = None
        for tbl in (self.tbl_squads, self.tbl_leaders, self.tbl_board,
                    self.tbl_art):
            tbl.setRowCount(0)
        self.chart_squads.set_data([])
        self.chart_leaders.set_data([])
        self.progress.setValue(0)
        self.elapsed.setText("00:00")
        self._run_started = time.time()

        self.worker = RunWorker(orch, start_squad, out_dir, self)
        self.worker.line.connect(self.console.write_line)
        self.worker.outputs.connect(self._on_outputs)
        self.worker.done.connect(self._on_done)

        self.running = True
        self._set_running(True)
        self.console.write_line(
            f"\n▶ starting: {scale.squads} squads × {scale.agents_per_squad} agents, "
            f"{scale.curators} curators, offline={not settings.live_llm}, "
            f"sandbox={settings.use_sandbox}", C["acc"])
        self.worker.start()

    def stop_run(self) -> None:
        if self.orch and self.running:
            self.orch.stop()
            self.console.write_line("! stop requested — finishing the current squad…",
                                    C["mid"])
            self.btn_stop.setEnabled(False)

    def _set_running(self, on: bool) -> None:
        self.running = on
        self.btn_start.setEnabled(not on)
        self.btn_stop.setEnabled(on)
        for w in (self.f_squads, self.f_agents, self.f_curators, self.f_conc,
                  self.f_timeout, self.f_reviews, self.f_disc, self.f_budget,
                  self.f_ctx, self.f_prefix, self.f_resume, self.goal,
                  self.f_offline, self.f_nosbx, self._out_edit):
            w.setEnabled(not on)
        self.pill.setText("RUNNING" if on else "IDLE")
        self.pill.setStyleSheet(
            f"background:{'rgba(61,220,151,.15)' if on else '#18233a'};"
            f" color:{C['good'] if on else C['mut']}; border-radius:13px;"
            " font-size:10px; letter-spacing:1.4px; font-weight:700;")
        prefix = "● RUNNING — " if on else ""
        self.setWindowTitle(prefix + "Agent Swarm — multi-agent orchestration console")

    def _on_outputs(self, json_path: str, txt_path: str, html_path: str) -> None:
        if html_path:
            self._last_dashboard = html_path
            self.btn_dash.setEnabled(True)
        self.refresh_history()

    def _on_done(self, success: bool, message: str) -> None:
        self._set_running(False)
        colour = C["good"] if success else C["low"]
        self.pill.setText("DONE" if success else "STOPPED")
        self.pill.setStyleSheet(
            f"background:rgba(255,255,255,.05); color:{colour}; border-radius:13px;"
            " font-size:10px; letter-spacing:1.4px; font-weight:700;")
        self.console.write_line(f"■ {message}", colour)
        self.phase.setText(message)
        self._tick()  # final paint with the finished state

    # --------------------------------------------------------------- polls
    def _tick_clock(self) -> None:
        if not self.running or not self._run_started:
            return
        s = int(time.time() - self._run_started)
        self.elapsed.setText(f"{s // 60:02d}:{s % 60:02d}")

    def _tick(self) -> None:
        orch = self.orch
        if orch is None:
            return

        snap = orch.metrics.snapshot()
        llm = snap.get("llm", {}) or {}
        completed = snap.get("squads_completed", 0)
        total = max(1, orch.scale.squads)
        self.st_squads.set(f"{completed}/{orch.scale.squads}",
                           C["good"] if completed >= orch.scale.squads else C["acc"])
        self.st_tokens.set(f"{llm.get('total_tokens', 0):,}")
        self.st_calls.set(f"{llm.get('calls', 0):,}")
        self.st_posts.set(str(orch.board.count()))
        self.st_latency.set(f"{llm.get('avg_latency', 0)}s")
        self.progress.setValue(min(100, int(100 * completed / total)))
        self.run_dir_lbl.setText(f"run dir: {orch.run_dir}")

        # Render even when idle: the final tick after `done` must still show
        # the finished state. Every block below is guarded by a change-signature
        # so an idle window costs one dict snapshot per tick.
        # newest console-ish phase line (only while a run is actually going,
        # so the "run complete" message written by _on_done is not clobbered)
        rows = orch.metrics.squad_rows()
        if self.running and rows:
            last = rows[-1]
            self.phase.setText(f"squad #{last['squad'] + 1} · {last['role']}")

        # squads table
        squad_sig = (len(rows), len(orch.reports),
                     tuple((r["squad"], r["duration"], r["avg_score"]) for r in rows))
        if squad_sig != self._squads_sig:
            self._squads_sig = squad_sig
            reports = {r.squad: r for r in list(orch.reports)}
            self.tbl_squads.setRowCount(len(rows))
            for i, row in enumerate(rows):
                rep = reports.get(row["squad"])
                scores = (rep.prompt_scores if rep else []) or []
                values = [
                    str(row["squad"] + 1), row["role"], f"{row['duration']:.1f}s",
                    str(row["agents"]), f"{row['avg_score']:.2f}",
                    rep.winning_curator if rep else "—",
                    ", ".join(f"{s:.1f}" for s in scores[:6]) or "—",
                    (rep.verification if rep else "—") or "—",
                    "ok" if (rep.ok if rep else True) else "failed",
                ]
                for c, val in enumerate(values):
                    item = QTableWidgetItem(val)
                    if c == 4:
                        item.setForeground(QColor(_score_color(row["avg_score"])))
                    if c == 6:
                        item.setToolTip("all curator scores: " +
                                        ", ".join(f"{s:.1f}" for s in scores))
                    if c == 7 and rep:
                        item.setToolTip(
                            rep.verification +
                            (f"\n\n{rep.error}" if rep.error else ""))
                    if c == 7 and rep:
                        item.setForeground(QColor(
                            C["good"] if rep.verification.startswith("ran `") and
                            "passed=True" in rep.verification
                            else (C["mut"] if rep.verification.startswith("verification")
                                  else C["mid"])))
                    if c == 8 and rep and not rep.ok:
                        item.setForeground(QColor(C["low"]))
                    self.tbl_squads.setItem(i, c, item)
            self.chart_squads.set_data([
                (f"#{r['squad'] + 1} {r['role']}", float(r["duration"]),
                 _score_color(r["avg_score"]))
                for r in rows[-14:]
            ])

        # leaderboard
        leaders = orch.metrics.leaderboard(50)
        leaders_sig = tuple((r["agent"], r["reputation"], r["reviews"])
                            for r in leaders)
        if leaders_sig != self._leaders_sig:
            self._leaders_sig = leaders_sig
            self.tbl_leaders.setRowCount(len(leaders))
            for i, row in enumerate(leaders):
                for c, val in enumerate([str(i + 1), row["agent"],
                                         f"{row['reputation']:.2f}",
                                         f"{row['avg']:.2f}",
                                         str(row["reviews"])]):
                    item = QTableWidgetItem(val)
                    if c == 2:
                        item.setForeground(QColor(_score_color(row["avg"])))
                    self.tbl_leaders.setItem(i, c, item)
            self.chart_leaders.set_data([
                (r["agent"], float(r["reputation"]), C["violet"])
                for r in leaders[:12]
            ])

        # board
        posts = orch.board.all()
        if len(posts) > self._board_seen:
            self._board_rows.extend(
                {"seq": p.seq, "squad": p.squad, "kind": p.kind, "author": p.author,
                 "role": p.role, "text": p.text} for p in posts[self._board_seen:])
            self._board_seen = len(posts)
            if len(self._board_rows) > 1500:
                self._board_rows = self._board_rows[-1500:]
            self._sync_board_kinds()
            self._render_board()

        # artefacts
        arts = orch.blackboard.artifact_list()
        if len(arts) != self._art_seen:
            self._art_seen = len(arts)
            self.tbl_art.setRowCount(len(arts))
            for i, a in enumerate(arts):
                self.tbl_art.setItem(i, 0, QTableWidgetItem(a))
                self.tbl_art.setItem(i, 1, QTableWidgetItem(f"{orch.run_dir}/{a}"))

    def _sync_board_kinds(self) -> None:
        """Add any brand-new post kind to the filter combo (the swarm may
        introduce kinds the static list does not know about)."""
        missing = sorted({r["kind"] for r in self._board_rows} - self._board_kinds)
        if not missing:
            return
        self._board_kinds.update(missing)
        current = self.board_filter.currentText()
        self.board_filter.blockSignals(True)
        self.board_filter.addItems(missing)
        idx = self.board_filter.findText(current)
        self.board_filter.setCurrentIndex(idx if idx >= 0 else 0)
        self.board_filter.blockSignals(False)

    def _render_board(self) -> None:
        want = self.board_filter.currentText()
        rows = [r for r in self._board_rows
                if want == "all kinds" or r["kind"] == want]
        rows = rows[-400:]
        self.tbl_board.setRowCount(len(rows))
        kind_color = {
            "digest": C["acc"], "discussion": C["violet"], "curator": C["mid"],
            "verification": C["good"], "system": C["mut"], "vote": C["mid"],
        }
        for i, r in enumerate(reversed(rows)):
            vals = [str(r["seq"]),
                    "-" if r["squad"] < 0 else str(r["squad"] + 1),
                    r["kind"], r["author"], r["role"], r["text"]]
            for c, val in enumerate(vals):
                item = QTableWidgetItem(val)
                if c == 2:
                    item.setForeground(QColor(kind_color.get(r["kind"], C["fg"])))
                if c == 5 and len(val) > 60:
                    item.setToolTip(val)   # full text is wider than the column
                self.tbl_board.setItem(i, c, item)
        if rows:
            self.tbl_board.scrollToBottom()

    # ------------------------------------------------------------- history
    def refresh_history(self) -> None:
        d = self.out_dir()
        files: List[str] = []
        if os.path.isdir(d):
            files = sorted(
                (os.path.join(d, f) for f in os.listdir(d)
                 if f.startswith("swarm-report-") and f.endswith(".json")),
                key=os.path.getmtime, reverse=True)
        self.tbl_hist.setRowCount(len(files))
        for i, path in enumerate(files):
            meta = {"squads": "?", "metrics": {}, "board": []}
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            except Exception:
                pass
            m = meta.get("metrics", {}) or {}
            name = os.path.basename(path)
            vals = [name,
                    time.strftime("%Y-%m-%d %H:%M",
                                  time.localtime(os.path.getmtime(path))),
                    f"{m.get('squads_completed', '?')}/{(meta.get('settings') or {}).get('squads', '?')}",
                    f"{(m.get('llm') or {}).get('total_tokens', 0):,}",
                    str(len(meta.get("board") or [])),
                    str(len(meta.get("artifacts") or []))]
            for c, val in enumerate(vals):
                self.tbl_hist.setItem(i, c, QTableWidgetItem(val))
            self.tbl_hist.item(i, 0).setData(Qt.ItemDataRole.UserRole, path)

    def _open_history_row(self, row: int, _col: int) -> None:
        item = self.tbl_hist.item(row, 0)
        if not item:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        dash = path.replace("-report-", "-dashboard-").replace(".json", ".html")
        if os.path.exists(dash):
            _open_path(dash)
        else:
            QMessageBox.information(self, "Dashboard not found",
                                    f"No dashboard next to:\n{os.path.basename(path)}")

    # ---------------------------------------------------------- life cycle
    def closeEvent(self, ev) -> None:  # noqa: N802
        if self.running:
            ans = QMessageBox.question(
                self, "Run in progress",
                "A swarm run is still going. Stop it and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                ev.ignore()
                return
            if self.orch:
                self.orch.stop()
            if self.worker:
                self.worker.wait(3000)
        ev.accept()


def main(argv: Optional[List[str]] = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    return app.exec()
