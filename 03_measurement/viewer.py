"""
측정 화면 (tkinter).

    열기            : 1단계의 orthorectified.png / aruco_detected.png, 2단계의 registered_mosaic.png
    마우스 휠        : 마우스가 있는 자리를 중심으로 확대·축소
    오른쪽 단추 끌기 : 그림 옮기기 (가운데 단추도 됨)
    왼쪽 단추        : 첫 번째는 시작점, 두 번째는 끝점 → 두 점 사이의 실제 길이(mm)가 목록에 추가된다
    Esc             : 찍어 둔 시작점 취소      Delete : 목록에서 고른 측정 지우기      F : 화면에 맞춤

픽셀을 mm 로 바꾸는 계산은 coords.py, 확대·이동 계산은 viewport.py 가 한다. 이 파일은 화면만 맡는다.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import coords
from config import Config
from viewport import Viewport

Image.MAX_IMAGE_PIXELS = 400_000_000      # 큰 합친 그림도 열 수 있게


@dataclass
class Measurement:
    no: int
    a_px: tuple[float, float]
    b_px: tuple[float, float]
    result: dict                          # coords.Mapper.measure() 의 결과


class MeasureApp(tk.Tk):
    def __init__(self, cfg: Config | None = None, image_path: str | None = None):
        super().__init__()
        self.cfg = cfg or Config()
        self.title("바닥판 치수 측정")
        self.geometry(self.cfg.window_size)
        self.minsize(900, 560)
        self.image: Image.Image | None = None
        self.mapper: coords.Mapper | None = None
        self.view = Viewport()
        self.measurements: list[Measurement] = []
        self.pending: tuple[float, float] | None = None       # 찍어 둔 시작점 (그림 픽셀)
        self.cursor: tuple[float, float] | None = None        # 마우스 위치 (화면)
        self._photo = None                                    # 화면에 그린 그림 (지워지지 않게 붙잡아 둔다)
        self._next_no = 1
        self._redraw_job = None
        self._drag = None
        self._fit_scale = 1.0
        self._build()
        if image_path:
            self.after(80, lambda: self.open_image(image_path))

    # ---------------------------------------------------------------- 화면 짜기
    def _build(self) -> None:
        bar = ttk.Frame(self, padding=(6, 4))
        bar.pack(side="top", fill="x")
        for text, cmd in (("열기", self.ask_open), ("화면 맞춤 (F)", self.fit), ("실제 크기 (1:1)", self.actual_size),
                          ("선택 지우기 (Del)", self.delete_selected), ("모두 지우기", self.clear_all)):
            ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=2)
        ttk.Label(bar, text="   왼쪽 단추: 시작점 → 끝점   |   휠: 확대·축소   |   오른쪽 단추 끌기: 이동   |   Esc: 시작점 취소",
                  foreground="#555555").pack(side="left", padx=8)

        status = ttk.Frame(self, padding=(6, 2))
        status.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="'열기' 를 눌러 그림을 고르세요.")
        self.zoom_var = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")
        ttk.Label(status, textvariable=self.zoom_var).pack(side="right")

        side = ttk.Frame(self, padding=6, width=self.cfg.side_panel_width)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        ttk.Label(side, text="그림 정보", font=("", 10, "bold")).pack(anchor="w")
        self.info_var = tk.StringVar(value="-")
        ttk.Label(side, textvariable=self.info_var, wraplength=self.cfg.side_panel_width - 16, justify="left").pack(anchor="w", pady=(2, 8))
        ttk.Label(side, text="측정 목록", font=("", 10, "bold")).pack(anchor="w")
        cols = (("no", "번호", 44), ("length", "길이 (mm)", 96), ("sigma", "예상 오차", 76), ("dx", "가로 (mm)", 74), ("dy", "세로 (mm)", 74))
        self.tree = ttk.Treeview(side, columns=[c[0] for c in cols], show="headings", height=12, selectmode="browse")
        for key, title, width in cols:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor="e" if key != "no" else "center", stretch=False)
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._on_select())
        self.detail_var = tk.StringVar(value="")
        ttk.Label(side, textvariable=self.detail_var, wraplength=self.cfg.side_panel_width - 16, justify="left").pack(anchor="w", pady=(6, 8))
        ttk.Label(side, text="알아 둘 점", font=("", 10, "bold")).pack(anchor="w")
        self.notes_var = tk.StringVar(value="-")
        ttk.Label(side, textvariable=self.notes_var, wraplength=self.cfg.side_panel_width - 16, justify="left", foreground="#8a4b00").pack(anchor="w", pady=(2, 0))

        self.canvas = tk.Canvas(self, background=self.cfg.background, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True)
        c = self.canvas
        c.bind("<Configure>", lambda e: self._on_resize())
        c.bind("<MouseWheel>", self._on_wheel)
        c.bind("<ButtonPress-1>", self._on_click)
        c.bind("<Motion>", self._on_motion)
        c.bind("<Leave>", lambda e: self._set_cursor(None))
        for n in (2, 3):
            c.bind(f"<ButtonPress-{n}>", self._on_drag_start)
            c.bind(f"<B{n}-Motion>", self._on_drag)
            c.bind(f"<ButtonRelease-{n}>", lambda e: setattr(self, "_drag", None))
        self.bind("<Escape>", lambda e: self.cancel_pending())
        self.bind("<Delete>", lambda e: self.delete_selected())
        self.bind("<Key-f>", lambda e: self.fit())
        self.bind("<Key-F>", lambda e: self.fit())

    # ---------------------------------------------------------------- 그림 열기
    def ask_open(self) -> None:
        start = self.mapper.image_path.parent if self.mapper else self.cfg.start_dir
        path = filedialog.askopenfilename(
            parent=self, title="측정할 그림 선택",
            initialdir=str(start) if Path(start).is_dir() else None,
            filetypes=[("그림", "*.png *.jpg *.jpeg"), ("모든 파일", "*.*")])
        if path:
            self.open_image(path)

    def open_image(self, path: str, ask_on_error: bool = True) -> bool:
        """그림과 그 옆의 좌표 정보를 읽는다. 좌표 정보를 못 찾으면 열지 않는다 (축척을 추측하지 않는다)."""
        p = Path(path)
        try:
            with Image.open(p) as im:
                image = im.convert("RGB")
        except (OSError, ValueError) as e:
            messagebox.showerror("그림 열기", f"그림을 읽지 못했습니다.\n{p.name}\n\n{e}", parent=self)
            return False
        try:
            mapper = coords.find_mapper(p, image.size)
        except coords.CoordError as e:
            if not ask_on_error:
                self.status_var.set(str(e).splitlines()[0])
                return False
            if not messagebox.askyesno("좌표 정보", f"{e}\n\n좌표 정보 파일(JSON)을 직접 고르시겠습니까?", parent=self):
                return False
            meta = filedialog.askopenfilename(parent=self, title="좌표 정보 파일 선택", initialdir=str(p.parent), filetypes=[("JSON", "*.json")])
            if not meta:
                return False
            try:
                mapper = coords.read_mapper(Path(meta), p, image.size)
            except coords.CoordError as e2:
                messagebox.showerror("좌표 정보", str(e2), parent=self)
                return False
        self.image, self.mapper = image, mapper
        self.measurements.clear()
        self.tree.delete(*self.tree.get_children())
        self.pending, self._next_no = None, 1
        self.detail_var.set("")
        self.title(f"바닥판 치수 측정 - {p.parent.name}\\{p.name}")
        scale = "자리마다 다름 (마우스를 올리면 아래에 표시)" if mapper.kind == "photo" else f"{mapper.plane_from_pixel[0, 0]:.4g} mm/픽셀"
        self.info_var.set(f"파일: {p.parent.name}\\{p.name}\n종류: {mapper.kind_name}\n크기: {mapper.width} × {mapper.height} 픽셀\n"
                          f"축척: {scale}\n좌표 정보: {mapper.meta_path.name}\n좌표계: {mapper.frame}")
        notes = list(mapper.notes)
        if mapper.marker_corners_photo_px is not None:
            notes.append(f"'예상 오차' 는 마커 모서리를 {self.cfg.corner_noise_px:g} 픽셀 틀리게 읽었다고 가정한 값입니다. 실제 오차는 더 클 수 있습니다.")
        self.notes_var.set("\n".join(f"· {n}" for n in notes) or "-")
        self.status_var.set("그림을 열었습니다. 휠로 확대한 뒤 왼쪽 단추로 시작점과 끝점을 차례로 찍으세요.")
        self.update_idletasks()
        self.fit()
        return True

    # ---------------------------------------------------------------- 확대·이동
    def _view_size(self) -> tuple[int, int]:
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def fit(self) -> None:
        if self.image is None:
            return
        w, h = self._view_size()
        self.view.fit(self.image.width, self.image.height, w, h)
        self._fit_scale = self.view.scale
        self.redraw()

    def actual_size(self) -> None:
        if self.image is None:
            return
        w, h = self._view_size()
        u, v = self.view.to_image(w / 2.0, h / 2.0)
        self.view.scale = 1.0
        self.view.centre_on(u, v, w, h)
        self.redraw()

    def zoom(self, sx: float, sy: float, steps: float) -> None:
        if self.image is None:
            return
        self.view.zoom_at(sx, sy, self.cfg.zoom_step ** steps, min(self._fit_scale * 0.5, 1.0), self.cfg.max_zoom)
        self.redraw()

    def _on_wheel(self, event) -> None:
        self.zoom(event.x, event.y, event.delta / 120.0)

    def _on_drag_start(self, event) -> None:
        self._drag = (event.x, event.y)

    def _on_drag(self, event) -> None:
        if self._drag is None or self.image is None:
            return
        self.view.pan(event.x - self._drag[0], event.y - self._drag[1])
        self._drag = (event.x, event.y)
        self._schedule_redraw()

    def _on_resize(self) -> None:
        self._schedule_redraw()

    # ---------------------------------------------------------------- 점 찍기
    def _on_click(self, event) -> None:
        if self.mapper is None:
            return
        p = self.view.to_image(event.x, event.y)
        if not self.mapper.inside(p):
            self.status_var.set("그림 밖입니다. 그림 안을 찍으세요.")
            return
        if self.pending is None:
            self.pending = p
            self._draw_live()
        else:
            a, self.pending = self.pending, None
            self.add_measurement(a, p)

    def add_measurement(self, a_px, b_px) -> Measurement | None:
        """두 픽셀 사이를 재서 목록에 넣는다. 계산할 수 없는 점이면 넣지 않고 이유를 아래에 표시한다."""
        try:
            result = self.mapper.measure(a_px, b_px, self.cfg.corner_noise_px)
        except coords.CoordError as e:
            self.status_var.set(str(e))
            self.redraw()
            return None
        m = Measurement(self._next_no, (float(a_px[0]), float(a_px[1])), (float(b_px[0]), float(b_px[1])), result)
        self._next_no += 1
        self.measurements.append(m)
        sigma = result["sigma_mm"]
        self.tree.insert("", "end", iid=str(m.no), values=(m.no, f"{result['length_mm']:,.1f}", f"±{sigma:.1f}" if sigma is not None else "-",
                                                           f"{abs(result['dx_mm']):,.1f}", f"{abs(result['dy_mm']):,.1f}"))
        self.tree.selection_set(str(m.no))
        self.tree.see(str(m.no))
        self.redraw()
        return m

    def cancel_pending(self) -> None:
        self.pending = None
        self._draw_live()

    def selected_no(self) -> int | None:
        sel = self.tree.selection()
        return int(sel[0]) if sel else None

    def delete_selected(self) -> None:
        no = self.selected_no()
        if no is None:
            return
        self.measurements = [m for m in self.measurements if m.no != no]
        self.tree.delete(str(no))
        self.detail_var.set("")
        self.redraw()

    def clear_all(self) -> None:
        self.measurements.clear()
        self.tree.delete(*self.tree.get_children())
        self.pending = None
        self.detail_var.set("")
        self.redraw()

    def _on_select(self) -> None:
        no = self.selected_no()
        m = next((x for x in self.measurements if x.no == no), None)
        if m is None:
            self.detail_var.set("")
        else:
            r = m.result
            self.detail_var.set(f"{m.no}번:  {r['length_mm']:,.2f} mm\n"
                                f"시작 ({r['a_mm'][0]:,.1f}, {r['a_mm'][1]:,.1f}) mm   픽셀 ({m.a_px[0]:.1f}, {m.a_px[1]:.1f})\n"
                                f"끝    ({r['b_mm'][0]:,.1f}, {r['b_mm'][1]:,.1f}) mm   픽셀 ({m.b_px[0]:.1f}, {m.b_px[1]:.1f})")
        self._draw_overlays()

    # ---------------------------------------------------------------- 그리기
    def _schedule_redraw(self) -> None:
        if self._redraw_job is None:
            self._redraw_job = self.after_idle(self.redraw)

    def redraw(self) -> None:
        """보이는 부분만 잘라 현재 배율로 그린다 (큰 그림도 빠르게)."""
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
            self._redraw_job = None
        c = self.canvas
        c.delete("all")
        w, h = self._view_size()
        if self.image is None:
            c.create_text(w / 2, h / 2, text="'열기' 를 눌러 그림을 고르세요\n(orthorectified.png · aruco_detected.png · registered_mosaic.png)",
                          fill="#bbbbbb", justify="center", font=("", 12))
            return
        box = self.view.visible_box(self.image.width, self.image.height, w, h)
        if box is not None:
            src, dest, size = box
            if self.view.scale >= 1.0:
                tile = self.image.resize(size, Image.NEAREST, box=src)             # 확대: 픽셀이 그대로 보이게
            else:
                tile = self.image.resize(size, Image.BILINEAR, box=src, reducing_gap=2.0)
            self._photo = ImageTk.PhotoImage(tile)
            c.create_image(dest[0], dest[1], anchor="nw", image=self._photo, tags="img")
        self.zoom_var.set(f"확대 {self.view.scale * 100:.0f} %")
        self._draw_overlays()

    def _draw_segment(self, a, b, color: str, label: str | None, tag: str, dash=None) -> None:
        c = self.canvas
        ax, ay = self.view.to_screen(*a)
        bx, by = self.view.to_screen(*b)
        c.create_line(ax, ay, bx, by, fill="black", width=4, tags=tag)
        c.create_line(ax, ay, bx, by, fill=color, width=2, dash=dash, tags=tag)
        for x, y in ((ax, ay), (bx, by)):
            for dx, dy in ((7, 0), (0, 7)):
                c.create_line(x - dx, y - dy, x + dx, y + dy, fill="black", width=3, tags=tag)
                c.create_line(x - dx, y - dy, x + dx, y + dy, fill=color, width=1, tags=tag)
        if label:
            t = c.create_text((ax + bx) / 2 + 10, (ay + by) / 2 - 12, text=label, fill="black", anchor="w", font=("", 10, "bold"), tags=tag)
            x0, y0, x1, y1 = c.bbox(t)
            r = c.create_rectangle(x0 - 4, y0 - 2, x1 + 4, y1 + 2, fill=color, outline="black", tags=tag)
            c.tag_raise(t, r)

    def _draw_overlays(self) -> None:
        self.canvas.delete("ov")
        sel = self.selected_no()
        for m in self.measurements:
            color = self.cfg.selected_color if m.no == sel else self.cfg.line_color
            self._draw_segment(m.a_px, m.b_px, color, f"{m.no}: {m.result['length_mm']:,.1f} mm", "ov")
        self._draw_live()

    def _draw_live(self) -> None:
        """시작점만 찍은 상태에서 마우스를 따라오는 선."""
        c = self.canvas
        c.delete("live")
        if self.pending is None or self.mapper is None:
            return
        if self.cursor is None:
            x, y = self.view.to_screen(*self.pending)
            c.create_oval(x - 5, y - 5, x + 5, y + 5, outline=self.cfg.pending_color, width=2, tags="live")
            return
        self._draw_segment(self.pending, self.view.to_image(*self.cursor), self.cfg.pending_color, None, "live", dash=(6, 4))

    def _set_cursor(self, pos) -> None:
        self.cursor = pos
        self._draw_live()

    def _on_motion(self, event) -> None:
        self._set_cursor((event.x, event.y))
        if self.mapper is None:
            return
        p = self.view.to_image(event.x, event.y)
        if not self.mapper.inside(p):
            self.status_var.set("그림 밖")
            return
        mm = self.mapper.to_mm([p])[0]
        if mm[0] != mm[0]:                                  # NaN: 평면 위의 위치로 바꿀 수 없는 곳
            self.status_var.set(f"픽셀 ({p[0]:.1f}, {p[1]:.1f})   →   평면 위의 위치로 바꿀 수 없는 곳")
            return
        text = f"픽셀 ({p[0]:.1f}, {p[1]:.1f})   →   평면 ({mm[0]:,.1f}, {mm[1]:,.1f}) mm"
        local = self.mapper.local_mm_per_pixel(p)
        if local is not None:
            text += f"      여기서 픽셀 하나 = 약 {local:.2f} mm"
        if self.pending is not None:
            start = self.mapper.to_mm([self.pending])[0]
            text += f"      시작점에서 {((mm[0] - start[0]) ** 2 + (mm[1] - start[1]) ** 2) ** 0.5:,.1f} mm"
        self.status_var.set(text)
