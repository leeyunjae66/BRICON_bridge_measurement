"""
측정 화면 (tkinter).

  목록 (왼쪽의 탭)
    탭 하나가 사진 묶음 하나다. 사진 추가·정사영·결합은 지금 보고 있는 탭의 목록에만 적용된다.
    마커 크기와 결합 방식도 목록마다 따로 두고, 결과도 목록 이름의 폴더에 따로 저장한다.
    목록 이름, 넣은 사진, 계산 결과는 session.json 에 기억해 두었다가 창을 다시 켜면 되살린다.
  사진을 넣고 계산 돌리기
    사진 추가        : 스마트폰 사진을 지금 목록에 넣는다 (여러 장 가능)
    정사영           : 지금 목록에서 고른 사진을 1단계 프로그램으로 정사영한다 (고른 것이 없으면 아직 안 한 사진 전부)
    결합             : 지금 목록의 정사영한 사진들을 2단계 프로그램으로 하나로 정합한다 (위에서부터, 첫 번째가 기준)
    결과 열기        : 이미 만들어 둔 결과 그림을 바로 연다
  그림에서 재기
    마우스 휠        : 마우스가 있는 자리를 중심으로 확대·축소
    오른쪽 단추 끌기 : 그림 옮기기 (가운데 단추도 됨)
    왼쪽 단추        : 첫 번째는 시작점, 두 번째는 끝점 → 두 점 사이의 실제 길이(mm)가 목록에 추가된다
    Esc             : 찍어 둔 시작점 취소      Delete : 목록에서 고른 측정 지우기      F : 화면에 맞춤

정사영·결합 단추는 앞 단계의 프로그램을 그대로 실행하고 그 결과 파일을 불러올 뿐이다 (runner.py).
픽셀을 mm 로 바꾸는 계산은 coords.py, 확대·이동 계산은 viewport.py 가 한다. 이 파일은 화면만 맡는다.
"""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

import coords
import runner
from config import Config
from viewport import Viewport

Image.MAX_IMAGE_PIXELS = 400_000_000      # 큰 합친 그림도 열 수 있게


@dataclass
class Measurement:
    no: int
    a_px: tuple[float, float]
    b_px: tuple[float, float]
    result: dict                          # coords.Mapper.measure() 의 결과


@dataclass
class PhotoItem:
    iid: str
    kind: str                             # "photo" = 넣은 사진, "mosaic" = 결합 결과
    path: Path                            # 원본 사진 (결합 결과면 합친 그림)
    status: str = "원본"
    ortho: Path | None = None             # 정사영이 끝났으면 orthorectified.png


@dataclass
class PhotoList:
    """탭 하나 = 사진 묶음 하나. 정사영·결합은 지금 보고 있는 탭의 사진에만 적용된다."""
    name: str
    marker_var: tk.StringVar              # 이 목록의 마커 한 변 (mm)
    mode_var: tk.StringVar                # 이 목록의 결합 방식 (화면에 보이는 글)
    frame: ttk.Frame
    tree: ttk.Treeview
    items: list[PhotoItem]


def short_name(name: str, limit: int = 22) -> str:
    """긴 파일 이름은 가운데를 줄인다. 이름이 비슷한 사진은 끝부분이 다르므로 끝을 남긴다."""
    if len(name) <= limit:
        return name
    tail = limit - 6
    return f"{name[:5]}…{name[-tail:]}"


class MeasureApp(tk.Tk):
    def __init__(self, cfg: Config | None = None, image_path: str | None = None):
        super().__init__()
        self.cfg = cfg or Config()
        self.title("바닥판 치수 측정")
        self.geometry(self.cfg.window_size)
        self.minsize(1100, 600)
        self.image: Image.Image | None = None
        self.mapper: coords.Mapper | None = None
        self.view = Viewport()
        self.measurements: list[Measurement] = []
        self.lists: list[PhotoList] = []                      # 탭마다 하나
        self.pending: tuple[float, float] | None = None       # 찍어 둔 시작점 (그림 픽셀)
        self.cursor: tuple[float, float] | None = None        # 마우스 위치 (화면)
        self.quiet = False                                    # True 면 알림 창을 띄우지 않는다 (자동 시험용)
        self.last_message = ""                                # 마지막 알림 내용
        self.last_log = ""                                    # 마지막으로 실행한 프로그램이 찍은 글
        self._photo = None                                    # 화면에 그린 그림 (지워지지 않게 붙잡아 둔다)
        self._next_no = 1
        self._next_item = 1
        self._redraw_job = None
        self._drag = None
        self._fit_scale = 1.0
        self._busy = False
        self._showing = False
        self._restoring = False
        self._shown_iid: str | None = None                    # 지금 가운데에 떠 있는 목록 항목
        self._action_buttons: list[ttk.Button] = []
        self._build()
        self._load_session()
        if image_path:
            self.after(80, lambda: self.open_image(image_path))

    # ---------------------------------------------------------------- 화면 짜기
    def _build(self) -> None:
        # 첫째 줄: 사진을 넣고 계산을 돌리는 단추
        work = ttk.Frame(self, padding=(6, 5, 6, 2))
        work.pack(side="top", fill="x")
        self._button(work, "사진 추가", self.add_photos)
        self._button(work, "목록에서 빼기", self.remove_selected_photos)
        ttk.Separator(work, orient="vertical").pack(side="left", fill="y", padx=8)
        # 마커 크기와 결합 방식은 목록마다 따로다. 탭을 바꾸면 그 목록의 값이 이 칸에 나타난다 (_on_list_change).
        ttk.Label(work, text="마커 한 변 (mm)").pack(side="left")
        self.marker_entry = ttk.Entry(work, width=7, justify="right")
        self.marker_entry.pack(side="left", padx=(4, 4))
        self._button(work, "정사영", self.orthorectify)
        ttk.Separator(work, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(work, text="결합 방식").pack(side="left")
        self.mode_box = ttk.Combobox(work, values=list(runner.MODES.values()), state="readonly", width=24)
        self.mode_box.pack(side="left", padx=(4, 4))
        self._button(work, "결합", self.register)
        self.scope_var = tk.StringVar(value="")
        ttk.Label(work, textvariable=self.scope_var, foreground="#0a5c9c").pack(side="left", padx=(8, 0))
        ttk.Separator(work, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(work, text="실행 기록", command=self.show_log).pack(side="left", padx=2)

        # 둘째 줄: 그림 보기와 측정
        bar = ttk.Frame(self, padding=(6, 2, 6, 4))
        bar.pack(side="top", fill="x")
        for text, cmd in (("결과 열기", self.ask_open), ("화면 맞춤 (F)", self.fit), ("실제 크기 (1:1)", self.actual_size),
                          ("선택 지우기 (Del)", self.delete_selected), ("모두 지우기", self.clear_all)):
            ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=2)
        ttk.Label(bar, text="   왼쪽 단추: 시작점 → 끝점   |   휠: 확대·축소   |   오른쪽 단추 끌기: 이동   |   Esc: 시작점 취소",
                  foreground="#555555").pack(side="left", padx=8)

        status = ttk.Frame(self, padding=(6, 2))
        status.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="'사진 추가' 로 사진을 넣거나, '결과 열기' 로 만들어 둔 결과를 여세요.")
        self.zoom_var = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")
        ttk.Label(status, textvariable=self.zoom_var).pack(side="right")

        # 왼쪽: 사진 목록
        left = ttk.Frame(self, padding=6, width=self.cfg.photo_panel_width)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        ttk.Label(left, text="사진 목록", font=("", 10, "bold")).pack(anchor="w")
        tools = ttk.Frame(left)
        tools.pack(fill="x", pady=(2, 4))
        self._button(tools, "새 목록", self.new_list)
        self._button(tools, "이름 바꾸기", self.ask_rename_list)
        self._button(tools, "목록 지우기", self.delete_list)
        self.book = ttk.Notebook(left)                    # 탭 하나가 목록 하나
        self.book.pack(fill="both", expand=True, pady=(0, 6))
        self.book.bind("<<NotebookTabChanged>>", lambda e: self._on_list_change())
        ttk.Label(left, text="사진 추가·정사영·결합은 지금 보고 있는 탭의 목록에만 적용됩니다.\n"
                             "사진을 누르면 가운데에 보입니다. 결합은 위에서부터 순서대로, 첫 번째 사진이 기준입니다.",
                  wraplength=self.cfg.photo_panel_width - 16, justify="left", foreground="#555555").pack(anchor="w")

        # 오른쪽: 그림 정보와 측정 목록
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

        # 가운데: 그림
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
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _button(self, parent, text: str, command) -> ttk.Button:
        """계산이 도는 동안 눌리지 않게 묶어 두는 단추."""
        b = ttk.Button(parent, text=text, command=command)
        b.pack(side="left", padx=2)
        self._action_buttons.append(b)
        return b

    def _notify(self, title: str, text: str, error: bool = False) -> None:
        self.last_message = text
        self.status_var.set(text.splitlines()[0] if text else "")
        if not self.quiet:
            (messagebox.showerror if error else messagebox.showinfo)(title, text, parent=self)

    # ---------------------------------------------------------------- 목록 (탭)
    @property
    def current(self) -> PhotoList:
        """지금 보고 있는 탭의 목록."""
        return self.lists[self.book.index("current")]

    # 아래 네 가지는 '지금 목록' 의 것을 가리킨다 (목록이 하나뿐이던 때의 이름을 그대로 쓴다)
    @property
    def photos(self) -> list[PhotoItem]:
        return self.current.items

    @property
    def photo_tree(self) -> ttk.Treeview:
        return self.current.tree

    @property
    def marker_var(self) -> tk.StringVar:
        return self.current.marker_var

    @property
    def mode_var(self) -> tk.StringVar:
        return self.current.mode_var

    def _free_name(self) -> str:
        used = {lst.name for lst in self.lists}
        n = 1
        while f"목록 {n}" in used:
            n += 1
        return f"목록 {n}"

    def new_list(self, name: str | None = None, marker: str | None = None, mode: str | None = None, select: bool = True) -> PhotoList:
        """새 목록(탭)을 만든다. 마커 크기와 결합 방식은 지금 목록의 값을 물려받는다 (첫 목록은 설정의 처음 값)."""
        if marker is None:
            marker = self.current.marker_var.get() if self.lists else f"{self.cfg.marker_size_mm:g}"
        if mode not in runner.MODES:
            mode = self.mode() if self.lists else self.cfg.registration_mode
        name = (name or "").strip()
        if not name or name in {lst.name for lst in self.lists}:
            name = self._free_name()
        frame = ttk.Frame(self.book)
        tree = ttk.Treeview(frame, columns=("name", "status"), show="headings", selectmode="extended")
        tree.heading("name", text="이름")
        tree.heading("status", text="상태")
        tree.column("name", width=self.cfg.photo_panel_width - 110, stretch=True)
        tree.column("status", width=80, anchor="center", stretch=False)
        tree.pack(fill="both", expand=True)
        tree.bind("<<TreeviewSelect>>", lambda e: self._on_photo_select())
        lst = PhotoList(name, tk.StringVar(value=marker), tk.StringVar(value=runner.MODES.get(mode, runner.MODES["marker"])), frame, tree, [])
        self.lists.append(lst)
        self.book.add(frame, text=name)
        if select:
            self.book.select(frame)
            self.update_idletasks()
            self._on_list_change()
            self._save_session()
        return lst

    def ask_rename_list(self) -> None:
        from tkinter import simpledialog
        name = simpledialog.askstring("이름 바꾸기", "목록의 새 이름:", initialvalue=self.current.name, parent=self)
        if name is not None:
            self.rename_list(name)

    def rename_list(self, name: str) -> bool:
        """지금 목록의 이름을 바꾼다. 이미 만든 결과는 예전 이름의 폴더에 그대로 있고, 새 결과부터 새 이름의 폴더에 저장된다."""
        name = name.strip()
        lst = self.current
        if not name or name == lst.name:
            return False
        if name in {x.name for x in self.lists}:
            self._notify("이름 바꾸기", f"'{name}' 이라는 목록이 이미 있습니다. 다른 이름을 쓰세요.", error=True)
            return False
        lst.name = name
        self.book.tab(lst.frame, text=name)
        self._on_list_change()
        self._save_session()
        return True

    def delete_list(self) -> None:
        """지금 목록을 화면에서 지운다. 사진과 계산 결과 파일은 지우지 않는다."""
        lst = self.current
        if lst.items and not self.quiet and not messagebox.askyesno(
                "목록 지우기", f"'{lst.name}' 목록을 화면에서 지울까요?\n사진과 계산 결과 파일은 지워지지 않습니다.", parent=self):
            return
        self.lists.remove(lst)
        self.book.forget(lst.frame)
        lst.frame.destroy()
        if not self.lists:
            self.new_list(select=False)                   # 목록은 늘 하나 이상 있다
        self.update_idletasks()
        self._on_list_change()
        self._save_session()

    def _on_list_change(self) -> None:
        """탭을 바꿨을 때: 입력 칸을 그 목록의 값으로 바꾸고, 그 목록의 그림을 가운데에 띄운다."""
        if not self.lists:
            return
        lst = self.current
        self.marker_entry.configure(textvariable=lst.marker_var)
        self.mode_box.configure(textvariable=lst.mode_var)
        self.scope_var.set(f"←  '{lst.name}' 에 적용")
        if self._restoring or self._busy or self._showing:
            return
        sel = lst.tree.selection()
        if len(sel) == 1:
            if sel[0] != self._shown_iid:                 # 이미 떠 있는 그림을 또 읽지 않는다
                self.show_item(self._item(sel[0]))
        elif lst.items and not sel:
            # 고른 사진이 없으면 결합 결과(있으면)나 첫 사진을 골라 보여 준다
            item = next((it for it in lst.items if it.kind == "mosaic"), lst.items[0])
            self._select_item(item)
            self.show_item(item)

    def _list_of(self, item: PhotoItem) -> PhotoList | None:
        return next((lst for lst in self.lists if item in lst.items), None)

    def marker_size(self) -> float:
        """지금 목록의 마커 크기. 숫자가 아니거나 0 이하면 ValueError."""
        value = float(self.marker_var.get().strip().replace(",", "."))
        if not value > 0:
            raise ValueError("0 보다 커야 합니다")
        return value

    def mode(self) -> str:
        return next((k for k, v in runner.MODES.items() if v == self.mode_var.get()), "marker")

    # ---------------------------------------------------------------- 목록 기억하기 (창을 다시 켜면 되살린다)
    def _save_session(self) -> None:
        if self._restoring:
            return
        data = {"active": self.book.index("current") if self.lists else 0, "lists": []}
        for lst in self.lists:
            mode = next((k for k, v in runner.MODES.items() if v == lst.mode_var.get()), "marker")
            photos = [{"path": str(it.path), "ortho": str(it.ortho) if it.ortho else None} for it in lst.items if it.kind == "photo"]
            mosaic = next((str(it.path) for it in lst.items if it.kind == "mosaic"), None)
            data["lists"].append({"name": lst.name, "marker_size_mm": lst.marker_var.get(), "registration_mode": mode,
                                  "photos": photos, "mosaic": mosaic})
        try:
            with open(self.cfg.session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _load_session(self) -> None:
        """지난번에 쓰던 목록을 되살린다. 결과 파일이 그사이 없어졌으면 그 사진은 '원본' 으로 돌아간다."""
        self._restoring = True
        active = 0
        try:
            with open(self.cfg.session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            active = int(data.get("active", 0))
            for d in data.get("lists", []):
                lst = self.new_list(str(d.get("name", "")), str(d.get("marker_size_mm", f"{self.cfg.marker_size_mm:g}")),
                                    d.get("registration_mode"), select=False)
                for ph in d.get("photos", []):
                    item = PhotoItem(f"p{self._next_item}", "photo", Path(ph["path"]))
                    self._next_item += 1
                    ortho = Path(ph["ortho"]) if ph.get("ortho") else None
                    if ortho is not None and ortho.is_file() and (ortho.parent / "homography.json").is_file():
                        item.ortho, item.status = ortho, "정사영 완료"
                    lst.items.append(item)
                    lst.tree.insert("", "end", iid=item.iid, values=(short_name(item.path.name), item.status))
                mosaic = Path(d["mosaic"]) if d.get("mosaic") else None
                if mosaic is not None and mosaic.is_file() and (mosaic.parent / "registration.json").is_file():
                    n = sum(1 for it in lst.items if it.ortho is not None)
                    item = PhotoItem(f"m{self._next_item}", "mosaic", mosaic, "결합 완료")
                    self._next_item += 1
                    lst.items.append(item)
                    lst.tree.insert("", "end", iid=item.iid, values=(f"[결합] {n}장", item.status))
        except (OSError, ValueError, TypeError, KeyError):
            pass                                          # 기억해 둔 것이 없거나 읽을 수 없으면 빈 목록으로 시작
        if not self.lists:
            self.new_list(select=False)
        self.book.select(self.lists[min(max(active, 0), len(self.lists) - 1)].frame)
        self.update_idletasks()
        self._restoring = False
        self.after(120, self._on_list_change)             # 창이 다 그려진 뒤에 그 목록의 그림을 띄운다

    def _on_close(self) -> None:
        self._save_session()
        self.destroy()

    # ---------------------------------------------------------------- 사진 목록
    def add_photos(self, paths: list[str] | None = None) -> None:
        if paths is None:
            start = self.cfg.photo_dir
            paths = filedialog.askopenfilenames(
                parent=self, title="정사영할 사진 선택 (여러 장 가능)",
                initialdir=str(start) if Path(start).is_dir() else None,
                filetypes=[("사진", "*.jpg *.jpeg *.png *.heic *.heif"), ("모든 파일", "*.*")])
        lst = self.current                                # 지금 보고 있는 탭의 목록에 넣는다
        have = {it.path.resolve() for it in lst.items if it.kind == "photo"}
        first = None
        for p in paths or []:
            path = Path(p)
            if path.resolve() in have:
                continue
            have.add(path.resolve())
            item = PhotoItem(f"p{self._next_item}", "photo", path)
            self._next_item += 1
            # 결합 결과는 늘 목록 맨 아래에 두므로 사진은 그 앞에 넣는다
            pos = sum(1 for it in lst.items if it.kind == "photo")
            lst.items.insert(pos, item)
            lst.tree.insert("", pos, iid=item.iid, values=(short_name(path.name), item.status))
            first = first or item
        if first is not None:
            lst.tree.selection_set(first.iid)
            lst.tree.see(first.iid)
            self._save_session()

    def remove_selected_photos(self) -> None:
        lst = self.current
        gone = set(lst.tree.selection())
        if not gone:
            return
        lst.items[:] = [it for it in lst.items if it.iid not in gone]
        lst.tree.delete(*gone)
        self._save_session()

    def _item(self, iid: str) -> PhotoItem | None:
        return next((it for lst in self.lists for it in lst.items if it.iid == iid), None)

    def _set_status(self, item: PhotoItem, status: str) -> None:
        item.status = status
        lst = self._list_of(item)
        if lst is not None and lst.tree.exists(item.iid):
            lst.tree.set(item.iid, "status", status)

    def _on_photo_select(self) -> None:
        if self._showing or self._busy:
            return
        sel = self.photo_tree.selection()
        if len(sel) == 1:
            self.show_item(self._item(sel[0]))

    def show_item(self, item: PhotoItem | None) -> None:
        """목록의 사진을 가운데에 띄운다. 정사영이 끝났으면 펴진 그림(측정 가능), 아니면 원본 사진(보기만)."""
        if item is None:
            return
        self._shown_iid = item.iid
        if item.kind == "mosaic":
            self.open_image(str(item.path), ask_on_error=False)
        elif item.ortho is not None:
            self.open_image(str(item.ortho), ask_on_error=False)
        else:
            self.show_photo(item.path)

    def show_photo(self, path: Path) -> bool:
        """정사영하기 전의 사진을 보기만 한다 (좌표 정보가 없으므로 치수는 잴 수 없다)."""
        p = Path(path)
        try:
            if p.suffix.lower() in (".heic", ".heif"):
                import pillow_heif                       # 설치돼 있을 때만 (없으면 아래에서 알림)
                pillow_heif.register_heif_opener()
            with Image.open(p) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")     # 세워 찍은 사진을 바로 세운다
        except ImportError:
            self._notify("사진 보기", f"HEIC 사진은 이 화면에서 미리 볼 수 없습니다 ({p.name}).\n정사영은 그대로 할 수 있습니다.")
            return False
        except (OSError, ValueError) as e:
            self._notify("사진 보기", f"사진을 읽지 못했습니다.\n{p.name}\n\n{e}", error=True)
            return False
        self._set_image(image, None)
        self.title(f"바닥판 치수 측정 - {p.name}")
        self.info_var.set(f"파일: {p.name}\n종류: 넣은 사진 (정사영 전)\n크기: {image.width} × {image.height} 픽셀")
        self.notes_var.set("· 아직 정사영하지 않아 치수를 잴 수 없습니다. 마커 크기를 확인하고 '정사영' 을 누르세요.")
        self.status_var.set("정사영 전의 사진입니다. '정사영' 단추를 누르면 펴진 그림이 뜨고 치수를 잴 수 있습니다.")
        self.fit()
        return True

    # ---------------------------------------------------------------- 앞 단계 프로그램 실행
    def _start(self, title: str, work, done, sync: bool = False) -> None:
        """work(progress) 를 실행하고 끝나면 done(결과) 를 부른다. 계산하는 동안 화면이 멈추지 않게 따로 돌린다.

        sync=True 면 그 자리에서 끝까지 실행한다 (자동 시험용).
        """
        if self._busy:
            return
        self._save_session()
        self.status_var.set(title)
        if sync:
            done(work(self.status_var.set))
            return
        self._busy = True
        for b in self._action_buttons:
            b.state(["disabled"])
        self.configure(cursor="watch")
        q: queue.Queue = queue.Queue()

        def worker():
            try:
                q.put(("done", work(lambda text: q.put(("progress", text)))))
            except Exception as e:                       # 화면이 '계산 중' 에서 멈춰 있지 않게 어떤 오류든 돌려준다
                q.put(("error", e))

        def poll():
            try:
                while True:
                    kind, value = q.get_nowait()
                    if kind == "progress":
                        self.status_var.set(value)
                        continue
                    self._busy = False
                    for b in self._action_buttons:
                        b.state(["!disabled"])
                    self.configure(cursor="")
                    if kind == "error":
                        self._notify(title, f"실행 중 문제가 생겼습니다.\n{value}", error=True)
                    else:
                        done(value)
                    return
            except queue.Empty:
                self.after(100, poll)

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, poll)

    def orthorectify(self, sync: bool = False) -> None:
        """지금 목록에서 고른 사진(없으면 아직 안 한 사진 전부)을 1단계 프로그램으로 정사영하고, 결과를 띄운다."""
        lst = self.current                                # 누른 순간의 목록. 계산 중에 탭을 바꿔도 이 목록에 적용된다
        photos = [it for it in lst.items if it.kind == "photo"]
        if not photos:
            self._notify("정사영", f"'{lst.name}' 목록에 사진이 없습니다. 먼저 '사진 추가' 로 사진을 넣으세요.")
            return
        sel = set(lst.tree.selection())
        targets = [it for it in photos if it.iid in sel] or [it for it in photos if it.ortho is None]
        if not targets:
            self._notify("정사영", "정사영할 사진이 없습니다. 다시 하려면 목록에서 사진을 고른 뒤 누르세요.")
            return
        try:
            size = self.marker_size()
        except ValueError:
            self._notify("정사영", f"마커 한 변의 길이를 mm 숫자로 넣으세요 (지금 값: '{self.marker_var.get()}').", error=True)
            return
        for it in targets:
            self._set_status(it, "계산 중")
        # 결과는 목록 이름의 폴더에 따로 저장한다: 1단계 output\<목록 이름>\<사진 이름>
        out_root = runner.STAGE1_DIR / "output" / runner.safe_name(lst.name)

        def work(progress):
            out = []
            for k, it in enumerate(targets, 1):
                progress(f"[{lst.name}] 정사영 계산 중 ({k}/{len(targets)}): {it.path.name}  (마커 {size:g} mm)")
                try:
                    out.append((it, runner.run_orthorectification(it.path, size, out_dir=out_root / runner.safe_name(it.path.stem))))
                except runner.RunnerError as e:
                    out.append((it, e))
            return out

        def done(results):
            logs, failed, last_ok = [], [], None
            for it, res in results:
                if isinstance(res, runner.RunResult):
                    logs.append(f"===== 정사영: {it.path.name} (마커 {size:g} mm)\n{res.output}")
                    if res.ok:
                        it.ortho = res.result_image
                        self._set_status(it, "정사영 완료")
                        last_ok = it
                        continue
                    failed.append((it, res.summary))
                else:
                    failed.append((it, str(res)))
                it.ortho = None
                self._set_status(it, "실패")
            self.last_log = "\n\n".join(logs)
            self._save_session()
            if last_ok is not None:
                self._select_item(last_ok)
                self.show_item(last_ok)
                self.status_var.set(f"[{lst.name}] 정사영 {len(results) - len(failed)}장 완료. 결과: {last_ok.ortho.parent.parent}")
            if failed:
                self._notify("정사영", f"{len(failed)}장을 정사영하지 못했습니다.\n\n" + "\n\n".join(f"[{it.path.name}]\n{why}" for it, why in failed), error=True)

        self._start(f"[{lst.name}] 정사영 계산 중… ({len(targets)}장)", work, done, sync)

    def register(self, sync: bool = False) -> None:
        """지금 목록의 정사영한 사진들을 2단계 프로그램으로 정합하고, 합친 그림을 띄운다. 다른 목록의 사진은 쓰지 않는다."""
        lst = self.current
        ready = [it for it in lst.items if it.kind == "photo" and it.ortho is not None]
        sel = set(lst.tree.selection())
        chosen = [it for it in ready if it.iid in sel]
        targets = chosen if len(chosen) >= 2 else ready
        if len(targets) < 2:
            self._notify("결합", f"결합하려면 '{lst.name}' 목록에 정사영한 사진이 2장 이상 필요합니다 (지금 {len(ready)}장).\n"
                                 "사진을 추가하고 '정사영' 을 먼저 누르세요.")
            return
        mode = self.mode()
        name = f"gui_{runner.safe_name(lst.name)}_{len(targets)}img_{mode}"      # 2단계 output 아래의 폴더 이름

        def work(progress):
            progress(f"[{lst.name}] 결합 계산 중… ({len(targets)}장, {runner.MODES[mode]})")
            return runner.run_registration([it.ortho for it in targets], mode, name=name)

        def done(res: runner.RunResult):
            self.last_log = f"===== 결합: {lst.name}, {len(targets)}장, {runner.MODES[mode]}\n{res.output}"
            if not res.ok:
                self._notify("결합", f"결합하지 못했습니다.\n\n{res.summary}", error=True)
                return
            for it in [x for x in lst.items if x.kind == "mosaic"]:         # 결합 결과는 목록마다 하나만 둔다
                lst.items.remove(it)
                lst.tree.delete(it.iid)
            item = PhotoItem(f"m{self._next_item}", "mosaic", res.result_image, "결합 완료")
            self._next_item += 1
            lst.items.append(item)
            lst.tree.insert("", "end", iid=item.iid, values=(f"[결합] {len(targets)}장", item.status))
            self._save_session()
            self._select_item(item)
            self.show_item(item)
            self.status_var.set(f"[{lst.name}] 결합 완료. 결과: {res.out_dir}")
            notice = "\n".join(ln for ln in res.summary.splitlines() if any(k in ln for k in ("[주의]", "[실패]", "실패 이유")))
            if notice:
                self._notify("결합", "결합은 됐지만 확인할 점이 있습니다.\n\n" + notice)

        self._start(f"[{lst.name}] 결합 계산 중… ({len(targets)}장)", work, done, sync)

    def _select_item(self, item: PhotoItem) -> None:
        """그 사진이 든 탭을 앞으로 내고 목록에서 고른다."""
        lst = self._list_of(item)
        if lst is None:
            return
        self._showing = True                              # 탭을 바꾸거나 목록을 고르는 것만으로 그림을 또 열지 않게
        try:
            self.book.select(lst.frame)
            lst.tree.selection_set(item.iid)
            lst.tree.see(item.iid)
            self.update()                                 # '골랐다' 는 알림이 여기서 처리되고 지나가게 한다
        finally:
            self._showing = False

    def show_log(self) -> None:
        """마지막으로 실행한 프로그램이 찍은 글을 새 창에 보여 준다."""
        win = tk.Toplevel(self)
        win.title("실행 기록")
        win.geometry("900x600")
        text = tk.Text(win, wrap="none", font=("Consolas", 10))
        bar = ttk.Scrollbar(win, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        text.insert("1.0", self.last_log or "아직 실행한 기록이 없습니다. '정사영' 이나 '결합' 을 누르면 여기에 남습니다.")
        text.configure(state="disabled")

    # ---------------------------------------------------------------- 결과 그림 열기
    def ask_open(self) -> None:
        start = self.mapper.image_path.parent if self.mapper else self.cfg.start_dir
        path = filedialog.askopenfilename(
            parent=self, title="측정할 결과 그림 선택",
            initialdir=str(start) if Path(start).is_dir() else None,
            filetypes=[("그림", "*.png *.jpg *.jpeg"), ("모든 파일", "*.*")])
        if path:
            self._shown_iid = None                        # 목록의 사진이 아니라 따로 연 그림
            self.open_image(path)

    def _set_image(self, image: Image.Image, mapper: coords.Mapper | None) -> None:
        self.image, self.mapper = image, mapper
        self.measurements.clear()
        self.tree.delete(*self.tree.get_children())
        self.pending, self._next_no = None, 1
        self.detail_var.set("")
        self.update_idletasks()

    def open_image(self, path: str, ask_on_error: bool = True) -> bool:
        """결과 그림과 그 옆의 좌표 정보를 읽는다. 좌표 정보를 못 찾으면 열지 않는다 (축척을 추측하지 않는다)."""
        p = Path(path)
        try:
            with Image.open(p) as im:
                image = im.convert("RGB")
        except (OSError, ValueError) as e:
            self._notify("그림 열기", f"그림을 읽지 못했습니다.\n{p.name}\n\n{e}", error=True)
            return False
        try:
            mapper = coords.find_mapper(p, image.size)
        except coords.CoordError as e:
            if not ask_on_error or self.quiet:
                self.last_message = str(e)
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
                self._notify("좌표 정보", str(e2), error=True)
                return False
        self._set_image(image, mapper)
        self.title(f"바닥판 치수 측정 - {p.parent.name}\\{p.name}")
        scale = "자리마다 다름 (마우스를 올리면 아래에 표시)" if mapper.kind == "photo" else f"{mapper.plane_from_pixel[0, 0]:.4g} mm/픽셀"
        self.info_var.set(f"파일: {p.parent.name}\\{p.name}\n종류: {mapper.kind_name}\n크기: {mapper.width} × {mapper.height} 픽셀\n"
                          f"축척: {scale}\n좌표 정보: {mapper.meta_path.name}\n좌표계: {mapper.frame}")
        notes = list(mapper.notes)
        if mapper.marker_corners_photo_px is not None:
            notes.append(f"'예상 오차' 는 마커 모서리를 {self.cfg.corner_noise_px:g} 픽셀 틀리게 읽었다고 가정한 값입니다. 실제 오차는 더 클 수 있습니다.")
        self.notes_var.set("\n".join(f"· {n}" for n in notes) or "-")
        self.status_var.set("그림을 열었습니다. 휠로 확대한 뒤 왼쪽 단추로 시작점과 끝점을 차례로 찍으세요.")
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
        if self.image is None:
            return
        if self.mapper is None:
            self.status_var.set("정사영 전의 사진이라 치수를 잴 수 없습니다. '정사영' 단추를 먼저 누르세요.")
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
        if self.mapper is None:
            return None
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
            c.create_text(w / 2, h / 2, text="'사진 추가' 로 사진을 넣고 '정사영' 을 누르세요\n\n이미 만든 결과는 '결과 열기' 로 엽니다\n"
                                             "(orthorectified.png · aruco_detected.png · registered_mosaic.png)",
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
        if self.mapper is None or self._busy:
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
