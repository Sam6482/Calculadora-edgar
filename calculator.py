import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import cm
import sympy as sp



# ==========================================
# BACKEND – Lógica de cálculo
# ==========================================
class SurfaceBackend:

    def __init__(self):
        self.x_sym, self.y_sym = sp.symbols('x y')
        self.func_lambda = None
        self.expr_str = ""

    def parse_function(self, func_str):
        try:
            expr = sp.sympify(func_str)
            self.expr_str = str(expr)

            free_symbols = expr.free_symbols
            if not free_symbols.issubset({self.x_sym, self.y_sym}):
                return False, f"Variables desconocidas. Use solo 'x' e 'y'. Encontrado: {free_symbols}"

            self.func_lambda = sp.lambdify((self.x_sym, self.y_sym), expr, modules='numpy')
            return True, "Función interpretada correctamente."

        except Exception as e:
            return False, str(e)

    def get_cartesian_data(self, x_range, y_range, points=100):
        x = np.linspace(x_range[0], x_range[1], points)
        y = np.linspace(y_range[0], y_range[1], points)
        X, Y = np.meshgrid(x, y)
        Z = self.evaluate_z(X, Y)
        return X, Y, Z

    def get_polar_data(self, limit, points=80):
        r = np.linspace(0, limit, points)
        theta = np.linspace(0, 2*np.pi, points)

        R, Theta = np.meshgrid(r, theta)

        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)

        Z = self.evaluate_z(X, Y)
        return X, Y, Z

    def evaluate_z(self, X, Y):
        try:
            Z = self.func_lambda(X, Y)

            if np.isscalar(Z):
                Z = np.full_like(X, Z)
            elif Z.shape != X.shape:
                Z = np.broadcast_to(Z, X.shape)

            return Z

        except Exception as e:
            raise ValueError(f"Error al evaluar: {e}")

    def calculate_volume(self, Z, x_range, y_range, points):
        x_axis = np.linspace(x_range[0], x_range[1], points)
        y_axis = np.linspace(y_range[0], y_range[1], points)

        Z_clean = np.nan_to_num(Z, nan=0.0)

        return np.trapz(np.trapz(Z_clean, x=x_axis, axis=1), x=y_axis)



# ==========================================
# FRONTEND – Interfaz gráfica
# ==========================================
class App3D:

    def __init__(self, root):
        self.root = root
        self.root.title("Calculadora 3D - Edición Profesional")
        self.root.geometry("1280x850")
        self.root.state('zoomed')

        self.backend = SurfaceBackend()

        # Layout
        self.frame_left = tk.Frame(root, width=340, bg="#f4f6f9", padx=15, pady=15)
        self.frame_left.pack(side=tk.LEFT, fill=tk.Y)

        self.frame_right = tk.Frame(root, bg="white")
        self.frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._init_ui()
        self._init_plot()



    # ========================================== UI ==========================================
    def _init_ui(self):

        tk.Label(self.frame_left, text="Calculadora 3D", font=("Segoe UI", 20, "bold"),
                 bg="#f4f6f9", fg="#2c3e50").pack(pady=(0, 15))

        # Función
        tk.Label(self.frame_left, text="Función z = f(x,y):", bg="#f4f6f9",
                 font=("Arial", 10, "bold")).pack(anchor="w")

        self.entry_func = tk.Entry(self.frame_left, font=("Consolas", 14), bd=1, relief="solid")
        self.entry_func.pack(fill=tk.X, pady=5, ipady=4)
        self.entry_func.insert(0, "x**2 + y**2")

        self._build_keypad()

        ttk.Separator(self.frame_left, orient='horizontal').pack(fill='x', pady=15)

        # Límites del dominio
        tk.Label(self.frame_left, text="Dominio de Visualización:", bg="#f4f6f9",
                 font=("Arial", 10, "bold")).pack(anchor="w")

        f_lims = tk.Frame(self.frame_left, bg="#f4f6f9")
        f_lims.pack(fill=tk.X, pady=5)

        self.e_xmin = self._make_entry(f_lims, "X Min:", "-3", 0)
        self.e_xmax = self._make_entry(f_lims, "X Max:", "3", 1)
        self.e_ymin = self._make_entry(f_lims, "Y Min:", "-3", 2)
        self.e_ymax = self._make_entry(f_lims, "Y Max:", "3", 3)

        # Opciones
        ttk.Separator(self.frame_left, orient='horizontal').pack(fill='x', pady=15)

        tk.Label(self.frame_left, text="Opciones de Renderizado:", bg="#f4f6f9",
                 font=("Arial", 10, "bold")).pack(anchor="w")

        self.var_circular = tk.BooleanVar(value=True)
        tk.Checkbutton(self.frame_left, text="Corte Circular (Polar Perfecto)",
                       variable=self.var_circular, bg="#f4f6f9",
                       command=self.procesar).pack(anchor="w")

        self.var_grid = tk.BooleanVar(value=True)
        tk.Checkbutton(self.frame_left, text="Mostrar Rejilla (Wireframe)",
                       variable=self.var_grid, bg="#f4f6f9",
                       command=self.procesar).pack(anchor="w")

        # ---- ZOOM ----
        ttk.Separator(self.frame_left, orient='horizontal').pack(fill='x', pady=15)
        tk.Label(self.frame_left, text="Zoom de Vista:", bg="#f4f6f9",
                 font=("Arial", 10, "bold")).pack(anchor="w")

        self.zoom = tk.DoubleVar(value=1.0)
        tk.Scale(self.frame_left, from_=0.5, to=3.0, resolution=0.1,
                 orient="horizontal", variable=self.zoom, length=200,
                 bg="#f4f6f9", command=lambda x:self.procesar()).pack(fill=tk.X, pady=5)

        # Botón
        tk.Button(self.frame_left, text="GRAFICAR", bg="#2980b9", fg="white",
                  font=("Segoe UI", 12, "bold"),
                  command=self.procesar, cursor="hand2").pack(fill=tk.X, pady=20, ipady=5)

        # Volumen
        self.lbl_vol = tk.Label(self.frame_left, text="Volumen: ---", bg="white", fg="#333",
                                font=("Consolas", 12), relief="solid", bd=1, pady=10)
        self.lbl_vol.pack(fill=tk.X)



    def _build_keypad(self):
        f = tk.Frame(self.frame_left, bg="#f4f6f9")
        f.pack(fill=tk.X, pady=5)

        keys = [
            ('x', 'x'), ('y', 'y'), ('+', ' + '), ('CLR', 'CLR'),
            ('sin', 'sin()'), ('cos', 'cos()'), ('-', ' - '), ('/', ' / '),
            ('x²', '**2'), ('√', 'sqrt()'), ('*', ' * '), ('(', '('),
            ('e', 'exp()'), ('^', '**'), ('π', 'pi'), (')', ')'),
        ]

        for i, (txt, val) in enumerate(keys):
            b = tk.Button(f, text=txt, width=4, bg="white" if val!='CLR' else "#e74c3c",
                          fg="black" if val!='CLR' else "white", font=("Arial", 9, "bold"),
                          command=lambda v=val: self.press_key(v))
            b.grid(row=i//4, column=i%4, padx=2, pady=2, sticky="nsew")

        for i in range(4):
            f.grid_columnconfigure(i, weight=1)



    def _make_entry(self, p, txt, val, row):
        tk.Label(p, text=txt, bg="#f4f6f9", width=6, anchor="e").grid(row=row, column=0, padx=2, pady=2)

        e = tk.Entry(p, width=8, justify="center")
        e.insert(0, val)
        e.grid(row=row, column=1, padx=5, pady=2, sticky="ew")
        return e



    def press_key(self, v):
        if v == 'CLR':
            self.entry_func.delete(0, tk.END)
            return

        idx = self.entry_func.index(tk.INSERT)
        self.entry_func.insert(idx, v)
        self.entry_func.focus()

        if v.endswith('()'):
            self.entry_func.icursor(idx + len(v) - 1)



    # ========================================== PLOT ==========================================
    def _init_plot(self):
        self.fig = plt.figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_right)
        self.canvas.draw()

        NavigationToolbar2Tk(self.canvas, self.frame_right).update()

        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)



    # ========================================== CÁLCULO + GRÁFICA ==========================================
    def procesar(self):

        func = self.entry_func.get()

        # ---- Límites ----
        try:
            xmin = float(self.e_xmin.get())
            xmax = float(self.e_xmax.get())
            ymin = float(self.e_ymin.get())
            ymax = float(self.e_ymax.get())

            if xmin >= xmax or ymin >= ymax:
                raise ValueError("Los límites mínimos deben ser menores a los máximos.")

        except:
            messagebox.showerror("Error", "Revise los valores numéricos de los límites.")
            return

        lims_rect = (xmin, xmax, ymin, ymax)

        ok, msg = self.backend.parse_function(func)
        if not ok:
            messagebox.showerror("Error", msg)
            return

        try:
            # ====== Cálculo — SIEMPRE RECTANGULAR ======
            pts_calc = 100
            Xc, Yc, Zc = self.backend.get_cartesian_data((xmin, xmax), (ymin, ymax), pts_calc)

            # ---- Volumen RECTANGULAR (nunca cambia con corte) ----
            vol = self.backend.calculate_volume(Zc, (xmin, xmax), (ymin, ymax), pts_calc)
            self.lbl_vol.config(text=f"Volumen Aprox: {vol:.4f} u³")

            # ====== Gráfica ======
            if self.var_circular.get():
                r_plot = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
                X_plot, Y_plot, Z_plot = self.backend.get_polar_data(r_plot, points=80)
            else:
                X_plot, Y_plot, Z_plot = Xc, Yc, Zc

            self.graficar(X_plot, Y_plot, Z_plot, lims_rect)

        except Exception as e:
            messagebox.showerror("Error Matemático", str(e))



    def graficar(self, X, Y, Z, lims_rect):
        self.ax.clear()

        xmin, xmax, ymin, ymax = lims_rect

        show_grid = self.var_grid.get()
        edge_color = 'black' if show_grid else None
        line_width = 0.2 if show_grid else 0

        self.ax.plot_surface(
            X, Y, Z,
            cmap='cool',
            edgecolor=edge_color,
            linewidth=line_width,
            alpha=0.9,
            rstride=1, cstride=1
        )

        # ---- ZOOM ----
        zoom_factor = self.zoom.get()
        max_val = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax)) * zoom_factor

        # Ejes estilo GeoGebra
        self.ax.plot([-max_val, max_val], [0, 0], [0, 0], 'k-', lw=1.5)
        self.ax.plot([0, 0], [-max_val, max_val], [0, 0], 'k-', lw=1.5)

        self.ax.set_xlim(-max_val, max_val)
        self.ax.set_ylim(-max_val, max_val)
        self.ax.set_zlabel('Z')

        self.ax.view_init(elev=30, azim=-45)
        self.canvas.draw()



# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    App3D(root)
    root.mainloop()
