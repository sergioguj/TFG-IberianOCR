import os
import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog, Toplevel, ttk

# Importar rutas fijas
from common import DATASET_DIR, TRAIN_OUT_DIR

PY = sys.executable
SCRIPT_DIR = Path(__file__).resolve().parent

# Funciones de Utilidad 

def run(cmd, env=None):
    """Ejecuta un comando en un proceso separado."""
    cmd_str = " ".join(map(str, cmd))
    print(f"→ {cmd_str}")
    try:
        # Usamos check=True para que lance excepción si falla
        subprocess.run(cmd, check=True, env=env, cwd=SCRIPT_DIR)
        return True
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error de Proceso", 
                             f"El proceso falló con código {e.returncode}.\nComando: {cmd_str}")
        return False
    except FileNotFoundError:
        messagebox.showerror("Error de Ejecutable", 
                             "No se encontró el ejecutable/script. Revisa las rutas.")
        return False

# 
#  LOGICA DE LA GUI 

class AppLauncher:
    def __init__(self, root):
        self.root = root
        root.title("Lanzador Iberian OCR")
        root.geometry("450x300")
        
        # Estilo para los botones 
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 10), padding=10)

        # Marco principal
        main_frame = ttk.Frame(root, padding="20 20 20 20")
        main_frame.pack(fill='both', expand=True)

        # Título
        ttk.Label(main_frame, text="Iberian OCR", 
                  font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Botones
        self.create_buttons(main_frame)
        
    def create_buttons(self, container):
        button_info = [
            ("1. Preparar Dataset", self.run_prepare_dataset),
            ("2. Entrenar Modelo", self.run_train_model),
            ("3. Analizar Resultados", self.run_analyze_results),
            ("4. Lanzar GUI de Predicción (OCR)", self.run_predict_gui),
        ]
        
        for text, command in button_info:
            ttk.Button(container, text=text, command=command, width=40).pack(pady=5)

    # Lógica de las Opciones

    def run_prepare_dataset(self):
        """Opción 1: Recopila y aumenta los datos."""
        
        seeds_input = simpledialog.askstring("Opción 1: Preparar Dataset", 
                                             "Ruta de las imágenes 'seeds' (dejar vacío para 'seeds'):",
                                             initialvalue=str(SCRIPT_DIR / "seeds"))
        if seeds_input is None: return

        use_font = messagebox.askyesno("Opción 1", "¿Mezclar muestras generadas por la fuente 'iberian.ttf'?")
        export_imgs = messagebox.askyesno("Opción 1", "¿Exportar imágenes del dataset para revisión?")
        
        seeds_abs = str(Path(seeds_input).expanduser().resolve()) if seeds_input else str(SCRIPT_DIR / "seeds")
        
        args = [PY, str(SCRIPT_DIR / "prepare_dataset.py"), "--seeds-dir", seeds_abs]
        if use_font:
            args += ["--use-font"]
        if export_imgs:
            args += ["--export-images", "--export-format", "jpg"] # Formato fijo a JPG para GUI

        if run(args):
            messagebox.showinfo("Éxito", f"Dataset regenerado en: {DATASET_DIR}")

    def run_train_model(self):
        """Opción 2: Entrenar el modelo."""
        
        # Pedir Epochs y Batch Size
        epochs = simpledialog.askinteger("Opción 2: Entrenamiento", "Número de Epochs (recomendado: 100):", initialvalue=100, minvalue=1)
        if epochs is None: return

        batch = simpledialog.askinteger("Opción 2: Entrenamiento", "Batch Size:", initialvalue=32, minvalue=1)
        if batch is None: return

        TRAIN_OUT_DIR.mkdir(parents=True, exist_ok=True)
        
        args = [PY, str(SCRIPT_DIR / "train_model.py"), "--epochs", str(epochs), "--batch-size", str(batch)]
        
        if run(args):
            messagebox.showinfo("Éxito", f"Entrenamiento guardado en: {TRAIN_OUT_DIR}")


    def run_analyze_results(self):
        """Opción 3: Analizar resultados."""
        
        # Preguntar por opciones de visualización en una subventana
        options_window = Toplevel(self.root)
        options_window.title("Opciones de Análisis")
        
        also_samples_var = tk.BooleanVar(value=False)
        show_graphs_var = tk.BooleanVar(value=False) # Para mostrar gráficas interactivamente

        ttk.Checkbutton(options_window, text="Dibujar muestras por clase", variable=also_samples_var).pack(pady=5, padx=10, anchor='w')
        ttk.Checkbutton(options_window, text="Mostrar gráficas interactivas", variable=show_graphs_var).pack(pady=5, padx=10, anchor='w')

        def execute_analysis():
            also = also_samples_var.get()
            show = show_graphs_var.get()
            
            analyze_py = str(SCRIPT_DIR / "analyze_results.py")
            args = [PY, analyze_py]
            if also: args += ["--also-samples"]
            
            env = None
            if show:
                env = dict(os.environ)
                env["IBERIAN_SHOW"] = "1"
            
            options_window.destroy()
            if run(args, env=env):
                messagebox.showinfo("Análisis Completo", "Resultados generados en la carpeta de la ejecución.")

        ttk.Button(options_window, text="Ejecutar Análisis", command=execute_analysis).pack(pady=15)
        options_window.transient(self.root) # Mantener sobre la ventana principal
        options_window.grab_set()
        self.root.wait_window(options_window)


    def run_predict_gui(self):
        """Opción 4: Lanzar la GUI de predicción (OCR)."""
        print("Lanzando la GUI de predicción...")
        predict_py = str(SCRIPT_DIR / "predict_gui.py")
        args = [PY, predict_py]
        
        if not run(args):
            messagebox.showerror("Fallo al Lanzar", "No se pudo iniciar la interfaz de predicción. Revisa el modelo (.keras).")


def main():
    root = tk.Tk()
    app = AppLauncher(root)
    root.mainloop()

if __name__ == "__main__":
    main()