import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFont
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

from common import CLASS_NAMES, TRAIN_OUT_DIR, CLASS_TO_CHAR_MAP, THRESH, IMG_SIZE, BASE_DIR

# cons
MODEL_PATH = TRAIN_OUT_DIR / "iberian_mlp_demo.keras"
IMG_PREVIEW_SIZE = 400
INNER_IMG_SIZE = 18

class OCR_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("iberianOCR")
        
        # Estilos
        self.style = ttk.Style()
        self.style.theme_use('clam') 
        
        self.model = None
        self.class_names = CLASS_NAMES
        self.char_map = CLASS_TO_CHAR_MAP
        self.pil_image = None
        self.show_bboxes = tk.BooleanVar(value=False) 
        
        # Configuración de fuente
        self.SYSTEM_FONT_NAME = "iberian" 
        self.SYSTEM_FONT_SIZE = 40  

        # Interfaz Gráfica
        top_frame = tk.Frame(root)
        top_frame.pack(pady=15) 
        
        # Botones
        tk.Button(top_frame, text="Cargar Imagen", command=self.open_image, bg="#e1e1e1").pack(side=tk.LEFT, padx=5)
        self.btn_predict = tk.Button(top_frame, text="Predecir Símbolos", command=self.predict_image, state=tk.DISABLED, bg="#e1e1e1")
        self.btn_predict.pack(side=tk.LEFT, padx=5)
        
        tk.Frame(top_frame, width=10).pack(side=tk.LEFT) 
        
        # Checkbox
        self.chk_box = ttk.Checkbutton(
            top_frame, 
            text="Ver Cajas", 
            variable=self.show_bboxes,
            onvalue=True, 
            offvalue=False,
            cursor="hand2",
            command=self.on_checkbox_toggle
        )
        self.chk_box.pack(side=tk.LEFT, padx=5)
        
        # Imagen de entrada
        self.lbl_image = tk.Label(root, bg="#f0f0f0") 
        self.lbl_image.pack(pady=5, padx=10)
        
        tk.Label(root, text="Predicción:", font=("Arial", 12)).pack(pady=(10, 0))
        
        # Usamos un Frame para contener el texto y el scrollbar
        self.frm_text_container = tk.Frame(root)
        self.frm_text_container.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        # Scrollbar vertical
        scrollbar = tk.Scrollbar(self.frm_text_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Widget de Texto 
        self.txt_results = tk.Text(
            self.frm_text_container, 
            height=5,             # Altura inicial en líneas
            width=50,             # Ancho inicial en caracteres
            font=("Arial", 12),   # Fuente por defecto para mensajes de error
            state=tk.DISABLED,    # Solo lectura por defecto
            wrap=tk.CHAR,         # Cortar línea al llegar al borde 
            yscrollcommand=scrollbar.set,
            bd=0, bg="#f9f9f9"    # Estilo limpio
        )
        self.txt_results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.txt_results.yview)

        # Definimos el estilo (tag) para la fuente ibérica dentro del texto
        self.txt_results.tag_configure("iberian_style", 
                                       font=(self.SYSTEM_FONT_NAME, self.SYSTEM_FONT_SIZE), 
                                       foreground="blue")
        
        self.txt_results.tag_configure("error_style", foreground="red")
        self.txt_results.tag_configure("info_style", foreground="gray")

        self.update_status("...")
        self.load_model()

    def load_model(self):
        try:
            if not MODEL_PATH.exists():
                messagebox.showerror("Error", f"Modelo no encontrado en:\n{MODEL_PATH}")
                return
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("Modelo cargado correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"Fallo al cargar modelo: {e}")

    def update_status(self, text, style="info_style"):
        self.txt_results.config(state=tk.NORMAL) # Habilitar escritura
        self.txt_results.delete("1.0", tk.END)   # Borrar todo
        self.txt_results.insert(tk.END, text, style) # Insertar
        self.txt_results.tag_add("center", "1.0", "end") 
        self.txt_results.config(state=tk.DISABLED) # Bloquear escritura

    def open_image(self):
        f_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not f_path: return
        
        self.pil_image = Image.open(f_path).convert("RGB")
        
        # Mostrar en GUI
        display = self.pil_image.copy()
        display.thumbnail((IMG_PREVIEW_SIZE, IMG_PREVIEW_SIZE))
        self.tk_image = ImageTk.PhotoImage(display)
        self.lbl_image.config(image=self.tk_image)
        
        self.btn_predict.config(state=tk.NORMAL)
        self.update_status("...")

    def on_checkbox_toggle(self):
        if self.pil_image:
            self.predict_image()

    def predict_image(self):
        if not self.pil_image or not self.model: return

        # Limpiar área de texto temporalmente
        self.txt_results.config(state=tk.NORMAL)
        self.txt_results.delete("1.0", tk.END)
        self.txt_results.config(state=tk.DISABLED)
        
        try:
            #Segmentación y previsualización
            draw_boxes = self.show_bboxes.get()
            symbols, display_pil = self.segment_and_preprocess(self.pil_image, draw_boxes=draw_boxes)
            
            display_img = display_pil.copy()
            display_img.thumbnail((IMG_PREVIEW_SIZE, IMG_PREVIEW_SIZE))
            self.tk_image = ImageTk.PhotoImage(display_img)
            self.lbl_image.config(image=self.tk_image)
            
            if not symbols:
                self.update_status("No se detectaron símbolos.", "error_style")
                return

            # Preparar Batch
            batch = np.stack(symbols, axis=0)
            if getattr(self.model, "input_shape", [])[-1] == 1 and batch.ndim == 3:
                batch = batch[..., np.newaxis]

            #Predecir
            probs = self.model.predict(batch, verbose=0)
            preds = np.argmax(probs, axis=1)
            
            # Mostrar resultados
            self.show_native_results(preds)

        except Exception as e:
            print(f"Error critico: {e}")
            self.update_status(f"Error: {e}", "error_style")
            self.open_image() # Restaurar visualización

    def show_native_results(self, preds):
        self.txt_results.config(state=tk.NORMAL) # Habilitar escritura
        self.txt_results.delete("1.0", tk.END)   # Limpiar
        
        for idx in preds:
            class_name = self.class_names[idx]
            char = self.char_map.get(class_name)
            
            if char:
                # Insertamos el carácter ibérico con su tag de estilo
                self.txt_results.insert(tk.END, f"{char} ", "iberian_style")
            else:
                #  para caracteres no mapeados
                self.txt_results.insert(tk.END, f"[{class_name}] ", "info_style")
        
        self.txt_results.config(state=tk.DISABLED) # Bloquear para que el usuario no borre

    def segment_and_preprocess(self, pil_img, draw_boxes=False):
        #  Preparar imagen para OpenCV
        img_gray = pil_img.convert('L')
        img_cv = np.array(img_gray)
        
        #  Binarizar
        arr = np.where(img_cv < THRESH, 0, 255).astype(np.uint8)
        img_inv = 255 - arr 

        #  Encontrar contornos
        contours, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return [], pil_img
        
        # Filtrar y obtener Bounding Boxes
        valid_boxes = [cv2.boundingRect(c) for c in contours]
        valid_boxes = [b for b in valid_boxes if b[2] > 5 and b[3] > 5]
        if not valid_boxes: return [], pil_img

        
        # Calcular la altura promedio de las cajas
        # Usaremos esta altura como umbral para saber si un símbolo está en la misma línea
        avg_height = np.mean([h for x, y, w, h in valid_boxes])
        Y_TOLERANCE = avg_height * 0.5  

        #  Agrupar cajas por filas (basado en la coordenada Y)
        rows = []
        
        # Ordenamos inicialmente por la coordenada Y (de arriba a abajo)
        valid_boxes.sort(key=lambda b: b[1]) 

        for box in valid_boxes:
            x, y, w, h = box
            
            # Centro vertical de la caja
            center_y = y + h / 2
            
            found_row = False
            for row in rows:
                # Comprobar si el centro vertical de la nueva caja está dentro de la tolerancia de la fila
                row_center_y = np.mean([b[1] + b[3] / 2 for b in row])
                
                if abs(center_y - row_center_y) < Y_TOLERANCE:
                    # Si está cerca, pertenece a esta fila
                    row.append(box)
                    found_row = True
                    break
            
            if not found_row:
                # Si no pertenece a ninguna fila, crear una nueva fila con esta caja
                rows.append([box])

        # Ordenar filas: primero por la Y de la fila, y DENTRO de cada fila, por la X
        final_boxes = []
        for row in rows:
            # Ordenar la fila individualmente por la coordenada X (izquierda a derecha)
            row.sort(key=lambda b: b[0])
            final_boxes.extend(row)
            

        # PREPARAR IMAGEN PARA DIBUJAR
        annotated_img = pil_img.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated_img)

        processed_symbols = []
        
        for (x, y, w, h) in final_boxes: 
            # PROCESAMIENTO PARA EL MODELO 
            roi_inv = img_inv[y:y+h, x:x+w] 
            roi_pil = Image.fromarray(255 - roi_inv, mode="L") 
            roi_contained = ImageOps.contain(roi_pil, (INNER_IMG_SIZE, INNER_IMG_SIZE), Image.BICUBIC)
            canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
            d_x = (IMG_SIZE - roi_contained.width) // 2
            d_y = (IMG_SIZE - roi_contained.height) // 2
            canvas.paste(roi_contained, (d_x, d_y))
            res = np.array(canvas).astype("float32") / 255.0
            processed_symbols.append(res)

            # DIBUJAR CAJA VERDE 
            if draw_boxes: 
                draw.rectangle([x, y, x + w, y + h], outline="green", width=3)

        final_display_img = annotated_img if draw_boxes else pil_img
        return processed_symbols, final_display_img

if __name__ == "__main__":
    root = tk.Tk()
    app = OCR_GUI(root)
    root.mainloop()