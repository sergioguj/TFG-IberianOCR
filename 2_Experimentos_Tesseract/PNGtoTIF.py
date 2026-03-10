from PIL import Image
import os

# Ruta de entrada y salida
input_folder = r'C:\Users\sergi\Desktop\ModeloMejoradoTIF'  # Carpeta con las imágenes en PNG
output_folder = r'C:\Users\sergi\Desktop\ModeloMejoradoTIF'  # Carpeta para guardar imágenes en TIFF

os.makedirs(output_folder, exist_ok=True)

# Funcion para convertir PNG a TIFF
def convert_png_to_tiff(input_path, output_path):
    with Image.open(input_path) as img:
        img = img.convert('RGB')  # Convierte a un formato compatible si es necesario
        img.save(output_path, format='TIFF')

# Procesar todas las imagenes PNG
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith('.png'):  # Procesar solo archivos PNG
        input_path = os.path.join(input_folder, file_name)
        tiff_file = os.path.splitext(file_name)[0] + '.tiff'  # Cambiar la extensión
        output_path = os.path.join(output_folder, tiff_file)

        # Convertir y guardar como TIFF
        convert_png_to_tiff(input_path, output_path)
        print(f'Convertido: {file_name} -> {tiff_file}')
