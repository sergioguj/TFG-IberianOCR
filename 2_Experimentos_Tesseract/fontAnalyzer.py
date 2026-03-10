from fontTools.ttLib import TTFont
import os

# Ruta al archivo de la fuente
path = os.path.dirname(os.path.abspath(__file__))
font_path = path + r'\iberian.ttf'  

# Cargar la fuente
font = TTFont(font_path)

# Obtener propiedades básicas
fontname = font["name"].getDebugName(1)
style = font["name"].getDebugName(2)

# Examinar si es de ancho fijo
os2_table = font["OS/2"]
fixed = "1" if os2_table.panose.bProportion == 9 else "0"

# Examinar si tiene estilo en cursiva (italic)
italic = "1" if hasattr(font["post"], "italicAngle") and font["post"].italicAngle != 0 else "0"

# Examinar si tiene estilo en negrita (bold)
bold = "1" if "OS/2" in font and os2_table.panose.bWeight == 9 else "0"

# Examinar si es Serif
serif = "1" if os2_table.panose.bSerifStyle > 1 else "0"

# Examinar si es un estilo fraktur
fraktur = "1" if os2_table.panose.bFamilyType == 2 else "0"

# Construir la salida final en el formato solicitado
result = f"[ {fontname} ] < {italic} > < {bold} > < {fixed} > < {serif} > < {fraktur} >"

# Imprimir la salida
print(result)
