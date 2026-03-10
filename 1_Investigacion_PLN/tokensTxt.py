import os
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import pruebaTokenExcel

# Ruta de este script
path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, "tokensDocx")

# Crear la carpeta si no existe
os.makedirs(path, exist_ok=True)

# Obtener los tokens desde pruebaTokenExcel
tokens = list(set(pruebaTokenExcel.allTokens))  # Asegurarse de que los tokens sean únicos

# Dividir los tokens en grupos de 300
chunk_size = 300
chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

# Crear un archivo .docx por cada grupo de 300 tokens
for index, chunk in enumerate(chunks, start=1):
    doc = Document()
    
    # Agregar los tokens al documento
    p = doc.add_paragraph(' '.join(chunk))
    run = p.runs[0]        
    run.font.name = 'Iberian'  # Establecer la fuente Iberian
    run.font.size = Pt(12)  # Tamaño de fuente
    
    # Forzar a Word a reconocer la fuente personalizada
    rPr = run._element.rPr
    rFonts = OxmlElement('w:rFonts')
    rFonts.set(qn('w:ascii'), 'Iberian')
    rFonts.set(qn('w:hAnsi'), 'Iberian')
    rFonts.set(qn('w:cs'), 'Iberian')
    rPr.append(rFonts)
    
    # Guardar el documento
    docx_output_path = os.path.join(path, f'tokens{index}.docx')
    doc.save(docx_output_path)
    print(f"Tokens exportados a: {docx_output_path}")

print("\nArchivos .docx generados con 300 tokens únicos cada uno usando la fuente Iberian.")
