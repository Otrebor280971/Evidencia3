import gradio as gr
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    df = pd.read_pickle("db_textos.pkl")
    embeddings = np.load("db_embeddings.npy")
    print(f"Datos cargados: {len(df)} registros.")
except FileNotFoundError:
    print("ERROR CRÍTICO: No se encontraron 'db_textos.pkl' o 'db_embeddings.npy'.")
    print("Ejecuta 'python preparar_datos.py' antes de lanzar la app.")
    sys.exit(1)

df['etiqueta'] = df['titulo'] + " - " + df['artista']

opciones_canciones = sorted(df[df['tipo'] == 'Canción']['etiqueta'].unique().tolist())
opciones_poemas = sorted(df[df['tipo'] == 'Poema']['etiqueta'].unique().tolist())

print(f"Canciones disponibles: {len(opciones_canciones)}")
print(f"Poemas disponibles: {len(opciones_poemas)}")


def title_filter(category):
    """Actualiza las opciones del dropdown según la categoría"""
    if category == "Canciones":
        nuevas_opciones = opciones_canciones
    else:
        nuevas_opciones = opciones_poemas
    
    return gr.Dropdown(choices=nuevas_opciones, value=None, interactive=True)


def limpiar_preview():
    """Limpia el preview cuando cambia la categoría"""
    return ""


def mostrar_texto(etiqueta):
    """Muestra el texto seleccionado"""
    if not etiqueta:
        return "" 
    
    filtro = df[df['etiqueta'] == etiqueta]
    if filtro.empty:
        return "Error: No se encontró el texto."
        
    row = filtro.iloc[0]
    
    preview = (
        f"Título: {row['titulo']}\n"
        f"Autor: {row['artista']}\n"
        f"Tipo: {row['tipo']}\n"
        f"{'-'*30}\n"
        f"{row['texto']}"
    )
    return preview


def similares(etiqueta, top_n):
    """Encuentra obras similares"""
    if not etiqueta:
        return pd.DataFrame(columns=["Info"], data=[["Selecciona una obra primero"]])
    
    try:
        lista_indices = df.index[df['etiqueta'] == etiqueta].tolist()
        
        if not lista_indices:
            return pd.DataFrame(columns=["Error"], data=[["No se encontró la obra seleccionada"]])
 
        idx_origen = lista_indices[0]
        
        vector = embeddings[idx_origen].reshape(1, -1)
        similitudes = cosine_similarity(vector, embeddings)[0]
        
        idx_ordenado = np.argsort(similitudes)[::-1]
        
        resultados = []
        count = 0
        
        for i in idx_ordenado:
            if i == idx_origen:
                continue
            
            if count >= top_n:
                break
            
            row = df.iloc[i]
            score = similitudes[i]
            
            resultados.append([
                row['titulo'],
                row['artista'],
                row['tipo'],
                f"{score:.1%}",
                row['texto'][:100].replace('\n', ' ') + "..."
            ])
            count += 1
        
        if not resultados:
            return pd.DataFrame(columns=["Info"], data=[["No se encontraron resultados"]])
            
        return pd.DataFrame(resultados, columns=["Título", "Artista", "Tipo", "Similitud", "Fragmento"])
        
    except Exception as e:
        print(f"Error en cálculo: {e}")
        return pd.DataFrame(columns=["Error"], data=[[f"Error interno: {str(e)}"]])



with gr.Blocks(title="Buscador Semántico") as demo:
    gr.Markdown("Buscador de Similitudes: Música y Poesía")
    
    with gr.Row():
        with gr.Column(scale=1):
            radio = gr.Radio(
                ["Canciones", "Poemas"], 
                label="1. Categoría", 
                value="Canciones"
            )
            
            drop_titulos = gr.Dropdown(
                choices=opciones_canciones, 
                label="2. Selecciona la obra",
                interactive=True,
                filterable=True,
                value=None
            )
            
            slider_n = gr.Slider(
                minimum=1, 
                maximum=10, 
                value=3, 
                step=1, 
                label="3. Número de Recomendaciones"
            )
            
            btn_buscar = gr.Button("Encontrar Similares", variant="primary")

        with gr.Column(scale=2):
            txt_preview = gr.Textbox(
                label="Texto Original", 
                lines=10, 
                interactive=False,
                placeholder="Selecciona una obra para ver su contenido..."
            )
            
            tabla_resultados = gr.Dataframe(
                label="Resultados", 
                interactive=False, 
                wrap=True
            )
    

    radio.change(
        fn=title_filter, 
        inputs=radio, 
        outputs=drop_titulos
    ).then(
        fn=limpiar_preview,
        outputs=txt_preview
    )
    
    drop_titulos.change(
        fn=mostrar_texto, 
        inputs=drop_titulos, 
        outputs=txt_preview
    )
    
    btn_buscar.click(
        fn=similares, 
        inputs=[drop_titulos, slider_n], 
        outputs=tabla_resultados
    )


if __name__ == "__main__":
    demo.launch(debug=True, server_port=7865, inbrowser=True)