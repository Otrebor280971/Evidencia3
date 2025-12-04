import gradio as gr
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
import re
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    df = pd.read_pickle("db_textos.pkl")
    embeddings = np.load("db_embeddings.npy")
    print(f"Datos cargados: {len(df)} registros.")
except FileNotFoundError:
    print("ERROR: No se encontraron 'db_textos.pkl' o 'db_embeddings.npy'.")
    print("Ejecuta 'python preparar_datos.py' antes de lanzar la app.")
    sys.exit(1)

df['etiqueta'] = df['titulo'] + " - " + df['artista']

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

opciones_canciones = sorted(df[df['tipo'] == 'Canción']['etiqueta'].unique().tolist())
opciones_poemas = sorted(df[df['tipo'] == 'Poema']['etiqueta'].unique().tolist())
todas_opciones = sorted(df['etiqueta'].unique().tolist())

print(f"Canciones disponibles: {len(opciones_canciones)}")
print(f"Poemas disponibles: {len(opciones_poemas)}")

print("Matriz de similitud global")
matriz_similitud_global = cosine_similarity(embeddings)
print(f"Matriz global: {matriz_similitud_global.shape}")

def segmentar_en_frases(texto):
    lineas = [l.strip() for l in texto.split('\n') if len(l.strip()) > 10]
    
    if len(lineas) >= 5:
        return lineas[:50]
    
    texto = texto.replace('\n', ' ')
    frases = re.split(r'[.!?]+', texto)
    frases = [f.strip() for f in frases if len(f.strip()) > 20]
    
    if len(frases) >= 3:
        return frases[:50]
    
    palabras = texto.split()
    chunk_size = 15
    frases = []
    for i in range(0, len(palabras), chunk_size):
        chunk = ' '.join(palabras[i:i+chunk_size])
        if chunk:
            frases.append(chunk)
    
    return frases[:50]


def title_filter(category):
    """Actualiza las opciones del dropdown según la categoría"""
    if category == "Canciones":
        nuevas_opciones = opciones_canciones
    else:
        nuevas_opciones = opciones_poemas
    
    return gr.Dropdown(choices=nuevas_opciones, value=None, interactive=True)


def limpiar_preview():
    return ""


def mostrar_texto(etiqueta):
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

def generar_matriz_frases(etiqueta1, etiqueta2):
    if not etiqueta1 or not etiqueta2:
        return None, "Selecciona ambos textos", "", ""
    
    try:
        texto1 = df[df['etiqueta'] == etiqueta1].iloc[0]['texto']
        texto2 = df[df['etiqueta'] == etiqueta2].iloc[0]['texto']
        
        frases1 = segmentar_en_frases(texto1)
        frases2 = segmentar_en_frases(texto2)
        
        if len(frases1) == 0 or len(frases2) == 0:
            return None, "No se pudieron segmentar los textos en frases", "", ""
        
        emb1 = model.encode(frases1)
        emb2 = model.encode(frases2)
        
        matriz_sim = cosine_similarity(emb1, emb2)
        
        fig = go.Figure(data=go.Heatmap(
            z=matriz_sim,
            x=[f"F{j+1}" for j in range(len(frases2))],
            y=[f"F{i+1}" for i in range(len(frases1))],
            colorscale='RdYlGn',
            text=matriz_sim,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Similitud")
        ))
        
        fig.update_layout(
            title=f"Matriz de Similitud de Frase-Frase<br>{etiqueta1[:30]}... vs {etiqueta2[:30]}...",
            xaxis_title=f"{etiqueta2} (Frases)",
            yaxis_title=f"{etiqueta1} (Frases)",
            height=600,
            width=800
        )
        
        global frases_guardadas
        frases_guardadas = {
            'texto1': etiqueta1,
            'texto2': etiqueta2,
            'frases1': frases1,
            'frases2': frases2,
            'matriz': matriz_sim
        }
        
        info = f"Matriz generada: {len(frases1)} frases t1 * {len(frases2)} frases t2"
        
        return fig, info, "", ""
    
    except Exception as e:
        print(f"Error en matriz de frases: {e}")
        return None, f"Error: {str(e)}", "", ""

def consultar_frase(i, j):
    if 'frases_guardadas' not in globals():
        return "Genera una matriz", "", ""
    
    try:
        i = int(i) - 1
        j = int(j) - 1
        
        frases1 = frases_guardadas['frases1']
        frases2 = frases_guardadas['frases2']
        matriz = frases_guardadas['matriz']
        
        if i < 0 or i >= len(frases1) or j < 0 or j >= len(frases2):
            return "Indices fuera de rango", "", ""
        
        similitud = matriz[i, j]
        
        frase1_txt = f"Frase {i+1} de '{frases_guardadas['texto1']}:\n{frases1[i]}"
        frase2_txt = f"Frase {j+1} de '{frases_guardadas['texto2']}:\n{frases2[j]}"
        
        info = f"Similitud {similitud:.2%}"
        
        return frase1_txt, frase2_txt, info
    
    except Exception as e:
        return f"Error: {str(e)}", "", ""
    
def generar_matriz_global():
    try:
        n_textos = min(50, len(df))
        
        matriz_parcial = matriz_similitud_global[:n_textos, :n_textos]
        etiquetas = df['etiqueta'].iloc[:n_textos].tolist()
        
        etiquetas_cortas = [e[:25] + "..." if len(e) > 25 else e for e in etiquetas]
        
        fig = go.Figure(data=go.Heatmap(
            z=matriz_parcial,
            x=etiquetas_cortas,
            y=etiquetas_cortas,
            colorscale='Viridis',
            colorbar=dict(title="Similitud")
        ))
        
        fig.update_layout(
            title=f"Matriz de Similitud Global (Primero  {n_textos} textos)",
            height = 800,
            width = 900,
            xaxis={'side': 'bottom'},
            yaxis={'autorange': 'reversed'}
        )
        
        info = f"Matriz global generada: {n_textos} * {n_textos} textos"
        
        return fig, info
    
    except Exception as e:
        print(f"Error en matriz global: {e}")
        return None, f"Error: {str(e)}"
    

def comparar_desde_matriz_global(idx1, idx2):
    try:
        idx1 = int(idx1) - 1
        idx2 = int(idx2) - 1
        
        n_textos = min(50, len(df))
        
        if idx1 < 0 or idx1 >= n_textos or idx2 < 0 or idx2 >= n_textos:
            return "Índices fuera de rango", "", None, ""
        
        etiqueta1 = df.iloc[idx1]['etiqueta']
        etiqueta2 = df.iloc[idx2]['etiqueta']
        
        texto1 = mostrar_texto(etiqueta1)
        texto2 = mostrar_texto(etiqueta2)
        
        similitud = matriz_similitud_global[idx1, idx2]
        info = f"Similitud global: {similitud:.2%}"
        
        fig, _, _, _ = generar_matriz_frases(etiqueta1, etiqueta2)
        
        return texto1, texto2, fig, info
        
    except Exception as e:
        return f"Error: {str(e)}", "", None, ""
    
with gr.Blocks(title="Buscador Semántico") as demo:
    gr.Markdown("Buscador de Similitudes: Música y Poesía")
    
    with gr.Row():
        with gr.Column(scale=1):
            radio = gr.Radio(
                ["Canciones", "Poemas"], 
                label="1. Categoría", 
                value="Canciones"
            )
            
            drop_texto1 = gr.Dropdown(
                choices=opciones_canciones, 
                label="2. Selecciona el texto 1",
                interactive=True,
                filterable=True,
                value=None
            )
            
            slider_n = gr.Slider(
                minimum=1, 
                maximum=10, 
                value=3, 
                step=1, 
                label="3. Número de similaridades"
            )
            
            btn_buscar = gr.Button("Encontrar Similares", variant="primary")

        with gr.Column(scale=2):
            txt_preview1 = gr.Textbox(
                label="Texto 1", 
                lines=10, 
                interactive=False,
                placeholder="Selecciona una obra para ver su contenido..."
            )
            
            tabla_resultados = gr.Dataframe(
                label="Textos similares", 
                interactive=False, 
                wrap=True
            )
    

    radio.change(
        fn=title_filter, 
        inputs=radio, 
        outputs=drop_texto1
    ).then(
        fn=limpiar_preview,
        outputs=txt_preview1
    )
    
    drop_texto1.change(
        fn=mostrar_texto, 
        inputs=drop_texto1, 
        outputs=txt_preview1
    )
    
    btn_buscar.click(
        fn=similares, 
        inputs=[drop_texto1, slider_n], 
        outputs=tabla_resultados
    )

    with gr.Tab("Matriz frase a frase"):
        gr.Markdown("Compara dos textos textos frase por frase")
        
        with gr.Row():
            with gr.Column():
                drop_texto1_n2 = gr.Dropdown(
                    choices=todas_opciones,
                    label="Texto 1",
                    filterable=True
                )
                
                drop_texto2_n2 = gr.Dropdown(
                    choices=todas_opciones,
                    label="Texto 2",
                    filterable= True
                )
                
                btn_generar_matriz = gr.Button("Generar matriz Frase-Frase", variant="primary")
                
                info_matriz = gr.Textbox(label="Info", lines=2, interactive=False)
                
                gr.Markdown("Consultar celda (i, j)")
                
                with gr.Row():
                    num_i = gr.Number(label="Frase i (Texto 1)", value=1, precision=0)
                    num_j = gr.Number(label="Frase j (Texto 2)", value=1, precision=0)
                    
                btn_consultar = gr.Button("Ver Frases", variant="secondary")
                
                info_similitud = gr.Textbox(label="Similitud", lines=1)
                frase_i = gr.Textbox(label="Frase i", lines=3)
                frase_j = gr.Textbox(label="Frase j", lines=3)
                
            with gr.Column():
                plot_matriz = gr.Plot(label="Mapa de calor)")
        
        btn_generar_matriz.click(
            fn=generar_matriz_frases,
            inputs=[drop_texto1_n2, drop_texto2_n2],
            outputs=[plot_matriz, info_matriz, frase_i, frase_j]
        )
        
        btn_consultar.click(
            fn=consultar_frase,
            inputs=[num_i, num_j],
            outputs=[frase_i, frase_j, info_similitud]
        )

    with gr.Tab("Matriz global de colección"):
        gr.Markdown("Matriz de similitud entre todos los textos")
        
        with gr.Row():
            with gr.Column(scale=1):
                btn_generar_global = gr.Button("Generar Matriz Global", variant="primary")
                
                info_global = gr.Textbox(label="Info", lines=2)
                
                gr.Markdown("### Comparar dos textos desde la matriz")
                
                num_idx1 = gr.Number(label="Índice Texto 1", value=1, precision=0)
                num_idx2 = gr.Number(label="Índice Texto 2", value=2, precision=0)
                
                btn_comparar_global = gr.Button("Comparar Textos", variant="secondary")
                
                info_comp = gr.Textbox(label="Similitud", lines=1)
                texto_comp1 = gr.Textbox(label="Texto 1", lines=5)
                texto_comp2 = gr.Textbox(label="Texto 2", lines=5)
            
            with gr.Column(scale=2):
                plot_global = gr.Plot(label="🌐 Matriz Global")
                plot_frases_global = gr.Plot(label="🔥 Matriz Frase-Frase")
            
            btn_generar_global.click(
                fn=generar_matriz_global,
                outputs=[plot_global, info_global]
            )
            
            btn_comparar_global.click(
                fn=comparar_desde_matriz_global,
                inputs=[num_idx1, num_idx2],
                outputs=[texto_comp1, texto_comp2, plot_frases_global, info_comp]
            )

if __name__ == "__main__":
    demo.launch(
        share=True,
        debug=False,
        server_port=7860,
        inbrowser=True
    )