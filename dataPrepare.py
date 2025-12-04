import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import glob

def procesar_datos():
    print("--- CARGANDO DATOS ---")
    
    ruta_canciones = glob.glob("data/*.csv")
    lista_dfs_canciones = []

    for archivo in ruta_canciones:
        if "poems.csv" in archivo:
            continue
            
        try:
            df_temp = pd.read_csv(archivo, on_bad_lines='skip')
            
            if 'Lyric' in df_temp.columns and 'Title' in df_temp.columns:

                df_temp = df_temp.rename(columns={
                    'Title': 'titulo',
                    'Lyric': 'texto',
                    'Artist': 'artista'
                })
                
                df_temp = df_temp[['titulo', 'texto', 'artista']]
                df_temp['tipo'] = 'Canción'
                lista_dfs_canciones.append(df_temp)
                print(f"Cargado: {archivo} ({len(df_temp)} canciones)")
        except Exception as e:
            print(f"Error leyendo {archivo}: {e}")

    df_canciones = pd.concat(lista_dfs_canciones, ignore_index=True)

    try:
        df_poemas = pd.read_csv("data/poems.csv", on_bad_lines='skip')
        
        df_poemas = df_poemas.rename(columns={
            'author': 'artista',
            'content': 'texto',
            'title': 'titulo'
        })
        
        df_poemas = df_poemas[['titulo', 'texto', 'artista']]
        df_poemas['tipo'] = 'Poema'
        print(f"Cargado: poems.csv ({len(df_poemas)} poems)")
        
    except Exception as e:
        print(f"Error leyendo poems.csv: {e}")
        df_poemas = pd.DataFrame()

    df_completo = pd.concat([df_canciones, df_poemas], ignore_index=True)
    
    df_completo = df_completo.dropna(subset=['texto'])
    df_completo = df_completo[df_completo['texto'].str.strip() != ""]
    
    print(f"Total de documentos a procesar: {len(df_completo)}")

    print("\n---GENERANDO EMBEDDINGS (Esto puede tardar unos minutos) ---")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    textos = df_completo['texto'].tolist()
    

    embeddings = model.encode(textos, show_progress_bar=True)

    print("\n--- 3. GUARDANDO ARCHIVOS ---")
    df_completo.to_pickle("db_textos.pkl")
    np.save("db_embeddings.npy", embeddings)
    
    print("Done")

if __name__ == "__main__":
    procesar_datos()