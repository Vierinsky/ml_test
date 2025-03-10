# Para ejecutar entrar a: http://127.0.0.1:8000/docs
# Estamos probando con : 11220, Tuesday, 16.0

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import os

# Inicializar FastAPI
app = FastAPI()

# Definir la ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Modelos_ML")

# Verificar si la carpeta de modelos existe
if not os.path.exists(MODELS_DIR):
    raise RuntimeError(f"❌ ERROR: La carpeta de modelos '{MODELS_DIR}' no existe.")

# Función para cargar modelos y archivos CSV
def load_model(file_name):
    path = os.path.join(MODELS_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ ERROR: Archivo no encontrado: {path}")
    return joblib.load(path)

def load_csv(file_name):
    path = os.path.join(MODELS_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ ERROR: Archivo no encontrado: {path}")
    return pd.read_csv(path)

# Cargar modelos y datos de forma segura
try:
    print("📂 Contenido de Modelos_ML:", os.listdir(MODELS_DIR))  # Para depuración en Render

    modelo_sentimientos_final = load_model("modelo_sentimientos_final.pkl")
    vectorizer = load_model("vectorizador_tfidf.pkl")
    modelo_knn = load_model("modelo_knn.pkl")
    df = load_csv("data_recomendacion.csv")

except Exception as e:
    raise RuntimeError(f"⚠️ Error al cargar modelos o datos: {str(e)}")

# Asegurar que zip_code sea string
df['zip_code'] = df['zip_code'].astype(str)

# Clase para el cuerpo de la petición de análisis de sentimientos
class Comentario(BaseModel):
    texto: str

# Clase para la respuesta de recomendaciones
class Recomendacion(BaseModel):
    name: str
    street_address: str
    zip_code: str
    num_of_reviews: int
    avg_rating: float
    mensaje: str

# Endpoint raíz
@app.get("/")
def read_root():

    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>API en Funcionamiento 🚀</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 50px;
                background-color: #f4f4f4;
            }
            h1 {
                color: #333;
            }
            .container {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                display: inline-block;
                text-align: left;
            }
            a {
                text-decoration: none;
                color: #007bff;
            }
            .btn {
                display: inline-block;
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin: 10px;
            }
            ul {
                list-style-type: none;
                padding: 0;
            }
            li {
                margin: 5px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 API en Funcionamiento</h1>
            <p>Bienvenido a la API de análisis de comentarios y recomendaciones de restaurantes.</p>
            <p><strong>Documentación interactiva:</strong></p>
            <a href="/docs" target="_blank" class="btn">Abrir Swagger UI</a>
            <p><strong>Endpoints disponibles:</strong></p>
            <ul>
                <li>📝 <a href="/clasificar_comentario?texto=Me%20encanta%20este%20lugar!" target="_blank">Análisis de sentimiento</a></li>
                <li>🍽️ <a href="/recomendar_restaurantes?zip_code=11220&dia=Tuesday&hora=16.0" target="_blank">Sistema de Recomendación de Restaurantes</a></li>
            </ul>
            <p>Visita la documentación para más información sobre cómo utilizar la API.</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Endpoint para análisis de sentimientos
@app.get("/clasificar_comentario")
def clasificar_comentario(texto: str):
    try:
        texto_tfidf = vectorizer.transform([texto])
        prediccion = modelo_sentimientos_final.predict(texto_tfidf)[0]
        return {"sentimiento": prediccion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

# Endpoint para recomendar restaurantes
@app.get("/recomendar_restaurantes", response_model=List[Recomendacion])
def recomendar_restaurantes(
    zip_code: str = Query(..., description="Código postal de la ubicación del usuario"),
    dia: str = Query(..., description="Día de la semana (Monday, Tuesday, etc.)"),
    hora: float = Query(..., description="Hora en formato decimal (por ejemplo, 14.5 para 14:30)")
):
    try:
        print(f"✅ Total de registros en df: {len(df)}")
        print(f"🔎 Filtrando por zip_code={zip_code}")
        df_filtrado = df[df['zip_code'] == zip_code]
        print(f"📊 Registros tras filtrar por zip_code={zip_code}: {len(df_filtrado)}")

        if df_filtrado.empty:
            raise HTTPException(status_code=404, detail="No se encontraron restaurantes para ese código postal.")

        # Verificar si las columnas de horario existen
        if f'{dia}_open' not in df.columns or f'{dia}_close' not in df.columns:
            raise HTTPException(status_code=400, detail=f"Las columnas de horario para {dia} no existen en los datos.")

        # Convertir las columnas de horarios a formato numérico
        df_filtrado.loc[:, f'{dia}_open'] = pd.to_numeric(df_filtrado[f'{dia}_open'], errors='coerce')
        df_filtrado.loc[:, f'{dia}_close'] = pd.to_numeric(df_filtrado[f'{dia}_close'], errors='coerce')
        
        print(f"🕒 Registros antes de filtrar por horario: {len(df_filtrado)}")
        df_filtrado = df_filtrado[(df_filtrado[f'{dia}_open'] <= hora) & (df_filtrado[f'{dia}_close'] >= hora)]
        print(f"✅ Registros después de filtrar por horario: {len(df_filtrado)}")

        if df_filtrado.empty:
            raise HTTPException(status_code=404, detail="No se encontraron restaurantes abiertos en ese horario.")

        # Seleccionar los 10 restaurantes con más reseñas
        top_10_reviews = df_filtrado.nlargest(10, 'num_of_reviews')
        print(f"🏆 Top 10 por num_of_reviews: {len(top_10_reviews)} registros encontrados")

        # Seleccionar los 5 con mejor rating
        top_5_rating = top_10_reviews.nlargest(5, 'avg_rating')

        resultado = [
            Recomendacion(
                name=row['name'],
                street_address=row['street_address'],
                zip_code=row['zip_code'],
                num_of_reviews=row['num_of_reviews'],
                avg_rating=row['avg_rating'],
                mensaje=f"El restaurante '{row['name']}', ubicado en '{row['street_address']}', posee {row['num_of_reviews']} comentarios, y el promedio de su puntuación es {row['avg_rating']}."
            )
            for _, row in top_5_rating.iterrows()
        ]

        return resultado
    
    except HTTPException as e:
        raise e  # Devolver errores personalizados sin cambios
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la recomendación: {str(e)}")
