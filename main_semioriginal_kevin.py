# Para ejecutar entrar a: http://127.0.0.1:8000/docs

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd

# Inicializar FastAPI
app = FastAPI()

# Cargar el modelo de análisis de sentimientos
modelo_sentimientos_final = joblib.load(r'C:\Users\felip\Desktop\Stuff\Cursos\SoyHenry\Clases\LABS\PF\PF_google_yelp\Proyecto-FInal\Proyecto-FInal\Modelos ML\modelo_sentimientos_final.pkl')
vectorizer = joblib.load(r'C:\Users\felip\Desktop\Stuff\Cursos\SoyHenry\Clases\LABS\PF\PF_google_yelp\Proyecto-FInal\Proyecto-FInal\Modelos ML\vectorizador_tfidf.pkl')

# Cargar modelo y datos para recomendaciones
modelo_knn = joblib.load(r'C:\Users\felip\Desktop\Stuff\Cursos\SoyHenry\Clases\LABS\PF\PF_google_yelp\Proyecto-FInal\Proyecto-FInal\Modelos ML\modelo_knn.pkl')
df = pd.read_csv(r'C:\Users\felip\Desktop\Stuff\Cursos\SoyHenry\Clases\LABS\PF\PF_google_yelp\Proyecto-FInal\Proyecto-FInal\Modelos ML\data_recomendacion.csv')

# Clase para el cuerpo de la petición de análisis de sentimientos
class Comentario(BaseModel):
    texto: str

# Clase para el cuerpo de la petición de recomendaciones
class Consulta(BaseModel):
    horario: str
    condiciones_binarias: List[int]
    zip_code: str

# Endpoint para análisis de sentimientos
@app.get("/clasificar_comentario")
def clasificar_comentario(texto: str):
    # Vectorizar el comentario
    texto_tfidf = vectorizer.transform([texto])
    # Predecir el sentimiento
    prediccion = modelo_sentimientos_final.predict(texto_tfidf)[0]
    return {"sentimiento": prediccion}

# Verificar las columnas del DataFrame
print(df.columns)

# Modelo de respuesta para recomendación de restaurantes
    # Define la estructura de la respuesta de la API
class Recomendacion(BaseModel):
    name: str
    street_address: str
    zip_code: str
    num_of_reviews: int
    avg_rating: float
    mensaje: str

# Endpoint para recomendar restaurantes
@app.get("/recomendar_restaurantes", response_model=List[Recomendacion])
def recomendar_restaurantes(
    zip_code: str = Query(..., description="Código postal de la ubicación del usuario"),
    dia: str = Query(..., description="Día de la semana (Monday, Tuesday, etc.)"),
    hora: float = Query(..., description="Hora en formato decimal (por ejemplo, 14.5 para 14:30)")
):
    # Filtrar los restaurantes por el código postal
    df_filtrado = df[df['zip_code'] == zip_code]
    
    # <Codigo Chatgpt: Para verificar si existen datos en el DataFrame con un determinado código postal

    print(f"Registros tras filtrar por zip_code={zip_code}: {len(df_filtrado)}")
    
    if df_filtrado.empty:
        return {"mensaje": "No se encontraron restaurantes para ese código postal."}

    # Codigo Chatgpt: Para verificar si existen datos en el DataFrame con un determinado código postal>


    # Convertir las columnas de horarios a formato numérico
    df_filtrado[f'{dia}_open'] = pd.to_numeric(df_filtrado[f'{dia}_open'], errors='coerce')
    df_filtrado[f'{dia}_close'] = pd.to_numeric(df_filtrado[f'{dia}_close'], errors='coerce')
    
    # Filtrar los restaurantes por la hora de apertura y cierre
    df_filtrado = df_filtrado[(df_filtrado[f'{dia}_open'] <= hora) & (df_filtrado[f'{dia}_close'] >= hora)]
    
    # Seleccionar los 10 restaurantes con mayor número de comentarios
    top_10_reviews = df_filtrado.nlargest(10, 'num_of_reviews')
    print(f"Top 10 por num_of_reviews: {len(top_10_reviews)} registros encontrados")
    
    # Seleccionar los 5 restaurantes con mayor puntuación promedio
    top_5_rating = top_10_reviews.nlargest(5, 'avg_rating')
    
    # Formatear la respuesta
    for index, row in top_5_rating.iterrows():
        print(f"El restaurante '{row['name']}', ubicado en '{row['street_address']}', posee {row['num_of_reviews']} comentarios, y el promedio de su puntuación es {row['avg_rating']}.")


    return top_5_rating.to_dict(orient="records")   # Esto convierte el DataFrame a una lista de diccionarios JSON

# Corrección chatgpt a función recomendar_restaurantes():
    # reemplazar:
        # return recomendar_restaurantes
    # por:
        # return top_5_rating.to_dict(orient="records")

