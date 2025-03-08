# ESTE FUNCIONA! pero no tiene paths relativos (Esta versión comparti con Kevin)
# Para ejecutar entrar a: http://127.0.0.1:8000/docs
# Estamos probando con : 11220, Tuesday, 16.0

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import os

# Inicializar FastAPI
app = FastAPI()

# Cargar modelos y datos de forma segura
try:
    modelo_sentimientos_final = joblib.load(r'C:\Users\felip\Desktop\Stuff\Cursos\SoyHenry\Clases\LABS\PF\PF_google_yelp\Proyecto-FInal\Proyecto-FInal\Modelos ML\modelo_sentimientos_final.pkl')
    vectorizer = joblib.load(r'C:\Users\felip\Desktop\Stuff\Cursos\SoyHenry\Clases\LABS\PF\PF_google_yelp\Proyecto-FInal\Proyecto-FInal\Modelos ML\vectorizador_tfidf.pkl')
    modelo_knn = joblib.load(r'C:\Users\felip\Desktop\Stuff\Cursos\SoyHenry\Clases\LABS\PF\PF_google_yelp\Proyecto-FInal\Proyecto-FInal\Modelos ML\modelo_knn.pkl')
    df = pd.read_csv(r'C:\Users\felip\Desktop\Stuff\Cursos\SoyHenry\Clases\LABS\PF\PF_google_yelp\Proyecto-FInal\Proyecto-FInal\Modelos ML\data_recomendacion.csv')
except Exception as e:
    raise RuntimeError(f"Error al cargar modelos o datos: {str(e)}")

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
        print(f"Total de registros en df: {len(df)}")
        print(f"Filtrando por zip_code={zip_code}")
        df_filtrado = df[df['zip_code'] == zip_code]
        print(f"Registros tras filtrar por zip_code={zip_code}: {len(df_filtrado)}")

        if df_filtrado.empty:
            raise HTTPException(status_code=404, detail="No se encontraron restaurantes para ese código postal.")

        # Convertir las columnas de horarios a formato numérico
        if f'{dia}_open' not in df.columns or f'{dia}_close' not in df.columns:
            raise HTTPException(status_code=400, detail=f"Las columnas de horario para {dia} no existen en los datos.")

        df_filtrado.loc[:, f'{dia}_open'] = pd.to_numeric(df_filtrado[f'{dia}_open'], errors='coerce')
        df_filtrado.loc[:, f'{dia}_close'] = pd.to_numeric(df_filtrado[f'{dia}_close'], errors='coerce')

        print(f"Registros antes de filtrar por horario: {len(df_filtrado)}")
        df_filtrado = df_filtrado[(df_filtrado[f'{dia}_open'] <= hora) & (df_filtrado[f'{dia}_close'] >= hora)]
        print(f"Registros después de filtrar por horario: {len(df_filtrado)}")

        if df_filtrado.empty:
            raise HTTPException(status_code=404, detail="No se encontraron restaurantes abiertos en ese horario.")

        # Seleccionar los 10 restaurantes con más reseñas
        top_10_reviews = df_filtrado.nlargest(10, 'num_of_reviews')
        print(f"Top 10 por num_of_reviews: {len(top_10_reviews)} registros encontrados")
        
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

# V3 #######################################################################################################################

# No tiene paths relativos
# Para ejecutar entrar a: http://127.0.0.1:8000/docs
# Estamos probando con : 11220, Tuesday, 16.0

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import os

# Inicializar FastAPI
app = FastAPI()

# Cargar modelos y datos de forma segura
try:
    modelo_sentimientos_final = joblib.load(r'C:\Users\felip\Desktop\Stuff\Cursos\SoyHenry\Clases\LABS\PF\PF_google_yelp\Proyecto-FInal\Proyecto-FInal\Modelos ML\modelo_sentimientos_final.pkl')
    vectorizer = joblib.load(r'C:\Users\felip\Desktop\Stuff\Cursos\SoyHenry\Clases\LABS\PF\PF_google_yelp\Proyecto-FInal\Proyecto-FInal\Modelos ML\vectorizador_tfidf.pkl')
    modelo_knn = joblib.load(r'C:\Users\felip\Desktop\Stuff\Cursos\SoyHenry\Clases\LABS\PF\PF_google_yelp\Proyecto-FInal\Proyecto-FInal\Modelos ML\modelo_knn.pkl')
    df = pd.read_csv(r'C:\Users\felip\Desktop\Stuff\Cursos\SoyHenry\Clases\LABS\PF\PF_google_yelp\Proyecto-FInal\Proyecto-FInal\Modelos ML\data_recomendacion.csv')
except Exception as e:
    raise RuntimeError(f"Error al cargar modelos o datos: {str(e)}")

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
        print(f"Total de registros en df: {len(df)}")
        print(f"Filtrando por zip_code={zip_code}")
        df_filtrado = df[df['zip_code'] == zip_code]
        print(f"Registros tras filtrar por zip_code={zip_code}: {len(df_filtrado)}")

        if df_filtrado.empty:
            raise HTTPException(status_code=404, detail="No se encontraron restaurantes para ese código postal.")

        # Convertir las columnas de horarios a formato numérico
        if f'{dia}_open' not in df.columns or f'{dia}_close' not in df.columns:
            raise HTTPException(status_code=400, detail=f"Las columnas de horario para {dia} no existen en los datos.")

        df_filtrado[f'{dia}_open'] = pd.to_numeric(df_filtrado[f'{dia}_open'], errors='coerce')
        df_filtrado[f'{dia}_close'] = pd.to_numeric(df_filtrado[f'{dia}_close'], errors='coerce')
        print(f"Registros antes de filtrar por horario: {len(df_filtrado)}")
        df_filtrado = df_filtrado[(df_filtrado[f'{dia}_open'] <= hora) & (df_filtrado[f'{dia}_close'] >= hora)]
        print(f"Registros después de filtrar por horario: {len(df_filtrado)}")

        if df_filtrado.empty:
            raise HTTPException(status_code=404, detail="No se encontraron restaurantes abiertos en ese horario.")

        # Seleccionar los 10 restaurantes con más reseñas
        top_10_reviews = df_filtrado.nlargest(10, 'num_of_reviews')
        print(f"Top 10 por num_of_reviews: {len(top_10_reviews)} registros encontrados")
        
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
