## Introducción ## 

El notebook de Inference recibe un video y realiza todos los pasos para devolver una predicción.

Utiliza un modelo de detección `megadetector 5.0.28` finetuneado utlizando el conjunto de datos de fauna uruguaya creado por Ambá y Tryolabs, publicado en [Kaggle](https://www.kaggle.com/competitions/cupybara/data). El clasificador es `speciesnet 5.0.0.`

Las etapas son:

1. Recorrer el video completo para obtener frames, guardarlos en la carpeta *cropped_images*, realizar detecciones y clasificaciones. El parámetro *steps* determina la cantidad de pasos a realizar, es decir, la cantidad de frames a iterar. Estos pasos se realizan de manera uniforme a lo largo del video, por lo cual mientras mayor sea steps más muestras se toman, con menor distancia entre cada una. El parámetro *find_n_frames* determina la condición de parada de la búsqueda de muestras.

2. Mapear las predicciones de `speciesnet 5.0.0.` a un diccionario provisto por el usuario con nombres de animales en formato no científico (sin taxónomia, orden, etc) contenido junto con los pesos del modelo.

3. Realizar la predicción final de la especie hallada utilizando heurísticas propuestas por el autor del PoC.

## Estructura del proyecto ## 

```
├── ./
├── dataset_utils/ # Para crear el dataset sobre el cual finetunear el MegaDetector
├── models/
│     ├── species_net_weights.pt
|     ├── megadetector_weights.pt
|     ├── species_dict.json
|     └── more species net metadata
├── videos/
│     ├── video_name.AVI/ # Videos a procesar
|     └── ... 
├── cropped_images/
│     ├── video_name_without_file_extension/ # Videos procesados
|         ├── extracted_image_frame_num # Imagenes de los frames que contienen objetos
|         ├── ...
|         └─ yolo_metadata.json
|     └── ... 

```

## Pesos y datos ## 

Se deben solicitar al autor del repo.
