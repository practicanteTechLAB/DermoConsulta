"""
Run a rest API exposing the yolov5s object detection mode
"""
import argparse
import io                    # Librerias para manejar carpetas del sistema operativo
from PIL import Image        # Transformacion de imagenes
import shutil                # Eliminacion de carpetas de sistema operativo
from shutil import rmtree
import os                    # Manejo del sistema operativo
import boto3                 # Para el servicio de predicción de edad
import requests              # Para controlar sistema operativo

import torch                                  
from flask import Flask, jsonify, render_template # Lib para crear el servidor web
from flask_ngrok import run_with_ngrok # Lib para crear la URL publica 
from flask import url_for
from flask import request    # Manejo de métodos de captura de APIs

from reportlab.pdfgen import canvas # Librerias para la generación del PDF
from reportlab.lib.utils import Image
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import mm
from reportlab.lib.utils import simpleSplit

application = Flask(__name__)
run_with_ngrok(application) # Linea para indicar que se arrancara el servidor con Ngrok

@application.route("/send-image2/<path:url>", methods=['POST']) # Se asigna la direccion y se indica que admite el metodo POST
def predictUrl(url):
        # Captura los Datos recibidos y obtiene el dato que tiene la llave "image"
        # Lee el archivo
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        img.save("./images/foto_descargada.jpg")

        # Capturar la respuestas a las preguntas capturadas en el BOT
        dataJSON = request.json
        global p0, p1, p2, p3, p4
        p0 = dataJSON["pregunta_1"]
        p1 = dataJSON["pregunta_2"]
        p2 = dataJSON["pregunta_3"]
        p3 = dataJSON["pregunta_4"]
        p4 = dataJSON["pregunta_5"]

        results = model(img, size=640) # Pasa la imagen al modelo con un tamaño de imagen de 640px de ancho
        results.save() # Guarda la imagen con la deteccion en la carpeta run/detect/exp

        contenido = os.listdir('./runs/detect/exp') # Almacena el nombre de la imagen en contenido, posicion 0
        shutil.copy("./runs/detect/exp/"+contenido[0], "./static/foto_detectada.jpg") # Copia la imagen a la carpeta static con el nombre "foto_detectada.jpg"
        rmtree("./runs/detect/exp") # Se elimina la carpeta runs con sus respectivas subcarpetas

        data = results.pandas().xyxy[0] # Se almacenan los parametros de detección

        # Generar la url del PDF generado
        urlsended = url_for('static', filename='Pdf_consulta.pdf')
        age()
        global imperfeccionValue

        if(len(data) == 0):
            responder = {'nameURL': urlsended, 'Tipo Imperfección': "Rostro bien cuidado (No tienes imperfecciones)",
            'Predicción edad': 'Edad promedio ' + str(prom)}
            imperfeccionValue = "No se detectan imperfecciones"

        if(len(data) == 1):
            responder = {'nameURL': urlsended, 'Tipo Imperfección': str(data.values[0][6]),
            'Predicción edad': 'Edad promedio ' + str(prom)}
            imperfeccionValue = str(data.values[0][6])
            if(data.values[0][6] == "Acne"):
                imperfeccionValue = "Acné"

        if(len(data) == 2):
            responder = {'nameURL': urlsended, 'Tipo Imperfección': str(data.values[0][6])
            + ", " + str(data.values[1][6]),
            'Predicción edad': 'Edad promedio ' + str(prom)}
            if(data.values[0][6] == data.values[1][6]):
                imperfeccionValue = str(data.values[0][6])
            else:
                imperfeccionValue = str(data.values[0][6]) + ", " + str(data.values[1][6])
            
            if(data.values[0][6] == "Acne"):
                imperfeccionValue = "Acné"
            if(data.values[1][6] == "Acne"):
                imperfeccionValue = "Acné"

        comparacionesActivos()
        principiosActivos()

        genPDFLocal()
        return jsonify(responder) # Envia el PDF generado con la imagen y valor de la detección en campo DetectionVal

# Método para cargar la imagen que se va a analizar
def age():
        photo= './images/foto_descargada.jpg'
        face_count=detect_faces(photo)
# Método para consumir servicio en AWS en el cual se realiza el promedio de la edad
def detect_faces(photo):
    # Conexión al servicio de AWS key
    client = boto3.client('rekognition',
                        aws_access_key_id="AKIA47P5BXX47DPG5JG5",
                        aws_secret_access_key="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",
                        region_name="us-east-1")
    # Usa el método de detección de rostro
    with open(photo, 'rb') as image:
        response = client.detect_faces(Image={'Bytes': image.read()}, Attributes=['ALL'])
    # Usa FaceDetails para estimar el rango de edad alto y bajo de la imagen y sacar un promedio        
    for faceDetail in response['FaceDetails']:
        age_hight = faceDetail['AgeRange']['High']
        age_low = faceDetail['AgeRange']['Low']
        global prom
        prom = (age_hight + age_low)/2
        prom = int(prom)

    return prom
# Método para la comparación de las variables capturadas por el bot para asignar el valor de tipo de piel
def comparacionesActivos():
    # global p0, p1, p2, p3, p4
    global varTipoPiel

    if((p0 == "Tirante") and (p1 == "Tirante") and (p2 == "Tirante") and (p3 == "Si") and (p4 == "Si")):
        varTipoPiel = "Piel Seca-Sensible"

    if((p0 == "Tirante") and (p1 == "Tirante") and (p2 == "Tirante") and (p3 == "Si") and ((p4 == "No") or (p4 == "N/A"))):
        varTipoPiel = "Piel Seca-Sensible"

    if((p0 == "Tirante") and (p1 == "Tirante") and (p2 == "Tirante") and (p3 == "No") and (p4 == "Si")):
        varTipoPiel = "Piel Seca"

    if((p0 == "Tirante") and (p1 == "Tirante") and (p2 == "Tirante") and (p3 == "No") and (p4 == "No")):
        varTipoPiel = "Piel Seca"

    if((p0 == "Tirante") and (p1 == "Tirante") and (p2 == "Oleosa") and (p3 == "Si") and (p4 == "Si")):
        varTipoPiel = "Piel Mixta-Sensible"

    if((p0 == "Tirante") and (p1 == "Tirante") and (p2 == "Oleosa") and (p3 == "Si") and (p4 == "No")):
        varTipoPiel = "Piel Mixta-Sensible"

    if((p0 == "Tirante") and (p1 == "Tirante") and (p2 == "Oleosa") and (p3 == "No") and (p4 == "Si")):
        varTipoPiel = "Piel Mixta"

    if((p0 == "Tirante") and (p1 == "Tirante") and (p2 == "Oleosa") and (p3 == "No") and (p4 == "No")):
        varTipoPiel = "Piel Mixta"

    # if((p0 == "Tirante") and (p1 == "Oleosa") and (p2 == "Tirante") and (p3 == "Si") and (p4 == "Si")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Tirante") and (p1 == "Oleosa") and (p2 == "Tirante") and (p3 == "Si") and (p4 == "No")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Tirante") and (p1 == "Oleosa") and (p2 == "Tirante") and (p3 == "No") and (p4 == "Si")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Tirante") and (p1 == "Oleosa") and (p2 == "Tirante") and (p3 == "No") and (p4 == "No")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    if((p0 == "Tirante") and (p1 == "Oleosa") and (p2 == "Oleosa") and (p3 == "Si") and ((p4 == "Si") or (p4 == "N/A"))):
        varTipoPiel = "Piel Grasa-Sensible"

    if((p0 == "Tirante") and (p1 == "Oleosa") and (p2 == "Oleosa") and (p3 == "Si") and (p4 == "No")):
        varTipoPiel = "Piel Grasa-Sensible"

    if((p0 == "Tirante") and (p1 == "Oleosa") and (p2 == "Oleosa") and (p3 == "No") and ((p4 == "Si") or (p4 == "N/A"))):
        varTipoPiel = "Piel Grasa"

    if((p0 == "Tirante") and (p1 == "Oleosa") and (p2 == "Oleosa") and (p3 == "No") and (p4 == "No")):
        varTipoPiel = "Piel Grasa"

    # if((p0 == "Oleosa") and (p1 == "Tirante") and (p2 == "Tirante") and (p3 == "Si") and (p4 == "Si")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Oleosa") and (p1 == "Tirante") and (p2 == "Tirante") and (p3 == "Si") and (p4 == "No")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Oleosa") and (p1 == "Tirante") and (p2 == "Tirante") and (p3 == "No") and (p4 == "Si")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Oleosa") and (p1 == "Tirante") and (p2 == "Tirante") and (p3 == "No") and (p4 == "No")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    if((p0 == "Oleosa") and (p1 == "Tirante") and (p2 == "Oleosa") and (p3 == "Si") and ((p4 == "Si") or (p4 == "N/A"))):
        varTipoPiel = "Piel Mixta-Sensible"

    if((p0 == "Oleosa") and (p1 == "Tirante") and (p2 == "Oleosa") and (p3 == "Si") and (p4 == "No")):
        varTipoPiel = "Piel Mixta-Sensible"

    if((p0 == "Oleosa") and (p1 == "Tirante") and (p2 == "Oleosa") and (p3 == "No") and ((p4 == "Si") or (p4 == "N/A"))):
        varTipoPiel = "Piel Mixta"

    if((p0 == "Oleosa") and (p1 == "Tirante") and (p2 == "Oleosa") and (p3 == "No") and (p4 == "No")):
        varTipoPiel = "Piel Mixta"

    # if((p0 == "Oleosa") and (p1 == "Oleosa") and (p2 == "Tirante") and (p3 == "Si") and (p4 == "Si")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Oleosa") and (p1 == "Oleosa") and (p2 == "Tirante") and (p3 == "Si") and (p4 == "No")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Oleosa") and (p1 == "Oleosa") and (p2 == "Tirante") and (p3 == "No") and (p4 == "Si")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Oleosa") and (p1 == "Oleosa") and (p2 == "Tirante") and (p3 == "No") and (p4 == "No")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    if((p0 == "Oleosa") and (p1 == "Oleosa") and (p2 == "Oleosa") and (p3 == "Si") and ((p4 == "Si") or (p4 == "N/A"))):
        varTipoPiel = "Piel Grasa-Sensible"

    if((p0 == "Oleosa") and (p1 == "Oleosa") and (p2 == "Oleosa") and (p3 == "Si") and (p4 == "No")):
        varTipoPiel = "Piel Grasa-Sensible"

    if((p0 == "Oleosa") and (p1 == "Oleosa") and (p2 == "Oleosa") and (p3 == "No") and ((p4 == "Si") or (p4 == "N/A"))):
        varTipoPiel = "Piel Grasa"

    if((p0 == "Oleosa") and (p1 == "Oleosa") and (p2 == "Oleosa") and (p3 == "No") and (p4 == "No")):
        varTipoPiel = "Piel Grasa"
# Método para los activos dependiendo del valor que tenga asignada la variable tipo de piel
def principiosActivos():
    global v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,rec
    tipoPiel = varTipoPiel

    if (tipoPiel == "Piel Grasa"):
        v1 = "Ácido glicólico"
        v2 = "Vitamina C"
        v3 = "Ácido salicílico"
        v4 = "Niacinamida"
        v5 = "Ácido láctico"
        v6 = "Ácido hialurónico"
        v7 = "Agua termal"
        v8 = "Fotoprotección"
        v9 = ""
        v10 = ""
        rec = "Se recomienda utilizar una rutina de limpieza dos veces al día tipo espuma, gel o loción, hidratación, tratamiento anti acné y fotoprotección"

    if (tipoPiel == "Piel Grasa-Sensible"):
        v1 = "Retinol"
        v2 = "Vitamina C, B5, E"
        v3 = "Niacinamida"
        v4 = "Ácido hialurónico"
        v5 = "Agua termal"
        v6 = "Fotoprotección"
        v7 = "Cafeína"
        v8 = "Ceramidas"
        v9 = ""
        v10 = ""
        rec = "Se recomienda utilizar una rutina de limpieza para disminuir la oleosidad, calma la piel con un tónico, aplicar una hidratante específica para pieles grasas y sensibles, indispensable fotoproteger la piel"

    if (tipoPiel == "Piel Seca"):
        v1 = "Retinol"
        v2 = "Vitamina C, B5, E"
        v3 = "Niacinamida"    
        v4 = "Ácido láctico"
        v5 = "Ácido hialurónico"
        v6 = "Agua termal"
        v7 = "Fotoprotección"
        v8 = "AHA's"
        v9 = ""
        v10 = ""
        rec = "Se recomienda mantener una rutina diaria de limpieza y mantener una hidratación durante la mañana y en la noche para proteger la piel de agresiones. No olvides fotoproteger."

    if (tipoPiel == "Piel Seca-Sensible"):
        v1 = "Vitamina C, B5, E"
        v2 = "Niacinamida"    
        v3 = "Ácido láctico"
        v4 = "Ácido hialurónico"
        v5 = "Agua termal"
        v6 = "Fotoprotección"
        v7 = "Cafeína"
        v8 = "Ceramidas"
        v9 = ""
        v10 = ""
        rec = "Se recomienda mantener una rutina diaria de limpieza utilizando productos que protejan la piel de agresiones externas, evita frotar la piel por que son movimientos que puede irritar la piel. Es fundamental mantener una hidratación en la mañana y en la noche para mejorar el aspecto de la piel y aumentar el umbral de tolerancia de la piel. No olvides fotoproteger."

    if (tipoPiel == "Piel Mixta"):
        v1 = "Ácido glicólico"
        v2 = "Retinol"
        v3 = "Vitamina C, B5"
        v4 = "Ácido salicílico"
        v5 = "Niacinamida"
        v6 = "Ácido láctico"
        v7 = "Ácido hialurónico"
        v8 = "Agua termal"
        v9 = "Fotoprotección"
        v10 = "AHA's"
        rec = "Se recomienda utilizar una rutina de limpieza dos veces al día tipo espuma, gel o loción, hidratación y fotoprotección."

    if (tipoPiel == "Piel Mixta-Sensible"):
        v1 = "Retinol"
        v2 = "Vitamina C, B5, E"
        v3 = "Niacinamida"
        v4 = "Ácido láctico"
        v5 = "Ácido hialurónico"
        v6 = "Agua termal"
        v7 = "Fotoprotección"
        v8 = "Cafeína"
        v9 = "Ceramidas"
        v10 = ""
        rec = "Se recomienda utilizar productos de limpieza adecuados para tu tipo de piel en especial agua micelar o tónicos, adicionalmente productos que ayuden a minimizar los poros en la zona T y aplicar hidratante tipo textura serum, gel o emulsión. No olvides fotoproteger la piel."
# Método para generar el PDF final de diagnóstico usando ReportLAB Python
def genPDFLocal():
    custom_size = (330*mm,339*mm)
    i = mm
    d = i/4
    w, h = custom_size
    c = canvas.Canvas("./Static/Pdf_consulta.pdf",pagesize=custom_size)

    c.setFont("Helvetica", 15)
    #Cambiar el Color del Fondo
    c.setFillColorRGB(1,1,1)
    c.rect(0, 0, w, h, fill=1, stroke=0)

    # Dimensiones Cambiaron definido como "custom_size = (294*mm,298*mm)" y en milimetros
    fotoia = ImageReader('./Static/foto_detectada.jpg')
    # c.drawImage(fotoia, 17 * mm, 166.45 * mm , width= 177 * mm ,  height= 119 * mm, preserveAspectRatio=False)
    # c.drawImage(fotoia, 32 * mm, 165.45 * mm , width= 145 * mm ,  height= 119 * mm, preserveAspectRatio=False)
    c.drawImage(fotoia, 32 * mm, 110 * mm , width= 140 * mm ,  height= 200 * mm, preserveAspectRatio=False)

    bg = Image.open("./imagesPDF/fondo_v1.png")
    bg.save("./imagesPDF/fondo_tranparente.png")
    bg = ImageReader("./imagesPDF/fondo_tranparente.png")
    width, height = bg.getSize()
    c.drawImage(bg, x= 0, y=0, width=bg._width, height=bg._height, mask='auto')

    logok = Image.open("./imagesPDF/logoEficacia.png")
    logok.save("./imagesPDF/logoEficacia_transparente.png")
    logok = ImageReader("./imagesPDF/logoEficacia_transparente.png")
    width, height = logok.getSize()
    c.drawImage(logok, x = 245.18*mm, y=12.34*mm, width=logok._width, height=logok._height, mask='auto')

    #_________________________________________________________________________
    c.setFont("Helvetica", 28)
    c.setFillColorRGB(0.25,0.32,0.12,1)
    # c.drawString(190.93 * mm , 302.14 * mm, imperfeccionValue)

    def text_wrap(text, width):
        lines = []
        for line in simpleSplit(text, "Times-Roman", 8, width):
            lines.append(line)
        return lines
    
    text = imperfeccionValue
    width = 100
    height = 20
    x = 207.93 * mm
    y = 302.14 * mm

    for line in text_wrap(text, width):
        c.drawCentredString(x, y, line)
        y -= 15
    #_________________________________________________________________________
    c.setFont("Helvetica", 14)
    c.setFillColorRGB(0.25,0.32,0.12,1)

    text = varTipoPiel
    width = 160
    height = 10
    x = 250 * mm 
    y = 277.2 * mm

    for line in text_wrap(text, width):
        c.drawCentredString(x, y, line)
        y -= 15
    #_________________________________________________________________________

        #¿Cómo siente su piel al levantarse?
    c.setFillColorRGB(0.25,0.32,0.12,1)
    c.drawString(242.61 * mm , 257.34 * mm, p0)
        
        #¿Cómo es la textura de su piel en frente y mentón?
    c.setFillColorRGB(0.25,0.32,0.12,1)
    c.drawString(242.61 * mm , 236.53 * mm, p1)

        #¿Cómo es la textura de su piel en mejillas?
    c.setFillColorRGB(0.25,0.32,0.12,1)
    c.drawString(242.61 * mm , 216.77 * mm, p2)

        #¿Siente alguna sensibilidad?
    c.setFillColorRGB(0.25,0.32,0.12,1)
    c.drawString(247.73 * mm , 177.36 * mm, p3)

        #¿Le dura el maquillaje?
    c.setFillColorRGB(0.25,0.32,0.12,1)
    c.drawString(247.73 * mm , 196.31 * mm, p4)
    #___________________________________________________________
        #La edad de su piel es    
    c.setFillColorRGB(0.25,0.32,0.12,1)
    # c.drawString(241 * mm , 157 * mm, str(prom) + " años")

    text = str(prom) + " años"
    width = 100
    height = 10
    x = 250 * mm
    y = 157 * mm

    for line in text_wrap(text, width):
        c.drawCentredString(x, y, line)
        y -= 15
    #___________________________________________________________

    #Parrafo 1 Izquierda
    c.setFont("Helvetica-Bold", 15)
    c.setFillColorRGB(0.25,0.32,0.12,1)
    
    text = rec
    width = 310
    height = 400
    x = 470
    y = height - 50

    for line in text_wrap(text, width):
        c.drawCentredString(x, y, line)
        y -= 15
        
    # Parrafo 2 Derecha
    c.setFont("Helvetica-Bold", 15)

    if (v10 == ""):
        text = v1 + ", " + v2 + ", " + v3 + ", " + v4 + ", " + v5 + ", " + v6 + ", " + v7 + ", " + v8 + ", " + v9
    if (v9 == ""):
        text = v1 + ", " + v2 + ", " + v3 + ", " + v4 + ", " + v5 + ", " + v6 + ", " + v7 + ", " + v8
    else:
        text = v1 + ", " + v2 + ", " + v3 + ", " + v4 + ", " + v5 + ", " + v6 + ", " + v7 + ", " + v8 + ", " + v9 + ", " + v10

    width = 310
    height = 240
    x = 470
    y = height - 80

    for line in text_wrap(text, width):
        c.drawCentredString(x, y, line)
        y -= 15

    c.showPage()
    c.save()

@application.route('/none') # Ruta para prueba de funcionamiento,  Solo muestra el memsaje de hola en el navegador
def none():
    return render_template('index.html') # se debe llamar con GET

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s') # Carga el detector con el modelo COCO
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True) # Carga el detector con el modelo Sonrisas
    model.conf = 0.7 # Indica el nivel de confianza minimo en la detección
    model.eval()

    #application.run(host="0.0.0.0", port=4000, debug=True)  # Inicia en servidor Local
    application.run() # inicia en Servidor Remoto,  Tener en cuenta que en cada inicio de servidor esta direccion cambia
    # debido a que se esta usando una libreria gratuita de tunelamiento.
