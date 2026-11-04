import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox

import cv2
import mediapipe as mp
from deepface import DeepFace
from PIL import Image, ImageTk
from ultralytics import YOLO


class SistemaVisionUFG:
    def __init__(self, root):
        self.root = root
        self.root.title("UFG - Facultad de Ingeniería y Sistemas | Examen #3")
        self.root.geometry("920x720")
        self.root.configure(bg="#d9d9d9")
        self.root.resizable(False, False)

        # Estado del sistema y cámara
        self.cap = cv2.VideoCapture(0)
        self.modo_actual = (
            "opencv_img"  # Modos: opencv_img, opencv_vid, yolo, deepface, mediapipe
        )
        self.frame_actual = None
        self.frame_procesado = None

        # Variables para grabación de video (OpenCV 2)
        self.grabando_video = False
        self.video_writer = None

        # Variables de control para IA
        self.frame_count = 0
        self.emocion_actual = "Analizando..."
        self.edad_actual = "N/A"
        self.genero_actual = "N/A"
        self.ultimo_json_yolo = []
        self.estado_mano = "Sin detectar"

        # 1. Inicializar YOLO (Modelo rápido yolov8n.pt visto en clase)
        try:
            self.modelo_yolo = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"Error cargando YOLO: {e}")
            self.modelo_yolo = None

        # MediaPipe 1.x para Python 3.13 ya no incluye la API legacy
        # mp.solutions usada por este proyecto.
        self.mp_disponible = hasattr(mp, "solutions")
        if self.mp_disponible:
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_hands = mp.solutions.hands

            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.hands = self.mp_hands.Hands(
                min_detection_confidence=0.7, min_tracking_confidence=0.7
            )
        else:
            self.face_mesh = None
            self.hands = None
            print(
                "MediaPipe compatible no disponible: esta versión requiere "
                "Python 3.12 y mediapipe<=0.10.21 para usar mp.solutions."
            )

        # Construir la interfaz gráfica idéntica a la hoja
        self.crear_interfaz()

        # Iniciar el bucle de video
        self.actualizar_frame()

    def crear_interfaz(self):
        # Contenedor principal con borde
        main_frame = tk.Frame(self.root, bg="#e6e6e6", bd=2, relief="solid")
        main_frame.place(relx=0.03, rely=0.03, relwidth=0.94, relheight=0.94)

        # ================= SECCIÓN 1: VISOR DE VIDEO SUPERIOR =================
        visor_container = tk.Frame(main_frame, bg="#e6e6e6", bd=1, relief="solid")
        visor_container.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=0.65)

        # Simulación del recuadro con bordes limpios
        self.lbl_video = tk.Label(
            visor_container,
            bg="#ffffff",
            text="Iniciando cámara...",
            relief="groove",
            bd=2,
        )
        self.lbl_video.place(relx=0.05, rely=0.06, relwidth=0.90, relheight=0.88)

        # ================= SECCIÓN 2: PANEL INTERMEDIO DE MODELOS =================
        mid_container = tk.Frame(main_frame, bg="#dcdcdc", bd=1, relief="solid")
        mid_container.place(relx=0.0, rely=0.65, relwidth=1.0, relheight=0.20)

        # Configuración de estilo de botones intermedios
        btn_style = {
            "font": ("Arial", 10, "bold"),
            "bg": "#f0f0f0",
            "fg": "#333333",
            "relief": "ridge",
            "bd": 2,
            "cursor": "hand2",
        }

        # --- Bloque 1: OpenCV (Imagen y Video) ---
        b1_frame = tk.Frame(mid_container, bg="#dcdcdc", bd=1, relief="solid")
        b1_frame.place(relx=0.0, rely=0.0, relwidth=0.34, relheight=1.0)

        # OpenCV 1 (Modo Imagen)
        self.btn_cv_img = tk.Button(
            b1_frame,
            text="OpenCv",
            command=lambda: self.cambiar_modo("opencv_img"),
            **btn_style,
        )
        self.btn_cv_img.place(relx=0.06, rely=0.12, relwidth=0.40, relheight=0.34)

        self.btn_save_cv_img = tk.Button(
            b1_frame, text="Guardar", command=self.guardar_opencv_imagen, **btn_style
        )
        self.btn_save_cv_img.place(relx=0.06, rely=0.54, relwidth=0.40, relheight=0.34)

        # OpenCV 2 (Modo Video)
        self.btn_cv_vid = tk.Button(
            b1_frame,
            text="OpenCv",
            command=lambda: self.cambiar_modo("opencv_vid"),
            **btn_style,
        )
        self.btn_cv_vid.place(relx=0.54, rely=0.12, relwidth=0.40, relheight=0.34)

        self.btn_save_cv_vid = tk.Button(
            b1_frame, text="Guardar", command=self.toggle_guardar_video, **btn_style
        )
        self.btn_save_cv_vid.place(relx=0.54, rely=0.54, relwidth=0.40, relheight=0.34)

        # --- Bloque 2: Yolo ---
        b2_frame = tk.Frame(mid_container, bg="#dcdcdc", bd=1, relief="solid")
        b2_frame.place(relx=0.34, rely=0.0, relwidth=0.22, relheight=1.0)

        self.btn_yolo = tk.Button(
            b2_frame,
            text="Yolo",
            command=lambda: self.cambiar_modo("yolo"),
            **btn_style,
        )
        self.btn_yolo.place(relx=0.15, rely=0.12, relwidth=0.70, relheight=0.34)

        self.btn_save_yolo = tk.Button(
            b2_frame, text="Guardar", command=self.guardar_yolo, **btn_style
        )
        self.btn_save_yolo.place(relx=0.15, rely=0.54, relwidth=0.70, relheight=0.34)

        # --- Bloque 3: deepFace ---
        b3_frame = tk.Frame(mid_container, bg="#dcdcdc", bd=1, relief="solid")
        b3_frame.place(relx=0.56, rely=0.0, relwidth=0.22, relheight=1.0)

        self.btn_deepface = tk.Button(
            b3_frame,
            text="deepFace",
            command=lambda: self.cambiar_modo("deepface"),
            **btn_style,
        )
        self.btn_deepface.place(relx=0.15, rely=0.12, relwidth=0.70, relheight=0.34)

        self.btn_save_deepface = tk.Button(
            b3_frame, text="Guardar", command=self.guardar_deepface, **btn_style
        )
        self.btn_save_deepface.place(
            relx=0.15, rely=0.54, relwidth=0.70, relheight=0.34
        )

        # --- Bloque 4: Mediapipe ---
        b4_frame = tk.Frame(mid_container, bg="#dcdcdc", bd=1, relief="solid")
        b4_frame.place(relx=0.78, rely=0.0, relwidth=0.22, relheight=1.0)

        self.btn_mediapipe = tk.Button(
            b4_frame,
            text="Mediapipe",
            command=lambda: self.cambiar_modo("mediapipe"),
            **btn_style,
        )
        self.btn_mediapipe.place(relx=0.15, rely=0.12, relwidth=0.70, relheight=0.34)

        self.btn_save_mediapipe = tk.Button(
            b4_frame, text="Guardar", command=self.guardar_mediapipe, **btn_style
        )
        self.btn_save_mediapipe.place(
            relx=0.15, rely=0.54, relwidth=0.70, relheight=0.34
        )

        # CAMBIO 1: estos son los botones que se bloquean al elegir un modo;
        # los botones Guardar permanecen disponibles.
        self.botones_modo = [
            self.btn_cv_img,
            self.btn_cv_vid,
            self.btn_yolo,
            self.btn_deepface,
            self.btn_mediapipe,
        ]

        # ================= SECCIÓN 3: PANEL INFERIOR =================
        bot_container = tk.Frame(main_frame, bg="#cccccc", bd=1, relief="solid")
        bot_container.place(relx=0.0, rely=0.85, relwidth=1.0, relheight=0.15)

        bot_btn_style = {
            "font": ("Arial", 12, "bold"),
            "bg": "#eeeeee",
            "fg": "#222222",
            "relief": "ridge",
            "bd": 3,
            "cursor": "hand2",
        }

        self.btn_analisis = tk.Button(
            bot_container,
            text="Analisis",
            command=self.mostrar_analisis,
            **bot_btn_style,
        )
        self.btn_analisis.place(relx=0.02, rely=0.20, relwidth=0.176, relheight=0.60)

        self.btn_guardar_gral = tk.Button(
            bot_container,
            text="Guardar",
            command=self.guardar_reporte_general,
            **bot_btn_style,
        )
        self.btn_guardar_gral.place(
            relx=0.216, rely=0.20, relwidth=0.176, relheight=0.60
        )

        self.btn_equipo = tk.Button(
            bot_container, text="Equipo", command=self.mostrar_equipo, **bot_btn_style
        )
        self.btn_equipo.place(relx=0.412, rely=0.20, relwidth=0.176, relheight=0.60)

        # CAMBIO 2: acceso a mi foto  y botón para volver a habilitar los modos.
        # CEVA = CARLOS ENMANUEL VENTURA ALVAREZ
        self.btn_ceva = tk.Button(
            bot_container, text="Mi Foto", command=self.mostrar_ceva, **bot_btn_style
        )
        self.btn_ceva.place(relx=0.608, rely=0.20, relwidth=0.176, relheight=0.60)

        self.btn_cambiar_modo = tk.Button(
            bot_container,
            text="Reactivar Botones",
            command=self.reactivar_modos,
            state=tk.DISABLED,
            **bot_btn_style,
        )
        self.btn_cambiar_modo.place(
            relx=0.804, rely=0.20, relwidth=0.176, relheight=0.60
        )

        # Resaltar el botón inicial activo
        self.actualizar_colores_botones()

    # ================= LÓGICA DE CAMBIO DE MODO =================
    def cambiar_modo(self, nuevo_modo):
        if nuevo_modo == "mediapipe" and not self.mp_disponible:
            messagebox.showwarning(
                "MediaPipe no disponible",
                "El programa está usando Python 3.13 y MediaPipe 1.x.\n\n"
                "El modo MediaPipe de este proyecto usa mp.solutions, "
                "que requiere Python 3.12 con mediapipe<=0.10.21.",
            )
            return
        self.modo_actual = nuevo_modo
        self.actualizar_colores_botones()
        # CAMBIO 1: tras elegir un modo, se bloquean solo los selectores;
        # "Cambiar modo" permite habilitarlos de nuevo.
        for boton in self.botones_modo:
            boton.configure(state=tk.DISABLED)
        self.btn_cambiar_modo.configure(state=tk.NORMAL)

    def reactivar_modos(self):
        """Vuelve a habilitar los botones para seleccionar un modo."""
        for boton in self.botones_modo:
            boton.configure(state=tk.NORMAL)
        self.btn_cambiar_modo.configure(state=tk.DISABLED)

    def actualizar_colores_botones(self):
        botones = {
            "opencv_img": self.btn_cv_img,
            "opencv_vid": self.btn_cv_vid,
            "yolo": self.btn_yolo,
            "deepface": self.btn_deepface,
            "mediapipe": self.btn_mediapipe,
        }
        for modo, btn in botones.items():
            if modo == self.modo_actual:
                btn.configure(bg="#a8d1ff")  # Azul claro indicando modo activo
            else:
                btn.configure(bg="#f0f0f0")

    # ================= PROCESAMIENTO DE VIDEO EN TIEMPO REAL =================
    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_actual = frame.copy()
            output_frame = frame.copy()
            self.frame_count += 1

            # --- 1. MODO OPENCV IMAGEN ---
            if self.modo_actual == "opencv_img":
                cv2.putText(
                    output_frame,
                    "Modo: OpenCV (Captura de Imagen)",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

            # --- 2. MODO OPENCV VIDEO ---
            elif self.modo_actual == "opencv_vid":
                estado_txt = (
                    "GRABANDO VIDEO..."
                    if self.grabando_video
                    else "Modo: OpenCV (Listo para grabar)"
                )
                color_txt = (0, 0, 255) if self.grabando_video else (255, 200, 0)
                cv2.putText(
                    output_frame,
                    estado_txt,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color_txt,
                    2,
                )

            # --- 3. MODO YOLO (Detección + Conteo + Extracción como en Ejemplos 3, 4 y 5) ---
            elif self.modo_actual == "yolo" and self.modelo_yolo is not None:
                resultados = self.modelo_yolo(output_frame, verbose=False)
                output_frame = resultados[0].plot()

                # Extraer datos para JSON y conteo de clases
                datos_temp = []
                for r in resultados:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_name = r.names[int(box.cls[0])]
                        datos_temp.append(
                            {
                                "objeto": cls_name,
                                "confianza": round(conf, 2),
                                "bbox": {
                                    "x1": round(x1),
                                    "y1": round(y1),
                                    "x2": round(x2),
                                    "y2": round(y2),
                                },
                            }
                        )
                self.ultimo_json_yolo = datos_temp

            # --- 4. MODO DEEPFACE (Emociones cada 30 frames como en deepface7.py) ---
            elif self.modo_actual == "deepface":
                if self.frame_count % 30 == 0:
                    try:
                        analisis = DeepFace.analyze(
                            output_frame,
                            actions=["emotion", "age", "gender"],
                            enforce_detection=False,
                            silent=True,
                        )
                        if analisis:
                            self.emocion_actual = analisis[0]["dominant_emotion"]
                            self.edad_actual = analisis[0]["age"]
                            self.genero_actual = analisis[0]["dominant_gender"]
                    except Exception:
                        self.emocion_actual = "Sin rostro"

                cv2.putText(
                    output_frame,
                    f"Emocion: {self.emocion_actual}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    output_frame,
                    f"Edad: {self.edad_actual} | Genero: {self.genero_actual}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            # --- 5. MODO MEDIAPIPE (Malla Facial + Conteo/Posición de Mano como en media1_1 y media2.py) ---
            elif self.modo_actual == "mediapipe":
                rgb_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False

                res_mesh = self.face_mesh.process(rgb_frame)
                res_hands = self.hands.process(rgb_frame)

                rgb_frame.flags.writeable = True
                output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                # Dibujar Malla Facial (468 puntos)
                if res_mesh.multi_face_landmarks:
                    for face_landmarks in res_mesh.multi_face_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image=output_frame,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                        )

                # Dibujar Manos y lógica de negocio (Mano Levantada / Bajada)
                if res_hands.multi_hand_landmarks:
                    for hand_landmarks in res_hands.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            output_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )
                        wrist_y = hand_landmarks.landmark[
                            self.mp_hands.HandLandmark.WRIST
                        ].y
                        middle_tip_y = hand_landmarks.landmark[
                            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                        ].y

                        if middle_tip_y < wrist_y:
                            self.estado_mano = "MANO LEVANTADA"
                            color_m = (0, 255, 0)
                        else:
                            self.estado_mano = "MANO BAJADA"
                            color_m = (0, 0, 255)

                        cv2.putText(
                            output_frame,
                            self.estado_mano,
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            color_m,
                            3,
                        )

            # Si está grabando video en OpenCV, escribir fotograma en el archivo
            if self.grabando_video and self.video_writer is not None:
                self.video_writer.write(output_frame)

            self.frame_procesado = output_frame.copy()

            # Ajustar tamaño al Label de Tkinter y convertir con PIL (Ejemplo 4 de clase)
            ancho_lbl = self.lbl_video.winfo_width()
            alto_lbl = self.lbl_video.winfo_height()
            if ancho_lbl > 10 and alto_lbl > 10:
                frame_resized = cv2.resize(output_frame, (ancho_lbl, alto_lbl))
            else:
                frame_resized = output_frame

            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lbl_video.imgtk = imgtk
            self.lbl_video.configure(image=imgtk, text="")

        self.lbl_video.after(15, self.actualizar_frame)

    # ================= FUNCIONES DE GUARDADO INDIVIDUALES =================
    def guardar_opencv_imagen(self):
        """Primer botón Guardar de OpenCV: Guarda una fotografía instantánea."""
        if self.frame_procesado is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"opencv_imagen_{timestamp}.jpg"
            cv2.imwrite(filename, self.frame_procesado)
            messagebox.showinfo(
                "OpenCV Imagen", f"Imagen guardada exitosamente como:\n{filename}"
            )

    def toggle_guardar_video(self):
        """Segundo botón Guardar de OpenCV: Inicia o detiene la grabación de video."""
        if not self.grabando_video:
            if self.frame_actual is None:
                return
            alto, ancho, _ = self.frame_actual.shape
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.nombre_video = f"opencv_video_{timestamp}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.video_writer = cv2.VideoWriter(
                self.nombre_video, fourcc, 20.0, (ancho, alto)
            )
            self.grabando_video = True
            self.btn_save_cv_vid.configure(text="Detener", bg="#ff7b7b")
            self.cambiar_modo("opencv_vid")
        else:
            self.grabando_video = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.btn_save_cv_vid.configure(text="Guardar", bg="#f0f0f0")
            messagebox.showinfo(
                "OpenCV Video",
                f"Video guardado exitosamente como:\n{self.nombre_video}",
            )

    def guardar_yolo(self):
        """Guarda la imagen anotada por YOLO y el archivo deteccion_log.json (Ejemplo 5)."""
        if self.frame_procesado is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = f"yolo_captura_{timestamp}.jpg"
            json_name = f"deteccion_log_{timestamp}.json"

            cv2.imwrite(img_name, self.frame_procesado)
            with open(json_name, "w", encoding="utf-8") as f:
                json.dump(self.ultimo_json_yolo, f, indent=4)

            messagebox.showinfo(
                "YOLO Guardado",
                f"Resultados guardados:\n- Imagen: {img_name}\n- JSON: {json_name}",
            )

    def guardar_deepface(self):
        """Guarda la foto analizada por DeepFace y un registro de atributos."""
        if self.frame_procesado is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = f"deepface_{timestamp}.jpg"
            json_name = f"deepface_atributos_{timestamp}.json"

            cv2.imwrite(img_name, self.frame_procesado)
            datos_df = {
                "fecha": timestamp,
                "emocion_dominante": self.emocion_actual,
                "edad_estimada": self.edad_actual,
                "genero_dominante": self.genero_actual,
            }
            with open(json_name, "w", encoding="utf-8") as f:
                json.dump(datos_df, f, indent=4)

            messagebox.showinfo(
                "DeepFace Guardado",
                f"Análisis facial guardado:\n- {img_name}\n- {json_name}",
            )

    def guardar_mediapipe(self):
        """Guarda la captura con los landmarks de MediaPipe dibujados."""
        if self.frame_procesado is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = f"mediapipe_{timestamp}.jpg"
            cv2.imwrite(img_name, self.frame_procesado)
            messagebox.showinfo(
                "MediaPipe Guardado", f"Captura de malla/postura guardada:\n{img_name}"
            )

    # ================= BOTONES DEL PANEL INFERIOR =================
    def mostrar_analisis(self):
        """Muestra un resumen en tiempo real de todos los módulos."""
        cant_objetos = len(self.ultimo_json_yolo)
        resumen = (
            f"=== RESUMEN DE ANÁLISIS EN TIEMPO REAL ===\n\n"
            f"• Modo Activo Actual: {self.modo_actual.upper()}\n"
            f"• Estado de Grabación: {'Grabando' if self.grabando_video else 'Inactivo'}\n\n"
            f"--- Últimos datos de IA ---\n"
            f"• YOLO (Objetos detectados): {cant_objetos}\n"
            f"• DeepFace (Emoción): {self.emocion_actual} | Edad: {self.edad_actual}\n"
            f"• MediaPipe (Estado Mano): {self.estado_mano}"
        )
        messagebox.showinfo("Análisis del Sistema", resumen)

    def guardar_reporte_general(self):
        """Botón Guardar general inferior: Exporta imagen actual y reporte completo JSON."""
        if self.frame_procesado is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_file = f"reporte_general_{timestamp}.jpg"
            json_file = f"reporte_general_{timestamp}.json"

            cv2.imwrite(img_file, self.frame_procesado)
            reporte = {
                "timestamp": timestamp,
                "modo_activo": self.modo_actual,
                "yolo_detecciones": self.ultimo_json_yolo,
                "deepface": {
                    "emocion": self.emocion_actual,
                    "edad": self.edad_actual,
                    "genero": self.genero_actual,
                },
                "mediapipe": {"estado_mano": self.estado_mano},
            }
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(reporte, f, indent=4)

            messagebox.showinfo(
                "Reporte General",
                f"Se guardó el reporte completo:\n- {img_file}\n- {json_file}",
            )

    # CAMBIO 2: la ventana, el texto y la imagen se comparten para Equipo y CEVA.
    def mostrar_ventana_con_foto(self, titulo, texto, nombre_imagen):
        ventana = tk.Toplevel(self.root)
        ventana.title(titulo)
        ventana.configure(bg="#e6e6e6")
        ventana.resizable(False, False)

        tk.Label(
            ventana, text=texto, bg="#e6e6e6", justify="center", padx=20, pady=15
        ).pack()

        ruta_imagen = Path(__file__).resolve().parent / nombre_imagen
        if ruta_imagen.exists():
            try:
                imagen = Image.open(ruta_imagen)
                imagen.thumbnail((700, 450), Image.Resampling.LANCZOS)
                foto = ImageTk.PhotoImage(imagen)

                etiqueta = tk.Label(ventana, image=foto, bg="#e6e6e6")
                etiqueta.image = foto
                etiqueta.pack(padx=20, pady=(0, 20))
            except (OSError, ValueError) as error:
                tk.Label(
                    ventana,
                    text=f"No se pudo cargar la imagen:\n{error}",
                    fg="#b00020",
                    bg="#e6e6e6",
                ).pack(padx=20, pady=(0, 20))
        else:
            tk.Label(
                ventana,
                text=f"No se encontró la imagen:\n{ruta_imagen.name}",
                fg="#b00020",
                bg="#e6e6e6",
            ).pack(padx=20, pady=(0, 20))

    def mostrar_equipo(self):
        """Muestra los integrantes del equipo y la fotografía del proyecto."""
        info_equipo = (
            "UNIVERSIDAD FRANCISCO GAVIDIA (UFG)\n"
            "Facultad de Ingeniería y Sistemas\n"
            "Introducción a la Inteligencia Artificial - Examen #3\n\n"
            "Integrantes del Equipo:\n"
            "• Raúl Antonio Elias Alegria EA100222\n"
            "• Emmanuel Antonio Escobar Luna EL100122\n"
            "• Luis Nelson Hernandez Blanco HB100122\n"
            "• Cinthia Yasmin Rivera Pineda RP101022\n"
        )
        self.mostrar_ventana_con_foto(
            "Información del Equipo", info_equipo, "opencv_imagen_20261029_194810.jpg"
        )

    def mostrar_ceva(self):
        """Muestra la fotografía de CEVA con su nombre."""
        self.mostrar_ventana_con_foto(
            "Carlos Enmanuel Ventura Alvarez",
            "Carlos Enmanuel Ventura Alvarez",
            "CEVA.png",
        )

    def cerrar_aplicacion(self):
        if self.grabando_video and self.video_writer is not None:
            self.video_writer.release()
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SistemaVisionUFG(root)
    root.protocol("WM_DELETE_WINDOW", app.cerrar_aplicacion)
    root.mainloop()
