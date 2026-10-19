




# """
# Deteccion de hitos en video de electrofisiologia (version en dos pasadas + visor multihilo).

#   Pasada 1 (extraer):   YOLO-World + tracking + analisis de contenido de pantallas.
#                         Implementacion Multithreading: La reproduccion corre a FPS nativos
#                         en el hilo principal mientras la IA procesa en segundo plano.
#   Pasada 2 (segmentar): suavizado + emisiones por fase + Viterbi izquierda-derecha
#                         -> segmentos de fase (CSV) y, opcionalmente, video con timeline.

# Uso:
#   python ep_fases.py --video ruta.mp4 --ver                # visor fluido en vivo + CSV + segmentacion
#   python ep_fases.py --video ruta.mp4 --ver --debug --tiles
#   python ep_fases.py --video ruta.mp4 --solo-segmentar --gt gt.csv --render

# Teclas del visor:  q salir | ESPACIO pausa | c mejora de luz off/on/auto | d debug on/off | t tiles on/off
# """
# import argparse
# import csv
# import os
# import queue
# import re
# import threading
# import time
# from collections import Counter, defaultdict, deque

# import cv2
# import numpy as np

# # ==============================================================================
# # CONFIGURACION
# # ==============================================================================
# FASES = [
#     "1. Preparacion medica (Colocacion de bata y guantes)",
#     "2. Preparacion de paciente y campo esteril",
#     "3. Puncion y acceso vascular femoral",
#     "4. Mapeo electroanatomico 3D (Navegador)",
#     "5. Intervencion / Ablacion por cateter",
#     "6. Retirada de material y cierre",
# ]

# VOCABULARIO = {
#     "monitor": ("pantalla", "Monitor"),
#     "computer screen": ("pantalla", "Monitor"),
#     "3D heart map on monitor": ("pantalla_carto", "Navegador 3D"),
#     "ECG waveform monitor": ("pantalla_poli", "Poligrafo"),
#     "x-ray monitor": ("fluoro", "Fluoroscopia"),
#     "patient on operating table": ("paciente", "Paciente en camilla"),
#     "operating table": ("paciente", "Paciente en camilla"),
#     "sterile surgical drapes": ("paciente", "Paciente en camilla"),
#     "surgical gown": ("bata", "Bata esteril"),
#     "surgical gloves": ("guantes", "Guantes"),
#     "gloved hand": ("guantes", "Guantes"),
#     "bare hand": ("mano_desnuda", "Mano desnuda"),
#     "cabinet": ("ignorar", ""),
#     "sink": ("ignorar", ""),
#     "cart": ("ignorar", ""),
# }
# PANTALLAS = {"pantalla", "pantalla_carto", "pantalla_poli", "fluoro"}

# CAMPOS_EV = ["navegador", "poligrafo", "fluoro", "paciente", "bata", "guantes",
#              "brazos_al_aire", "carto_rojo"]

# UMBRALES = dict(
#     conf_min=0.10,
#     conf_pantalla=0.18,
#     conf_pantalla_hint=0.20,
#     conf_fluoro=0.30,
#     conf_paciente=0.25,
#     camilla_area_min=0.05,
#     camilla_aspect_min=1.15,
#     hold_paciente_s=6.0,
#     conf_bata=0.30,
#     conf_guantes=0.28,
#     iou_absorcion=0.60,
#     fluoro_sat_media=25.0,
#     fluoro_laplaciano=45.0,
#     carto_area_min=0.04,
#     poli_hv=1.60,
#     poli_edge=0.12,
#     poli_oscuro_min=0.45,
#     poli_hv_b=1.20,
#     rojo_ablacion=0.004,
#     pose_conf=0.25,
#     kp_conf=0.50,
#     antebrazo_piel_alta=0.45,
#     antebrazo_piel_baja=0.12,
#     conf_guantes_mano=0.25,
#     votos_persona=12,
#     votos_min=3,
#     corte_correlacion=0.50,
#     ventana_suavizado_s=2.0,
#     min_segmento_s=6.0,
# )

# COLOR_FASE = [(200, 200, 90), (200, 160, 90), (90, 200, 120), (60, 190, 240), (70, 90, 240), (170, 120, 200)]
# FONT = cv2.FONT_HERSHEY_SIMPLEX


# # ==============================================================================
# # UTILIDADES
# # ==============================================================================
# def iou(a, b):
#     ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
#     ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
#     inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
#     ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
#     return inter / ua if ua > 0 else 0.0


# def limitar_lado(frame, max_lado=1920):
#     h, w = frame.shape[:2]
#     s = max_lado / float(max(h, w))
#     if s >= 1.0:
#         return frame
#     return cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


# def mejorar(frame, modo):
#     if modo == "off":
#         return frame
#     media = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
#     if modo == "auto" and media >= 85:
#         return frame
#     gamma = float(np.clip(np.log(0.45) / np.log(max(media, 10.0) / 255.0), 0.4, 1.0))
#     lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
#     lab = cv2.cvtColor(cv2.LUT(frame, lut), cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
#     return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# def recortar(frame, box):
#     h, w = frame.shape[:2]
#     x1, y1, x2, y2 = box
#     return frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]


# def histograma_hsv(frame):
#     small = cv2.resize(frame, (160, 90))
#     hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
#     return cv2.normalize(hist, hist).flatten()


# def es_pantalla_valida(box, shape):
#     h_img, w_img = shape[:2]
#     w, h = max(1, box[2] - box[0]), max(1, box[3] - box[1])
#     return 0.50 <= w / float(h) <= 4.8 and (w * h) / float(h_img * w_img) >= 0.006


# def es_camilla_real(box, shape):
#     h_img, w_img = shape[:2]
#     w, h = max(1, box[2] - box[0]), max(1, box[3] - box[1])
#     if w / float(h) < UMBRALES["camilla_aspect_min"] or (w * h) / float(h_img * w_img) < UMBRALES["camilla_area_min"]:
#         return False
#     return (box[1] + box[3]) / 2 >= h_img * 0.30


# # ==============================================================================
# # ANALISIS DE CONTENIDO
# # ==============================================================================
# def clasificar_pantalla(roi):
#     met = dict(s=0.0, b=0.0, o=0.0, hv=0.0, e=0.0, lap=0.0)
#     if roi.size == 0 or min(roi.shape[:2]) < 24:
#         return None, 0.0, met
#     U = UMBRALES
#     h0, w0 = roi.shape[:2]
#     roi = cv2.resize(roi, (320, max(24, int(h0 * 320 / w0))), interpolation=cv2.INTER_AREA)
#     hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#     hch, s, v = cv2.split(hsv)
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     H, W = gray.shape
#     total = float(gray.size)

#     frac_color = np.count_nonzero((s > 60) & (v > 50)) / total
#     lap = cv2.Laplacian(gray, cv2.CV_64F).var()
#     rojo = float(np.count_nonzero(((hch < 10) | (hch > 170)) & (s > 110) & (v > 80))) / total

#     k = max(3, int(min(H, W) * 0.02) | 1)
#     mask = (((s > 40) & (v > 35)).astype(np.uint8)) * 255
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
#     n, _, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
#     blob, central = 0.0, False
#     if n > 1:
#         i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
#         blob = stats[i, cv2.CC_STAT_AREA] / total
#         cx, cy = cent[i]
#         central = 0.1 * W < cx < 0.9 * W and 0.1 * H < cy < 0.9 * H

#     oscuro = np.count_nonzero(v < 65) / total
#     gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)).mean()
#     gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1)).mean()
#     hv = gy / (gx + 1e-6)
#     edge = float(np.count_nonzero(cv2.Canny(gray, 60, 150))) / total
#     met.update(s=float(s.mean()), b=float(blob), o=float(oscuro), hv=float(hv), e=edge, lap=float(lap))

#     if s.mean() < U["fluoro_sat_media"] and frac_color < 0.005 and lap >= U["fluoro_laplaciano"]:
#         return "fluoro", rojo, met
#     if (hv > U["poli_hv"] and edge > U["poli_edge"]) or (oscuro > U["poli_oscuro_min"] and hv > U["poli_hv_b"]):
#         return "poligrafo", rojo, met
#     if blob >= U["carto_area_min"] and central:
#         return "carto", rojo, met
#     return None, rojo, met


# def analizar_antebrazo(img, codo, muneca):
#     codo, muneca = np.asarray(codo, float), np.asarray(muneca, float)
#     L = float(np.hypot(*(muneca - codo)))
#     if L < 25:
#         return None
#     a = codo + 0.2 * (muneca - codo)
#     b = codo + 0.9 * (muneca - codo)
#     grosor = max(5, int(0.30 * L))
#     H, W = img.shape[:2]
#     x1 = int(max(0, min(a[0], b[0]) - grosor)); x2 = int(min(W, max(a[0], b[0]) + grosor))
#     y1 = int(max(0, min(a[1], b[1]) - grosor)); y2 = int(min(H, max(a[1], b[1]) + grosor))
#     if x2 - x1 < 8 or y2 - y1 < 8:
#         return None
#     crop = img[y1:y2, x1:x2]
#     mask = np.zeros(crop.shape[:2], np.uint8)
#     cv2.line(mask, (int(a[0] - x1), int(a[1] - y1)), (int(b[0] - x1), int(b[1] - y1)), 255, grosor)
#     sel = mask > 0
#     if np.count_nonzero(sel) < 80:
#         return None
#     hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#     ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
#     h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
#     piel = ((h <= 25) & (s >= 35) & (s <= 180) & (v >= 70)
#             & (ycc[..., 1] >= 133) & (ycc[..., 1] <= 173) & (ycc[..., 2] >= 77) & (ycc[..., 2] <= 127)
#             & (ycc[..., 0] >= 60))
#     return float(np.count_nonzero(piel & sel)) / float(np.count_nonzero(sel))


# # ==============================================================================
# # DETECTOR
# # ==============================================================================
# class Detector:
#     def __init__(self, modelo, imgsz, mejora, tiles_habilitados, usar_pose, modelo_pose, fps_analisis):
#         import torch
#         from ultralytics import YOLO
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"[*] YOLO-World ({modelo}) en {self.device.upper()}")
#         self.prompts = list(VOCABULARIO.keys())
#         self.tipos = [VOCABULARIO[p][0] for p in self.prompts]
#         self.model = YOLO(modelo)
#         self.model.to(self.device)
#         self.model.set_classes(self.prompts)
#         self.model_aux = None
#         if tiles_habilitados or usar_pose:
#             self.model_aux = YOLO(modelo)
#             self.model_aux.to(self.device)
#             self.model_aux.set_classes(self.prompts)
#         self.pose = None
#         if usar_pose:
#             print(f"[*] Pose ({modelo_pose})")
#             self.pose = YOLO(modelo_pose)
#             self.pose.to(self.device)
#         self.imgsz = imgsz
#         self.mejora = mejora
#         self.votos = defaultdict(lambda: deque(maxlen=UMBRALES["votos_persona"]))
#         self.votos_pantalla = defaultdict(lambda: deque(maxlen=10))
#         self.decay = 0.5 ** (1.0 / max(1.0, UMBRALES["hold_paciente_s"] * fps_analisis))
#         self.pac_mem = 0.0

#     def reset(self):
#         self.votos.clear()
#         self.votos_pantalla.clear()
#         self.pac_mem = 0.0
#         for m in (self.model, self.pose):
#             if m is None:
#                 continue
#             try:
#                 for t in m.predictor.trackers:
#                     t.reset()
#             except Exception:
#                 pass

#     def _dets(self, res, ox=0, oy=0):
#         out = []
#         if res.boxes is None:
#             return out
#         for b in res.boxes:
#             cid = int(b.cls[0])
#             if cid >= len(self.tipos):
#                 continue
#             x1, y1, x2, y2 = map(int, b.xyxy[0])
#             tid = int(b.id[0]) if getattr(b, "id", None) is not None else None
#             out.append((self.tipos[cid], (x1 + ox, y1 + oy, x2 + ox, y2 + oy), float(b.conf[0]), tid))
#         return out

#     def _dets_tiles(self, entrada, existentes):
#         U = UMBRALES
#         H, W = entrada.shape[:2]
#         tw, th = int(W * 0.6), int(H * 0.6)
#         nuevos = []
#         for ox in (0, W - tw):
#             for oy in (0, H - th):
#                 r = self.model_aux.predict(entrada[oy:oy + th, ox:ox + tw], conf=U["conf_min"],
#                                            iou=0.45, imgsz=960, verbose=False)[0]
#                 for d in self._dets(r, ox, oy):
#                     if d[0] in ("paciente", "mano_desnuda"):
#                         continue
#                     if any(d[0] == e[0] and iou(d[1], e[1]) > 0.5 for e in existentes + nuevos):
#                         continue
#                     nuevos.append(d)
#         return nuevos

#     def _personas_pose(self, entrada):
#         U = UMBRALES
#         out = []
#         if self.pose is None:
#             return out
#         rp = self.pose.track(entrada, persist=True, conf=U["pose_conf"], imgsz=self.imgsz, verbose=False)[0]
#         if rp.boxes is None or rp.keypoints is None or len(rp.boxes) == 0:
#             return out
#         xy = rp.keypoints.xy.cpu().numpy()
#         kc = (rp.keypoints.conf.cpu().numpy() if rp.keypoints.conf is not None
#               else np.ones(xy.shape[:2], dtype=float))
#         for i, b in enumerate(rp.boxes):
#             tid = int(b.id[0]) if getattr(b, "id", None) is not None else None
#             out.append((tid, tuple(map(int, b.xyxy[0])), xy[i], kc[i]))
#         return out

#     def _clasificar_mano(self, entrada, centro, lado):
#         H, W = entrada.shape[:2]
#         r = int(lado / 2)
#         x1, y1 = int(max(0, centro[0] - r)), int(max(0, centro[1] - r))
#         x2, y2 = int(min(W, centro[0] + r)), int(min(H, centro[1] + r))
#         if x2 - x1 < 24 or y2 - y1 < 24:
#             return None
#         crop = cv2.resize(entrada[y1:y2, x1:x2], (320, 320), interpolation=cv2.INTER_CUBIC)
#         res = self.model_aux.predict(crop, conf=0.10, iou=0.45, imgsz=320, verbose=False)[0]
#         gl = md = 0.0
#         for b in (res.boxes if res.boxes is not None else []):
#             t, c = self.tipos[int(b.cls[0])], float(b.conf[0])
#             if t == "guantes":
#                 gl = max(gl, c)
#             elif t == "mano_desnuda":
#                 md = max(md, c)
#         return (x1, y1, x2, y2), gl, md

#     def inferir(self, frame, usar_tiles=False):
#         U = UMBRALES
#         entrada = mejorar(frame, self.mejora)
#         res = self.model.track(entrada, persist=True, conf=U["conf_min"], iou=0.45,
#                                imgsz=self.imgsz, verbose=False)[0]
#         dets = self._dets(res)
#         if usar_tiles and self.model_aux is not None:
#             dets += self._dets_tiles(entrada, dets)

#         ev = dict.fromkeys(CAMPOS_EV, 0.0)
#         dibujo = []

#         def sube(k, v):
#             ev[k] = max(ev[k], v)

#         def rechazar(box, txt):
#             dibujo.append(dict(box=box, texto=txt, color=(130, 130, 130), ok=False, pts=None))

#         def aceptar(box, txt, color, pts=None):
#             dibujo.append(dict(box=box, texto=txt, color=color, ok=True, pts=pts))

#         dets.sort(key=lambda d: -d[2])
#         pant_vistas = []
#         ignorar = [d for d in dets if d[0] == "ignorar"]
#         for g in ignorar:
#             rechazar(g[1], f"ignorar {int(g[2] * 100)}%")

#         pacientes = []
#         for tipo, box, conf, tid in dets:
#             pct = int(conf * 100)
#             if tipo in ("ignorar", "mano_desnuda"):
#                 continue

#             if tipo in PANTALLAS:
#                 hint = {"pantalla_carto": "carto", "pantalla_poli": "poligrafo"}.get(tipo)
#                 umbral = (U["conf_fluoro"] if tipo == "fluoro"
#                           else U["conf_pantalla_hint"] if hint else U["conf_pantalla"])
#                 nombre = VOCABULARIO_INV[tipo]
#                 if any(iou(box, g[1]) > U["iou_absorcion"] and g[2] >= conf for g in ignorar):
#                     rechazar(box, f"{nombre} {pct}% absorbida")
#                     continue
#                 if conf < umbral:
#                     rechazar(box, f"{nombre} {pct}% conf baja")
#                     continue
#                 if not es_pantalla_valida(box, frame.shape):
#                     rechazar(box, f"{nombre} {pct}% forma")
#                     continue
#                 if any(iou(box, b) > 0.60 for b in pant_vistas):
#                     continue
#                 pant_vistas.append(box)
#                 contenido, rojo, met = clasificar_pantalla(recortar(frame, box))
#                 m = (f" [det:{tipo[9:] or 'mon'} s{met['s']:.0f} b{met['b']:.2f} o{met['o']:.2f} "
#                      f"hv{met['hv']:.1f} e{met['e']:.2f}]")
#                 final, fuerza = contenido, 0.4 + 0.6 * conf
#                 if final is None and hint:
#                     final, fuerza = hint, conf
#                 if tid is not None:
#                     vp = self.votos_pantalla[tid]
#                     vp.append(final)
#                     top = Counter(v for v in vp if v).most_common(1)
#                     if top and top[0][1] >= 2 and top[0][1] >= 0.4 * len(vp) and final != top[0][0]:
#                         final, fuerza = top[0][0], min(fuerza, conf)
#                 if tipo == "fluoro" and final != "fluoro":
#                     rechazar(box, f"Fluoro {pct}% sin textura{m}")
#                     continue
#                 if final == "fluoro" and conf < U["conf_fluoro"]:
#                     final = None
#                 if final == "carto":
#                     sube("navegador", fuerza)
#                     sube("carto_rojo", rojo)
#                     aceptar(box, f"Navegador 3D (CARTO) {pct}%{m}", (0, 255, 255))
#                 elif final == "poligrafo":
#                     sube("poligrafo", fuerza)
#                     aceptar(box, f"Poligrafo (EP) {pct}%{m}", (0, 200, 255))
#                 elif final == "fluoro":
#                     sube("fluoro", fuerza)
#                     aceptar(box, f"Fluoroscopia {pct}%{m}", (255, 200, 0))
#                 else:
#                     rechazar(box, f"Monitor {pct}% contenido?{m}")

#             elif tipo == "paciente":
#                 if not es_camilla_real(box, frame.shape):
#                     rechazar(box, f"Paciente {pct}% forma/pos")
#                 elif conf < U["conf_paciente"]:
#                     rechazar(box, f"Paciente {pct}% conf baja")
#                 else:
#                     sube("paciente", conf)
#                     pacientes.append(box)
#                     aceptar(box, f"Paciente en camilla {pct}%", (50, 220, 50))

#             elif tipo == "bata":
#                 if conf >= U["conf_bata"]:
#                     sube("bata", conf)
#                     aceptar(box, f"Bata esteril {pct}%", (50, 220, 50))
#                 else:
#                     rechazar(box, f"Bata {pct}% conf baja")

#             elif tipo == "guantes":
#                 if conf >= U["conf_guantes"]:
#                     sube("guantes", conf)
#                     aceptar(box, f"Guantes {pct}%", (255, 255, 0))
#                 else:
#                     rechazar(box, f"Guantes {pct}% conf baja")

#         self.pac_mem = max(ev["paciente"], self.pac_mem * self.decay)
#         if self.pac_mem < 0.10:
#             self.pac_mem = 0.0
#         ev["paciente"] = self.pac_mem

#         n_personas = 0
#         for tid, box, kxy, kc in self._personas_pose(entrada):
#             if any(iou(box, p) > 0.40 for p in pacientes):
#                 rechazar(box, "persona = paciente")
#                 continue
#             n_personas += 1
#             votos_brazo, pts, ratios = [], [], []
#             for ie, iw in ((7, 9), (8, 10)):
#                 if kc[ie] < U["kp_conf"] or kc[iw] < U["kp_conf"]:
#                     continue
#                 codo, mun = kxy[ie], kxy[iw]
#                 pts += [(float(codo[0]), float(codo[1])), (float(mun[0]), float(mun[1]))]
#                 r = analizar_antebrazo(entrada, codo, mun)
#                 if r is not None:
#                     ratios.append(r)
#                     votos_brazo.append("bare" if r > U["antebrazo_piel_alta"]
#                                        else "cubierto" if r < U["antebrazo_piel_baja"] else None)
#                 if self.model_aux is not None:
#                     L = float(np.hypot(*(mun - codo)))
#                     centro = mun + 0.25 * (mun - codo)
#                     out = self._clasificar_mano(entrada, centro, float(np.clip(1.1 * L, 48, 260)))
#                     if out is not None:
#                         caja, gl, md = out
#                         if gl >= U["conf_guantes_mano"] and gl >= md:
#                             sube("guantes", gl)
#                             aceptar(caja, f"Guantes (mano) {int(gl * 100)}%", (255, 255, 0))
#                         elif md >= U["conf_guantes_mano"]:
#                             aceptar(caja, f"Mano desnuda {int(md * 100)}%", (180, 180, 255))
#             voto = "bare" if "bare" in votos_brazo else "cubierto" if "cubierto" in votos_brazo else None
#             decision = voto
#             if tid is not None:
#                 if voto:
#                     self.votos[tid].append(voto)
#                 v = self.votos[tid]
#                 decision = None
#                 if len(v) >= U["votos_min"]:
#                     c, n = Counter(v).most_common(1)[0]
#                     if n >= 0.6 * len(v):
#                         decision = c
#             etiq = f"id{tid}" if tid is not None else "id?"
#             ptxt = f" piel{max(ratios):.2f}" if ratios else " (sin antebrazos visibles)"
#             if decision == "bare":
#                 ev["brazos_al_aire"] = 1.0
#                 aceptar(box, f"Sin bata (antebrazos) {etiq}{ptxt}", (0, 140, 255), pts)
#             elif decision == "cubierto":
#                 sube("bata", 0.5)
#                 aceptar(box, f"Personal cubierto {etiq}{ptxt}", (80, 200, 80), pts)
#             else:
#                 aceptar(box, f"Persona ? {etiq}{ptxt}", (200, 200, 200), pts)
#         ev["n_personas"] = n_personas
#         return ev, dibujo


# VOCABULARIO_INV = {}
# for _p, (_t, _e) in VOCABULARIO.items():
#     VOCABULARIO_INV.setdefault(_t, _e)


# # ==============================================================================
# # VISOR
# # ==============================================================================
# def dibujar_det(img, box, texto, color, esc, grosor):
#     x1, y1, x2, y2 = [int(v * esc) for v in box]
#     cv2.rectangle(img, (x1, y1), (x2, y2), color, grosor)
#     (tw, th), _ = cv2.getTextSize(texto, FONT, 0.38, 1)
#     ty = max(th + 6, y1)
#     cv2.rectangle(img, (x1, ty - th - 6), (min(img.shape[1] - 1, x1 + tw + 8), ty), (15, 15, 15), -1)
#     cv2.putText(img, texto, (x1 + 4, ty - 4), FONT, 0.38, color, 1, cv2.LINE_AA)


# COLOR_VERDE = (0, 255, 0)
# COLOR_COMPLETA = (70, 180, 70)
# COLOR_AMARILLO = (0, 215, 255)
# COLOR_GRIS = (120, 120, 120)

# UMBRAL_FASE = 0.35
# CONFIRM_S = [3.0, 3.0, 3.0, 3.0, 3.0, 12.0]


# class GestorFases:
#     def __init__(self, fps_analisis):
#         self.fps = fps_analisis
#         self.detectadas = set()
#         self.actual = None
#         self.max_idx = -1
#         self.racha = [0] * len(FASES)

#     def _permitidas(self):
#         perm = [k >= self.max_idx for k in range(len(FASES))]
#         for desde, hacia in RETROCESOS:
#             if self.max_idx == desde:
#                 perm[hacia] = True
#         return perm

#     def actualizar(self, scores, corte=False):
#         if corte:
#             self.racha = [0] * len(FASES)
#         s = np.where(self._permitidas(), scores, -1.0)
#         k = int(np.argmax(s))
#         cand = k if s[k] >= UMBRAL_FASE else None
#         self.racha = [r + 1 if i == cand else 0 for i, r in enumerate(self.racha)]
#         if cand is None or cand == self.actual:
#             return
#         salto = cand - self.max_idx
#         req = CONFIRM_S[cand] * (2 if salto > 1 else 1)
#         if self.racha[cand] >= req * self.fps:
#             self.actual = cand
#             self.detectadas.add(cand)
#             self.max_idx = max(self.max_idx, cand)

#     def obtener_estados(self):
#         estados = []
#         for i, nombre in enumerate(FASES):
#             if i == self.actual:
#                 estados.append((nombre, "[ ACTIVA ]", COLOR_VERDE))
#             elif i in self.detectadas:
#                 estados.append((nombre, "[ COMPLETADA ]", COLOR_COMPLETA))
#             elif i < self.max_idx:
#                 estados.append((nombre, "[ PASADA NO DETECTADA ]", COLOR_AMARILLO))
#             else:
#                 estados.append((nombre, "[ PENDIENTE ]", COLOR_GRIS))
#         return estados


# SEMIVIDA_S = dict(navegador=1.5, poligrafo=1.5, fluoro=1.5, paciente=1.5, bata=1.5, guantes=1.0, brazos_al_aire=1.0)
# ON_T, OFF_T = 0.50, 0.25


# class EvidenciaTemporal:
#     def __init__(self, fps_analisis):
#         self.dt = 1.0 / fps_analisis
#         self.reset()

#     def reset(self):
#         self.p = dict.fromkeys(CAMPOS_EV, 0.0)
#         self.m = dict.fromkeys(CAMPOS_EV, 0.0)
#         self.on = dict.fromkeys(CAMPOS_EV, False)

#     def actualizar(self, ev):
#         out = {}
#         for k in CAMPOS_EV:
#             x = float(ev.get(k, 0.0))
#             if k == "carto_rojo":
#                 a = 0.5 ** (self.dt / 1.5)
#                 self.p[k] = a * self.p[k] + (1 - a) * x
#                 out[k] = self.p[k] if self.on["navegador"] else 0.0
#                 continue
#             a = 0.5 ** (self.dt / SEMIVIDA_S[k])
#             self.p[k] = a * self.p[k] + (1 - a) * (1.0 if x > 0 else 0.0)
#             if x > 0:
#                 self.m[k] = x if self.m[k] == 0 else 0.7 * self.m[k] + 0.3 * x
#             if self.on[k] and self.p[k] < OFF_T:
#                 self.on[k] = False
#                 self.m[k] = 0.0
#             elif not self.on[k] and self.p[k] > ON_T:
#                 self.on[k] = True
#             out[k] = self.m[k] if self.on[k] else 0.0
#         return out


# def clave_visual(texto):
#     return re.sub(r"\d+(\.\d+)?%?", "", texto.split("[")[0]).strip()


# class RastroVisual:
#     def __init__(self, vida=2, alfa=0.5):
#         self.vida, self.alfa, self.prev = vida, alfa, []

#     def reset(self):
#         self.prev = []

#     def actualizar(self, dibujo):
#         salida, usados = [], set()
#         for d in (x for x in dibujo if x["ok"]):
#             clave = clave_visual(d["texto"])
#             mejor, mi = None, 0.30
#             for j, p in enumerate(self.prev):
#                 if j in usados or p["clave"] != clave:
#                     continue
#                 v = iou(d["box"], p["box"])
#                 if v > mi:
#                     mejor, mi = j, v
#             box = d["box"]
#             if mejor is not None:
#                 usados.add(mejor)
#                 pb = self.prev[mejor]["box"]
#                 box = tuple(int(self.alfa * pb[i] + (1 - self.alfa) * box[i]) for i in range(4))
#             salida.append(dict(d, box=box, clave=clave, edad=0, fantasma=False))
#         for j, p in enumerate(self.prev):
#             if j not in usados and p["edad"] < self.vida:
#                 salida.append(dict(p, edad=p["edad"] + 1, fantasma=True))
#         self.prev = salida
#         return salida + [x for x in dibujo if not x["ok"]]


# def renderizar_panel(alto, ancho, t, ev_prom, estados, scores):
#     panel = np.full((alto, ancho, 3), (33, 26, 22), dtype=np.uint8)
#     cv2.putText(panel, "ELECTROFISIOLOGIA - AI", (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
#     cv2.putText(panel, f"Tiempo: {int(t // 60):02d}:{int(t % 60):02d}", (20, 54), FONT, 0.42, (150, 150, 150), 1)
#     cv2.line(panel, (20, 66), (ancho - 20, 66), (75, 65, 60), 1)

#     cv2.putText(panel, "EVIDENCIAS (estables)", (20, 84), FONT, 0.42, (200, 200, 200), 1)
#     y = 94
#     for k in CAMPOS_EV:
#         val = ev_prom.get(k, 0.0)
#         if k == "carto_rojo":
#             val = min(1.0, val / UMBRALES["rojo_ablacion"])
#         cv2.putText(panel, k, (20, y + 9), FONT, 0.36, (220, 220, 220), 1)
#         cv2.rectangle(panel, (140, y + 1), (ancho - 30, y + 9), (60, 55, 50), 1)
#         cv2.rectangle(panel, (140, y + 1), (140 + int((ancho - 170) * min(1, val)), y + 9), (90, 200, 90), -1)
#         y += 15

#     cv2.putText(panel, "FLUJO QUIRURGICO:", (20, 236), FONT, 0.46, (200, 200, 200), 1)
#     y = 262
#     for i, (nombre, estado_str, color) in enumerate(estados):
#         cv2.putText(panel, estado_str, (28, y), FONT, 0.42, color, 1, cv2.LINE_AA)
#         cv2.putText(panel, f"{scores[i]:.2f}", (ancho - 50, y), FONT, 0.36, (130, 130, 130), 1, cv2.LINE_AA)
#         cv2.putText(panel, nombre, (28, y + 16), FONT, 0.37, (230, 230, 230), 1, cv2.LINE_AA)
#         cv2.rectangle(panel, (16, y - 12), (20, y + 20), color, -1)
#         y += 40

#     cv2.line(panel, (20, alto - 46), (ancho - 20, alto - 46), (75, 65, 60), 1)
#     cv2.putText(panel, "Verde: activa / completada", (20, alto - 29), FONT, 0.36, COLOR_VERDE, 1)
#     cv2.putText(panel, "Amarillo: pasada no detectada", (20, alto - 12), FONT, 0.36, COLOR_AMARILLO, 1)
#     return panel


# # ==============================================================================
# # PASADA 1: EXTRACCION MULTIHILO (REPRODUCCION FLUIDA)
# # ==============================================================================
# def extraer(video, csv_path, fps_analisis, modelo, imgsz, mejora, ver=False, tiles=False, debug=False,
#             usar_pose=True, modelo_pose="yolov8s-pose.pt"):
#     cap = cv2.VideoCapture(video)
#     if not cap.isOpened():
#         raise SystemExit(f"No se pudo abrir {video}")
#     fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
#     duracion_frame_s = 1.0 / fps
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#     paso = max(1, int(round(fps / fps_analisis)))

#     # Estructuras de sincronizacion y paso de mensajes entre hilos
#     cola_ia = queue.Queue(maxsize=2)
#     stop_event = threading.Event()
#     init_event = threading.Event()
#     lock_compartido = threading.Lock()

#     conf_compartida = {
#         "mejora": mejora,
#         "tiles": tiles,
#     }
#     estado_ui = {
#         "debug": debug,
#     }
#     estado_compartido = {
#         "dibujo": [],
#         "ev_prom": {k: 0.0 for k in CAMPOS_EV},
#         "scores": np.zeros(len(FASES)),
#         "estados": [(f, "[ PENDIENTE ]", COLOR_GRIS) for f in FASES],
#         "n_muestras": 0,
#     }
#     columnas = ["frame", "t", "corte"] + CAMPOS_EV + ["n_personas"]

#     # --------------------------------------------------------------------------
#     # HILO DE IA (Worker en segundo plano)
#     # --------------------------------------------------------------------------
#     def worker_ia():
#         try:
#             det = Detector(modelo, imgsz, mejora, tiles, usar_pose, modelo_pose, fps_analisis)
#             gestor = GestorFases(fps_analisis)
#             temporal = EvidenciaTemporal(fps_analisis)
#             rastro = RastroVisual()
#             init_event.set()
#         except Exception as e:
#             print(f"[!] Error inicializando modelos en el hilo de IA: {e}")
#             init_event.set()
#             return

#         prev_hist = None
#         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
#         with open(csv_path, "w", newline="", encoding="utf-8") as f:
#             w = csv.DictWriter(f, fieldnames=columnas)
#             w.writeheader()

#             while not stop_event.is_set():
#                 try:
#                     item = cola_ia.get(timeout=0.1)
#                 except queue.Empty:
#                     continue
#                 if item is None:
#                     cola_ia.task_done()
#                     break

#                 frame_proc, idx_ia, t_ia = item

#                 hist = histograma_hsv(frame_proc)
#                 corte = int(prev_hist is not None and
#                             cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < UMBRALES["corte_correlacion"])
#                 prev_hist = hist
#                 if corte:
#                     det.reset()
#                     temporal.reset()
#                     rastro.reset()

#                 with lock_compartido:
#                     det.mejora = conf_compartida["mejora"]
#                     usar_t = conf_compartida["tiles"]

#                 ev, dib = det.inferir(frame_proc, usar_tiles=usar_t)
#                 fila = {"frame": idx_ia, "t": round(t_ia, 3), "corte": corte, "n_personas": ev["n_personas"]}
#                 fila.update({k: round(float(ev[k]), 4) for k in CAMPOS_EV})
#                 w.writerow(fila)
#                 f.flush()

#                 dib_suav = rastro.actualizar(dib)
#                 ev_prom = temporal.actualizar(ev)
#                 scores = puntuar_fases({k: np.array([v]) for k, v in ev_prom.items()}, 1)[0]
#                 gestor.actualizar(scores, bool(corte))
#                 estados = gestor.obtener_estados()

#                 with lock_compartido:
#                     estado_compartido["dibujo"] = dib_suav
#                     estado_compartido["ev_prom"] = ev_prom
#                     estado_compartido["scores"] = scores
#                     estado_compartido["estados"] = estados
#                     estado_compartido["n_muestras"] += 1
#                     n_m = estado_compartido["n_muestras"]

#                 cola_ia.task_done()
#                 if not ver and n_m % 20 == 0:
#                     print(f"    {n_m} muestras analizadas (t={t_ia:.1f}s)", end="\r")

#     # Iniciar hilo secundario
#     hilo_ia = threading.Thread(target=worker_ia, daemon=True)
#     hilo_ia.start()
#     init_event.wait()

#     # --------------------------------------------------------------------------
#     # HILO PRINCIPAL (Reproduccion y Renderizado UI)
#     # --------------------------------------------------------------------------
#     def pintar(frame_vis, t):
#         ancho_v = 960
#         alto_v = int(frame_vis.shape[0] * ancho_v / frame_vis.shape[1])
#         esc = ancho_v / frame_vis.shape[1]
#         img = cv2.resize(frame_vis, (ancho_v, alto_v), interpolation=cv2.INTER_AREA)
#         alto_t = max(alto_v, 540)
#         if alto_t > alto_v:
#             img = cv2.copyMakeBorder(img, 0, alto_t - alto_v, 0, 0, cv2.BORDER_CONSTANT, value=(33, 26, 22))

#         with lock_compartido:
#             dibujo_actual = list(estado_compartido["dibujo"])
#             ev_prom_actual = dict(estado_compartido["ev_prom"])
#             estados_actual = list(estado_compartido["estados"])
#             scores_actual = np.copy(estado_compartido["scores"])
#             modo_mejora = conf_compartida["mejora"]
#             modo_tiles = conf_compartida["tiles"]

#         for d in dibujo_actual:
#             if d["ok"] and d.get("fantasma"):
#                 dibujar_det(img, d["box"], d["texto"], tuple(int(c * 0.55) for c in d["color"]), esc, 1)
#             elif d["ok"]:
#                 dibujar_det(img, d["box"], d["texto"], d["color"], esc, 2)
#             elif estado_ui["debug"]:
#                 dibujar_det(img, d["box"], d["texto"], d["color"], esc, 1)
#             if estado_ui["debug"] and d.get("pts"):
#                 for px, py in d["pts"]:
#                     cv2.circle(img, (int(px * esc), int(py * esc)), 3, (0, 255, 255), -1)

#         cv2.putText(img, f"MEJORA:{modo_mejora}  DEBUG:{'on' if estado_ui['debug'] else 'off'}  "
#                          f"TILES:{'on' if modo_tiles else 'off'}   [c][d][t] [espacio] [q]",
#                     (8, alto_v - 8), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
#         panel = renderizar_panel(alto_t, 380, t, ev_prom_actual, estados_actual, scores_actual)
#         cv2.imshow("Monitor Quirurgico EP", np.hstack((img, panel)))

#     def tecla(k):
#         if k == ord("q"):
#             return True
#         if k == ord("c"):
#             with lock_compartido:
#                 conf_compartida["mejora"] = {"off": "on", "on": "auto", "auto": "off"}[conf_compartida["mejora"]]
#         elif k == ord("d"):
#             estado_ui["debug"] = not estado_ui["debug"]
#         elif k == ord("t"):
#             with lock_compartido:
#                 conf_compartida["tiles"] = not conf_compartida["tiles"]
#         return False

#     idx = 0
#     k = 0
#     while True:
#         t_frame_inicio = time.perf_counter()
#         muestra = (idx % paso == 0)

#         if ver:
#             ok, raw = cap.read()
#         else:
#             ok = cap.grab()
#             raw = None
#             if ok and muestra:
#                 ok, raw = cap.retrieve()
#         if not ok:
#             break

#         frame = limitar_lado(raw) if raw is not None else None
#         t = idx / fps

#         if muestra and frame is not None:
#             if ver:
#                 try:
#                     # En vivo no bloqueamos el hilo de visualizacion si la cola estuviera llena
#                     cola_ia.put_nowait((frame.copy(), idx, t))
#                 except queue.Full:
#                     pass
#             else:
#                 cola_ia.put((frame.copy(), idx, t), block=True)

#         if ver and frame is not None:
#             pintar(frame, t)
#             dt_proc = time.perf_counter() - t_frame_inicio
#             espera_ms = max(1, int(round((duracion_frame_s - dt_proc) * 1000)))
#             k = cv2.waitKey(espera_ms) & 0xFF
#             if k == ord(" "):
#                 while True:
#                     k = cv2.waitKey(0) & 0xFF
#                     if k == ord(" "):
#                         break
#                     if tecla(k):
#                         idx = None
#                         break
#                     pintar(frame, t)
#                 if idx is None:
#                     break
#             elif tecla(k):
#                 break
#         idx += 1

#     # Finalizacion limpia del hilo
#     if idx is None or k == ord("q"):
#         stop_event.set()
#         while not cola_ia.empty():
#             try:
#                 cola_ia.get_nowait()
#                 cola_ia.task_done()
#             except Exception:
#                 break
#     cola_ia.put(None)
#     hilo_ia.join()
#     cap.release()
#     if ver:
#         cv2.destroyAllWindows()

#     with lock_compartido:
#         total_muestras = estado_compartido["n_muestras"]
#     print(f"\n[+] Evidencias: {total_muestras} muestras -> {csv_path}")


# # ==============================================================================
# # PASADA 2: SEGMENTACION
# # ==============================================================================
# def cargar_csv(path):
#     with open(path, encoding="utf-8") as f:
#         filas = list(csv.DictReader(f))
#     if not filas:
#         raise SystemExit("CSV vacio")
#     return {k: np.array([float(r[k]) for r in filas]) for k in filas[0].keys()}


# def suavizar(x, n):
#     if n <= 1:
#         return x
#     pad = n // 2
#     xp = np.pad(x, (pad, n - 1 - pad), mode="edge")
#     return np.convolve(xp, np.ones(n) / n, mode="valid")


# def puntuar_fases(d, n_suav):
#     U = UMBRALES
#     g = {k: suavizar(d[k], n_suav) for k in CAMPOS_EV}
#     ep = np.maximum(g["navegador"], g["poligrafo"])
#     pant = np.maximum(ep, g["fluoro"])
#     pac = g["paciente"]
#     est = np.maximum(g["bata"], g["guantes"])
#     abl = ep * np.clip(g["carto_rojo"] / U["rojo_ablacion"], 0, 1)

#     s = np.zeros((len(ep), 6))
#     s[:, 0] = np.maximum(g["brazos_al_aire"], est) * (1 - pac)
#     s[:, 1] = pac * (1 - est) * (1 - pant)
#     s[:, 2] = pac * est * (1 - pant)
#     s[:, 3] = pant * (1 - abl)
#     s[:, 4] = np.maximum(abl, 0.15 * ep * pac)
#     s[:, 5] = pac * est * (1 - pant)
#     return s


# RETROCESOS = [(4, 3)]


# def viterbi(logp, cortes, pen_avance=-6.0, pen_atras=-8.0, max_salto=2):
#     T, K = logp.shape
#     dp = np.full((T, K), -np.inf)
#     bp = np.zeros((T, K), dtype=int)
#     for k in range(K):
#         dp[0, k] = logp[0, k] + 0.5 * pen_avance * k
#     for t in range(1, T):
#         m = 0.3 if cortes[t] else 1.0
#         for k in range(K):
#             cands = [(dp[t - 1, k], k)]
#             for n in range(1, max_salto + 1):
#                 if k - n >= 0:
#                     cands.append((dp[t - 1, k - n] + m * pen_avance * n, k - n))
#             for desde, hacia in RETROCESOS:
#                 if hacia == k:
#                     cands.append((dp[t - 1, desde] + m * pen_atras, desde))
#             score, prev = max(cands, key=lambda c: c[0])
#             dp[t, k] = score + logp[t, k]
#             bp[t, k] = prev
#     path = [int(np.argmax(dp[-1]))]
#     for t in range(T - 1, 0, -1):
#         path.append(int(bp[t, path[-1]]))
#     return np.array(path[::-1])


# def segmentos_desde_path(ts, path):
#     segs, ini = [], 0
#     for i in range(1, len(path) + 1):
#         if i == len(path) or path[i] != path[ini]:
#             fin_t = ts[i] if i < len(ts) else ts[-1]
#             segs.append((int(path[ini]), float(ts[ini]), float(fin_t)))
#             ini = i
#     return segs


# def fmt(t):
#     return f"{int(t // 60):02d}:{int(t % 60):02d}"


# def limpiar_path(ts, path, min_s):
#     path = path.copy()
#     while True:
#         segs = segmentos_desde_path(ts, path)
#         cortos = [(i, s) for i, s in enumerate(segs) if len(segs) > 1 and s[2] - s[1] < min_s]
#         if not cortos:
#             return path
#         i, (k, a, b) = min(cortos, key=lambda x: x[1][2] - x[1][1])
#         destino = segs[i - 1][0] if i > 0 else segs[i + 1][0]
#         path[(ts >= a) & (ts < b) if i < len(segs) - 1 else (ts >= a)] = destino


# def segmentar(csv_path, out_path, fps_analisis):
#     d = cargar_csv(csv_path)
#     ts = d["t"]
#     n_suav = max(1, int(round(UMBRALES["ventana_suavizado_s"] * fps_analisis)))
#     s = puntuar_fases(d, n_suav)
#     path = viterbi(np.log(s + 0.05), d["corte"])
#     path = limpiar_path(ts, path, UMBRALES["min_segmento_s"])
#     segs = segmentos_desde_path(ts, path)

#     with open(out_path, "w", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         w.writerow(["fase", "nombre", "inicio_s", "fin_s"])
#         for k, a, b in segs:
#             w.writerow([k + 1, FASES[k], round(a, 2), round(b, 2)])

#     print("\n=== SEGMENTACION ===")
#     for k, a, b in segs:
#         print(f"  {fmt(a)} - {fmt(b)}  {FASES[k]}")
#     for k in range(len(FASES)):
#         if k not in {k_ for k_, _, _ in segs}:
#             print(f"  [no detectada] {FASES[k]}")
#     print(f"[+] Segmentos -> {out_path}")
#     return ts, path


# # ==============================================================================
# # VALIDACION Y RENDER
# # ==============================================================================
# def evaluar(gt_path, ts, path):
#     gt = np.full(len(ts), -1)
#     with open(gt_path, encoding="utf-8") as f:
#         filas = [(int(r["fase"]) - 1, float(r["inicio_s"]), float(r["fin_s"])) for r in csv.DictReader(f)]
#     for k, a, b in filas:
#         gt[(ts >= a) & (ts < b)] = k
#     ok = gt >= 0
#     if not ok.any():
#         print("[!] Ground truth sin solape con el video")
#         return
#     K = len(FASES)
#     conf = np.zeros((K, K), dtype=int)
#     for g_, p_ in zip(gt[ok], path[ok]):
#         conf[g_, p_] += 1
#     print(f"\n=== VALIDACION === precision por muestra: {100 * np.mean(gt[ok] == path[ok]):.1f}%")
#     print("matriz de confusion (filas = real, columnas = predicho):")
#     print(conf)
#     for k, a, _ in filas:
#         pred = ts[path == k]
#         err = f"{pred[0] - a:+.1f} s" if len(pred) else "fase no detectada"
#         print(f"  inicio fase {k + 1}: real {fmt(a)} | error {err}")


# def render(video, ts, path, out_path, ancho=960):
#     cap = cv2.VideoCapture(video)
#     fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
#     w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     alto = int(h0 * ancho / w0)
#     out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))
#     dur = ts[-1] if len(ts) else 1.0
#     idx = 0
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break
#         t = idx / fps
#         j = max(0, int(np.searchsorted(ts, t, side="right")) - 1)
#         k = int(path[min(len(path) - 1, j)])
#         frame = cv2.resize(frame, (ancho, alto))
#         cv2.rectangle(frame, (0, 0), (ancho, 34), (20, 24, 30), -1)
#         cv2.putText(frame, f"{fmt(t)}  {FASES[k]}", (10, 23), FONT, 0.55, COLOR_FASE[k], 1, cv2.LINE_AA)
#         for i in range(len(ts) - 1):
#             x1, x2 = int(ancho * ts[i] / dur), int(ancho * ts[i + 1] / dur)
#             cv2.rectangle(frame, (x1, alto - 14), (max(x2, x1 + 1), alto), COLOR_FASE[int(path[i])], -1)
#         xc = int(ancho * min(t, dur) / dur)
#         cv2.line(frame, (xc, alto - 18), (xc, alto), (255, 255, 255), 2)
#         out.write(frame)
#         idx += 1
#     cap.release()
#     out.release()
#     print(f"[+] Video con timeline -> {out_path}")


# # ==============================================================================
# # MAIN
# # ==============================================================================
# def main():
#     ap = argparse.ArgumentParser(description="Hitos de electrofisiologia en video")
#     ap.add_argument("--video", required=True)
#     ap.add_argument("--csv", help="CSV de evidencias (por defecto <video>_evidencias.csv)")
#     ap.add_argument("--fps", type=float, default=4.0, help="fps efectivos de analisis")
#     ap.add_argument("--modelo", default="yolov8l-worldv2.pt")
#     ap.add_argument("--imgsz", type=int, default=1088)
#     ap.add_argument("--mejora", choices=["off", "on", "auto"], default="auto",
#                     help="gamma+CLAHE: off, on, auto (solo si el frame es oscuro). Tecla 'c' en el visor")
#     ap.add_argument("--no-pose", action="store_true", help="desactivar YOLOv8-pose (sin atuendo ni manos)")
#     ap.add_argument("--modelo-pose", default="yolov8s-pose.pt")
#     ap.add_argument("--ver", action="store_true", help="mostrar video con detecciones y panel")
#     ap.add_argument("--debug", action="store_true", help="mostrar detecciones rechazadas, motivos y metricas")
#     ap.add_argument("--tiles", action="store_true", help="cargar modelo de tiles (pantallas pequenas); conmutable con 't'")
#     ap.add_argument("--solo-segmentar", action="store_true")
#     ap.add_argument("--gt", help="CSV de ground truth (fase,inicio_s,fin_s)")
#     ap.add_argument("--render", action="store_true", help="generar video con timeline de fases")
#     args = ap.parse_args()

#     base = os.path.splitext(args.video)[0]
#     csv_ev = args.csv or f"{base}_evidencias.csv"
#     csv_seg = f"{base}_segmentos.csv"

#     if not args.solo_segmentar:
#         extraer(args.video, csv_ev, args.fps, args.modelo, args.imgsz, args.mejora,
#                 ver=args.ver, tiles=args.tiles, debug=args.debug,
#                 usar_pose=not args.no_pose, modelo_pose=args.modelo_pose)
#     ts, path = segmentar(csv_ev, csv_seg, args.fps)
#     if args.gt:
#         evaluar(args.gt, ts, path)
#     if args.render:
#         render(args.video, ts, path, f"{base}_fases.mp4")


# if __name__ == "__main__":
#     main()







































"""
Deteccion e hitos en video de electrofisiologia con Segmentacion SAM / FastSAM / YOLO-seg.
Version con Pipeline Sincronizado (Cero desfase espacial/temporal).

Uso:
  python ep_fases.py --video ruta.mp4 --ver
  python ep_fases.py --video ruta.mp4 --ver --modelo-seg yolov8s-seg.pt   # ultra rapido en tiempo real
  python ep_fases.py --video ruta.mp4 --ver --modelo-seg FastSAM-s.pt
  python ep_fases.py --video ruta.mp4 --solo-segmentar --gt gt.csv --render

Teclas del visor:
  q: salir | ESPACIO: pausa | c: iluminacion off/on/auto | d: debug | t: tiles | s: segmentacion on/off
"""
import argparse
import csv
import os
import queue
import re
import threading
import time
from collections import Counter, defaultdict, deque

import cv2
import numpy as np
import torch
from ultralytics import FastSAM, YOLO

# ==============================================================================
# CONFIGURACION
# ==============================================================================
FASES = [
    "1. Preparacion medica (Colocacion de bata y guantes)",
    "2. Preparacion de paciente y campo esteril",
    "3. Puncion y acceso vascular femoral",
    "4. Mapeo electroanatomico 3D (Navegador)",
    "5. Intervencion / Ablacion por cateter",
    "6. Retirada de material y cierre",
]

VOCABULARIO = {
    "monitor": ("pantalla", "Monitor"),
    "computer screen": ("pantalla", "Monitor"),
    "3D heart map on monitor": ("pantalla_carto", "Navegador 3D"),
    "ECG waveform monitor": ("pantalla_poli", "Poligrafo"),
    "x-ray monitor": ("fluoro", "Fluoroscopia"),
    "patient on operating table": ("paciente", "Paciente en camilla"),
    "operating table": ("paciente", "Paciente en camilla"),
    "sterile surgical drapes": ("paciente", "Paciente en camilla"),
    "surgical gown": ("bata", "Bata esteril"),
    "surgical gloves": ("guantes", "Guantes"),
    "gloved hand": ("guantes", "Guantes"),
    "bare hand": ("mano_desnuda", "Mano desnuda"),
    "cabinet": ("ignorar", ""),
    "sink": ("ignorar", ""),
    "cart": ("ignorar", ""),
}
PANTALLAS = {"pantalla", "pantalla_carto", "pantalla_poli", "fluoro"}

CAMPOS_EV = ["navegador", "poligrafo", "fluoro", "paciente", "bata", "guantes",
             "brazos_al_aire", "carto_rojo"]

UMBRALES = dict(
    conf_min=0.10,
    conf_pantalla=0.18,
    conf_pantalla_hint=0.20,
    conf_fluoro=0.30,
    conf_paciente=0.25,
    camilla_area_min=0.05,
    camilla_aspect_min=1.15,
    hold_paciente_s=6.0,
    conf_bata=0.30,
    conf_guantes=0.28,
    iou_absorcion=0.60,
    fluoro_sat_media=25.0,
    fluoro_laplaciano=45.0,
    carto_area_min=0.04,
    poli_hv=1.60,
    poli_edge=0.12,
    poli_oscuro_min=0.45,
    poli_hv_b=1.20,
    rojo_ablacion=0.004,
    pose_conf=0.25,
    kp_conf=0.50,
    antebrazo_piel_alta=0.45,
    antebrazo_piel_baja=0.12,
    conf_guantes_mano=0.25,
    votos_persona=12,
    votos_min=3,
    corte_correlacion=0.50,
    ventana_suavizado_s=2.0,
    min_segmento_s=6.0,
)

COLOR_FASE = [(200, 200, 90), (200, 160, 90), (90, 200, 120), (60, 190, 240), (70, 90, 240), (170, 120, 200)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ==============================================================================
# UTILIDADES
# ==============================================================================
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def limitar_lado(frame, max_lado=1280):
    h, w = frame.shape[:2]
    s = max_lado / float(max(h, w))
    if s >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def mejorar(frame, modo):
    if modo == "off":
        return frame
    media = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    if modo == "auto" and media >= 85:
        return frame
    gamma = float(np.clip(np.log(0.45) / np.log(max(media, 10.0) / 255.0), 0.4, 1.0))
    lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    lab = cv2.cvtColor(cv2.LUT(frame, lut), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def recortar(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    return frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]


def histograma_hsv(frame):
    small = cv2.resize(frame, (160, 90))
    hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def es_pantalla_valida(box, shape):
    h_img, w_img = shape[:2]
    w, h = max(1, box[2] - box[0]), max(1, box[3] - box[1])
    return 0.50 <= w / float(h) <= 4.8 and (w * h) / float(h_img * w_img) >= 0.006


def es_camilla_real(box, shape):
    h_img, w_img = shape[:2]
    w, h = max(1, box[2] - box[0]), max(1, box[3] - box[1])
    if w / float(h) < UMBRALES["camilla_aspect_min"] or (w * h) / float(h_img * w_img) < UMBRALES["camilla_area_min"]:
        return False
    return (box[1] + box[3]) / 2 >= h_img * 0.30


def clasificar_pantalla(roi):
    met = dict(s=0.0, b=0.0, o=0.0, hv=0.0, e=0.0, lap=0.0)
    if roi.size == 0 or min(roi.shape[:2]) < 24:
        return None, 0.0, met
    U = UMBRALES
    h0, w0 = roi.shape[:2]
    roi = cv2.resize(roi, (320, max(24, int(h0 * 320 / w0))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hch, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    total = float(gray.size)

    frac_color = np.count_nonzero((s > 60) & (v > 50)) / total
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    rojo = float(np.count_nonzero(((hch < 10) | (hch > 170)) & (s > 110) & (v > 80))) / total

    k = max(3, int(min(H, W) * 0.02) | 1)
    mask = (((s > 40) & (v > 35)).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blob, central = 0.0, False
    if n > 1:
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        blob = stats[i, cv2.CC_STAT_AREA] / total
        cx, cy = cent[i]
        central = 0.1 * W < cx < 0.9 * W and 0.1 * H < cy < 0.9 * H

    oscuro = np.count_nonzero(v < 65) / total
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)).mean()
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1)).mean()
    hv = gy / (gx + 1e-6)
    edge = float(np.count_nonzero(cv2.Canny(gray, 60, 150))) / total
    met.update(s=float(s.mean()), b=float(blob), o=float(oscuro), hv=float(hv), e=edge, lap=float(lap))

    if s.mean() < U["fluoro_sat_media"] and frac_color < 0.005 and lap >= U["fluoro_laplaciano"]:
        return "fluoro", rojo, met
    if (hv > U["poli_hv"] and edge > U["poli_edge"]) or (oscuro > U["poli_oscuro_min"] and hv > U["poli_hv_b"]):
        return "poligrafo", rojo, met
    if blob >= U["carto_area_min"] and central:
        return "carto", rojo, met
    return None, rojo, met


def analizar_antebrazo(img, codo, muneca):
    codo, muneca = np.asarray(codo, float), np.asarray(muneca, float)
    L = float(np.hypot(*(muneca - codo)))
    if L < 25:
        return None
    a = codo + 0.2 * (muneca - codo)
    b = codo + 0.9 * (muneca - codo)
    grosor = max(5, int(0.30 * L))
    H, W = img.shape[:2]
    x1 = int(max(0, min(a[0], b[0]) - grosor))
    x2 = int(min(W, max(a[0], b[0]) + grosor))
    y1 = int(max(0, min(a[1], b[1]) - grosor))
    y2 = int(min(H, max(a[1], b[1]) + grosor))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = img[y1:y2, x1:x2]
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.line(mask, (int(a[0] - x1), int(a[1] - y1)), (int(b[0] - x1), int(b[1] - y1)), 255, grosor)
    sel = mask > 0
    if np.count_nonzero(sel) < 80:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    piel = ((h <= 25) & (s >= 35) & (s <= 180) & (v >= 70)
            & (ycc[..., 1] >= 133) & (ycc[..., 1] <= 173) & (ycc[..., 2] >= 77) & (ycc[..., 2] <= 127)
            & (ycc[..., 0] >= 60))
    return float(np.count_nonzero(piel & sel)) / float(np.count_nonzero(sel))


# ==============================================================================
# DETECTOR + SEGMENTADOR
# ==============================================================================
class Detector:
    def __init__(self, modelo, imgsz, mejora, tiles_habilitados, usar_pose, modelo_pose,
                 usar_seg, modelo_seg, fps_analisis):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] GPU/Dispositivo: {self.device.upper()}")
        print(f"[*] Modelo Deteccion: {modelo}")
        
        self.prompts = list(VOCABULARIO.keys())
        self.tipos = [VOCABULARIO[p][0] for p in self.prompts]
        self.model = YOLO(modelo).to(self.device)
        self.model.set_classes(self.prompts)
        
        self.model_aux = None
        if tiles_habilitados or usar_pose:
            self.model_aux = YOLO(modelo).to(self.device)
            self.model_aux.set_classes(self.prompts)
            
        self.pose = None
        if usar_pose:
            self.pose = YOLO(modelo_pose).to(self.device)

        self.seg = None
        self.is_fastsam = False
        if usar_seg:
            print(f"[*] Modelo Segmentacion: {modelo_seg}")
            try:
                if "fastsam" in modelo_seg.lower():
                    self.seg = FastSAM(modelo_seg)
                    self.is_fastsam = True
                else:
                    self.seg = YOLO(modelo_seg)
                self.seg.to(self.device)
            except Exception as e:
                print(f"[!] Error al cargar segmentador ({e}). Continuando sin mascaras.")
                self.seg = None

        self.imgsz = imgsz
        self.mejora = mejora
        self.votos = defaultdict(lambda: deque(maxlen=UMBRALES["votos_persona"]))
        self.votos_pantalla = defaultdict(lambda: deque(maxlen=10))
        self.decay = 0.5 ** (1.0 / max(1.0, UMBRALES["hold_paciente_s"] * fps_analisis))
        self.pac_mem = 0.0

    def reset(self):
        self.votos.clear()
        self.votos_pantalla.clear()
        self.pac_mem = 0.0
        for m in (self.model, self.pose, self.seg):
            if m is None:
                continue
            try:
                for t in m.predictor.trackers:
                    t.reset()
            except Exception:
                pass

    def _dets(self, res, ox=0, oy=0):
        out = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            cid = int(b.cls[0])
            if cid >= len(self.tipos):
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            tid = int(b.id[0]) if getattr(b, "id", None) is not None else None
            out.append((self.tipos[cid], (x1 + ox, y1 + oy, x2 + ox, y2 + oy), float(b.conf[0]), tid))
        return out

    def _dets_tiles(self, entrada, existentes):
        U = UMBRALES
        H, W = entrada.shape[:2]
        tw, th = int(W * 0.6), int(H * 0.6)
        nuevos = []
        for ox in (0, W - tw):
            for oy in (0, H - th):
                r = self.model_aux.predict(entrada[oy:oy + th, ox:ox + tw], conf=U["conf_min"],
                                           iou=0.45, imgsz=640, verbose=False)[0]
                for d in self._dets(r, ox, oy):
                    if d[0] in ("paciente", "mano_desnuda"):
                        continue
                    if any(d[0] == e[0] and iou(d[1], e[1]) > 0.5 for e in existentes + nuevos):
                        continue
                    nuevos.append(d)
        return nuevos

    def _personas_pose(self, entrada):
        U = UMBRALES
        out = []
        if self.pose is None:
            return out
        rp = self.pose.track(entrada, persist=True, conf=U["pose_conf"], imgsz=self.imgsz, verbose=False)[0]
        if rp.boxes is None or rp.keypoints is None or len(rp.boxes) == 0:
            return out
        xy = rp.keypoints.xy.cpu().numpy()
        kc = (rp.keypoints.conf.cpu().numpy() if rp.keypoints.conf is not None
              else np.ones(xy.shape[:2], dtype=float))
        for i, b in enumerate(rp.boxes):
            tid = int(b.id[0]) if getattr(b, "id", None) is not None else None
            out.append((tid, tuple(map(int, b.xyxy[0])), xy[i], kc[i]))
        return out

    def inferir(self, frame, usar_tiles=False, usar_seg=True):
        U = UMBRALES
        entrada = mejorar(frame, self.mejora)
        res = self.model.track(entrada, persist=True, conf=U["conf_min"], iou=0.45,
                               imgsz=self.imgsz, verbose=False)[0]
        dets = self._dets(res)
        if usar_tiles and self.model_aux is not None:
            dets += self._dets_tiles(entrada, dets)

        ev = dict.fromkeys(CAMPOS_EV, 0.0)
        dibujo = []
        cajas_prompt_seg = []

        def sube(k, v):
            ev[k] = max(ev[k], v)

        def rechazar(box, txt):
            dibujo.append(dict(box=box, texto=txt, color=(130, 130, 130), ok=False, pts=None, poly=None))

        def aceptar(box, txt, color, pts=None, segmentable=True):
            d = dict(box=box, texto=txt, color=color, ok=True, pts=pts, poly=None)
            dibujo.append(d)
            if segmentable and usar_seg and self.seg is not None:
                cajas_prompt_seg.append((box, d))

        dets.sort(key=lambda d: -d[2])
        pant_vistas = []
        ignorar = [d for d in dets if d[0] == "ignorar"]
        for g in ignorar:
            rechazar(g[1], f"ignorar {int(g[2] * 100)}%")

        pacientes = []
        for tipo, box, conf, tid in dets:
            pct = int(conf * 100)
            if tipo in ("ignorar", "mano_desnuda"):
                continue

            if tipo in PANTALLAS:
                hint = {"pantalla_carto": "carto", "pantalla_poli": "poligrafo"}.get(tipo)
                umbral = (U["conf_fluoro"] if tipo == "fluoro"
                          else U["conf_pantalla_hint"] if hint else U["conf_pantalla"])
                nombre = VOCABULARIO_INV[tipo]
                if any(iou(box, g[1]) > U["iou_absorcion"] and g[2] >= conf for g in ignorar):
                    rechazar(box, f"{nombre} {pct}% absorbida")
                    continue
                if conf < umbral:
                    rechazar(box, f"{nombre} {pct}% conf baja")
                    continue
                if not es_pantalla_valida(box, frame.shape):
                    rechazar(box, f"{nombre} {pct}% forma")
                    continue
                if any(iou(box, b) > 0.60 for b in pant_vistas):
                    continue
                pant_vistas.append(box)
                contenido, rojo, met = clasificar_pantalla(recortar(frame, box))
                m = (f" [det:{tipo[9:] or 'mon'} s{met['s']:.0f} b{met['b']:.2f} o{met['o']:.2f} "
                     f"hv{met['hv']:.1f} e{met['e']:.2f}]")
                final, fuerza = contenido, 0.4 + 0.6 * conf
                if final is None and hint:
                    final, fuerza = hint, conf
                if tid is not None:
                    vp = self.votos_pantalla[tid]
                    vp.append(final)
                    top = Counter(v for v in vp if v).most_common(1)
                    if top and top[0][1] >= 2 and top[0][1] >= 0.4 * len(vp) and final != top[0][0]:
                        final, fuerza = top[0][0], min(fuerza, conf)
                if tipo == "fluoro" and final != "fluoro":
                    rechazar(box, f"Fluoro {pct}% sin textura{m}")
                    continue
                if final == "fluoro" and conf < U["conf_fluoro"]:
                    final = None
                if final == "carto":
                    sube("navegador", fuerza)
                    sube("carto_rojo", rojo)
                    aceptar(box, f"Navegador 3D (CARTO) {pct}%{m}", (0, 255, 255))
                elif final == "poligrafo":
                    sube("poligrafo", fuerza)
                    aceptar(box, f"Poligrafo (EP) {pct}%{m}", (0, 200, 255))
                elif final == "fluoro":
                    sube("fluoro", fuerza)
                    aceptar(box, f"Fluoroscopia {pct}%{m}", (255, 200, 0))
                else:
                    rechazar(box, f"Monitor {pct}% contenido?{m}")

            elif tipo == "paciente":
                if not es_camilla_real(box, frame.shape):
                    rechazar(box, f"Paciente {pct}% forma/pos")
                elif conf < U["conf_paciente"]:
                    rechazar(box, f"Paciente {pct}% conf baja")
                else:
                    sube("paciente", conf)
                    pacientes.append(box)
                    aceptar(box, f"Paciente en camilla {pct}%", (80, 180, 220))

            elif tipo == "bata":
                if conf >= U["conf_bata"]:
                    sube("bata", conf)
                    aceptar(box, f"Bata esteril {pct}%", (50, 220, 50), segmentable=False)
                else:
                    rechazar(box, f"Bata {pct}% conf baja")

            elif tipo == "guantes":
                if conf >= U["conf_guantes"]:
                    sube("guantes", conf)
                    aceptar(box, f"Guantes {pct}%", (255, 255, 0), segmentable=False)
                else:
                    rechazar(box, f"Guantes {pct}% conf baja")

        self.pac_mem = max(ev["paciente"], self.pac_mem * self.decay)
        if self.pac_mem < 0.10:
            self.pac_mem = 0.0
        ev["paciente"] = self.pac_mem

        n_personas = 0
        for tid, box, kxy, kc in self._personas_pose(entrada):
            if any(iou(box, p) > 0.40 for p in pacientes):
                rechazar(box, "persona = paciente")
                continue
            n_personas += 1
            votos_brazo, pts, ratios = [], [], []
            for ie, iw in ((7, 9), (8, 10)):
                if kc[ie] < U["kp_conf"] or kc[iw] < U["kp_conf"]:
                    continue
                codo, mun = kxy[ie], kxy[iw]
                pts += [(float(codo[0]), float(codo[1])), (float(mun[0]), float(mun[1]))]
                r = analizar_antebrazo(entrada, codo, mun)
                if r is not None:
                    ratios.append(r)
                    votos_brazo.append("bare" if r > U["antebrazo_piel_alta"]
                                       else "cubierto" if r < U["antebrazo_piel_baja"] else None)

            voto = "bare" if "bare" in votos_brazo else "cubierto" if "cubierto" in votos_brazo else None
            decision = voto
            if tid is not None:
                if voto:
                    self.votos[tid].append(voto)
                v = self.votos[tid]
                decision = None
                if len(v) >= U["votos_min"]:
                    c, n = Counter(v).most_common(1)[0]
                    if n >= 0.6 * len(v):
                        decision = c
            etiq = f"id{tid}" if tid is not None else "id?"
            ptxt = f" piel{max(ratios):.2f}" if ratios else ""
            if decision == "bare":
                ev["brazos_al_aire"] = 1.0
                aceptar(box, f"Sin bata {etiq}{ptxt}", (0, 140, 255), pts)
            elif decision == "cubierto":
                sube("bata", 0.5)
                aceptar(box, f"Cirujano (bata) {etiq}{ptxt}", (50, 220, 50), pts)
            else:
                aceptar(box, f"Personal {etiq}{ptxt}", (200, 200, 200), pts)

        ev["n_personas"] = n_personas

        # ======================================================================
        # SEGMENTACION EXACTA DEL FOTOGRAMA ACTUAL
        # ======================================================================
        if usar_seg and self.seg is not None and len(cajas_prompt_seg) > 0:
            cajas_prompt = [c[0] for c in cajas_prompt_seg]
            try:
                if self.is_fastsam:
                    sam_res = self.seg(entrada, bboxes=cajas_prompt, retina_masks=False,
                                       imgsz=self.imgsz, conf=0.25, verbose=False)[0]
                else:
                    sam_res = self.seg(entrada, imgsz=self.imgsz, conf=0.25, verbose=False)[0]

                if sam_res.masks is not None and sam_res.masks.xy is not None:
                    poligonos = [np.array(p, dtype=np.float32) for p in sam_res.masks.xy if len(p) >= 3]
                    for poly in poligonos:
                        px1, py1 = np.min(poly, axis=0)
                        px2, py2 = np.max(poly, axis=0)
                        caja_poly = (int(px1), int(py1), int(px2), int(py2))
                        
                        mejor_d, max_iou = None, 0.25
                        for c_box, d_ref in cajas_prompt_seg:
                            v = iou(caja_poly, c_box)
                            if v > max_iou:
                                max_iou = v
                                mejor_d = d_ref
                        if mejor_d is not None and mejor_d["poly"] is None:
                            mejor_d["poly"] = poly
            except Exception:
                pass

        return ev, dibujo


VOCABULARIO_INV = {}
for _p, (_t, _e) in VOCABULARIO.items():
    VOCABULARIO_INV.setdefault(_t, _e)


# ==============================================================================
# VISUALIZADOR
# ==============================================================================
def dibujar_det(img, box, texto, color, esc, grosor):
    x1, y1, x2, y2 = [int(v * esc) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, grosor)
    (tw, th), _ = cv2.getTextSize(texto, FONT, 0.38, 1)
    ty = max(th + 6, y1)
    cv2.rectangle(img, (x1, ty - th - 6), (min(img.shape[1] - 1, x1 + tw + 8), ty), (15, 15, 15), -1)
    cv2.putText(img, texto, (x1 + 4, ty - 4), FONT, 0.38, color, 1, cv2.LINE_AA)


COLOR_VERDE = (0, 255, 0)
COLOR_COMPLETA = (70, 180, 70)
COLOR_AMARILLO = (0, 215, 255)
COLOR_GRIS = (120, 120, 120)

UMBRAL_FASE = 0.35
CONFIRM_S = [3.0, 3.0, 3.0, 3.0, 3.0, 12.0]


class GestorFases:
    def __init__(self, fps_analisis):
        self.fps = fps_analisis
        self.detectadas = set()
        self.actual = None
        self.max_idx = -1
        self.racha = [0] * len(FASES)

    def _permitidas(self):
        perm = [k >= self.max_idx for k in range(len(FASES))]
        for desde, hacia in RETROCESOS:
            if self.max_idx == desde:
                perm[hacia] = True
        return perm

    def actualizar(self, scores, corte=False):
        if corte:
            self.racha = [0] * len(FASES)
        s = np.where(self._permitidas(), scores, -1.0)
        k = int(np.argmax(s))
        cand = k if s[k] >= UMBRAL_FASE else None
        self.racha = [r + 1 if i == cand else 0 for i, r in enumerate(self.racha)]
        if cand is None or cand == self.actual:
            return
        salto = cand - self.max_idx
        req = CONFIRM_S[cand] * (2 if salto > 1 else 1)
        if self.racha[cand] >= req * self.fps:
            self.actual = cand
            self.detectadas.add(cand)
            self.max_idx = max(self.max_idx, cand)

    def obtener_estados(self):
        estados = []
        for i, nombre in enumerate(FASES):
            if i == self.actual:
                estados.append((nombre, "[ ACTIVA ]", COLOR_VERDE))
            elif i in self.detectadas:
                estados.append((nombre, "[ COMPLETADA ]", COLOR_COMPLETA))
            elif i < self.max_idx:
                estados.append((nombre, "[ PASADA NO DETECTADA ]", COLOR_AMARILLO))
            else:
                estados.append((nombre, "[ PENDIENTE ]", COLOR_GRIS))
        return estados


SEMIVIDA_S = dict(navegador=1.5, poligrafo=1.5, fluoro=1.5, paciente=1.5, bata=1.5, guantes=1.0, brazos_al_aire=1.0)
ON_T, OFF_T = 0.50, 0.25


class EvidenciaTemporal:
    def __init__(self, fps_analisis):
        self.dt = 1.0 / fps_analisis
        self.reset()

    def reset(self):
        self.p = dict.fromkeys(CAMPOS_EV, 0.0)
        self.m = dict.fromkeys(CAMPOS_EV, 0.0)
        self.on = dict.fromkeys(CAMPOS_EV, False)

    def actualizar(self, ev):
        out = {}
        for k in CAMPOS_EV:
            x = float(ev.get(k, 0.0))
            if k == "carto_rojo":
                a = 0.5 ** (self.dt / 1.5)
                self.p[k] = a * self.p[k] + (1 - a) * x
                out[k] = self.p[k] if self.on["navegador"] else 0.0
                continue
            a = 0.5 ** (self.dt / SEMIVIDA_S[k])
            self.p[k] = a * self.p[k] + (1 - a) * (1.0 if x > 0 else 0.0)
            if x > 0:
                self.m[k] = x if self.m[k] == 0 else 0.7 * self.m[k] + 0.3 * x
            if self.on[k] and self.p[k] < OFF_T:
                self.on[k] = False
                self.m[k] = 0.0
            elif not self.on[k] and self.p[k] > ON_T:
                self.on[k] = True
            out[k] = self.m[k] if self.on[k] else 0.0
        return out


def renderizar_panel(alto, ancho, t, ev_prom, estados, scores):
    panel = np.full((alto, ancho, 3), (33, 26, 22), dtype=np.uint8)
    cv2.putText(panel, "ELECTROFISIOLOGIA - AI", (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Tiempo: {int(t // 60):02d}:{int(t % 60):02d}", (20, 54), FONT, 0.42, (150, 150, 150), 1)
    cv2.line(panel, (20, 66), (ancho - 20, 66), (75, 65, 60), 1)

    cv2.putText(panel, "EVIDENCIAS (estables)", (20, 84), FONT, 0.42, (200, 200, 200), 1)
    y = 94
    for k in CAMPOS_EV:
        val = ev_prom.get(k, 0.0)
        if k == "carto_rojo":
            val = min(1.0, val / UMBRALES["rojo_ablacion"])
        cv2.putText(panel, k, (20, y + 9), FONT, 0.36, (220, 220, 220), 1)
        cv2.rectangle(panel, (140, y + 1), (ancho - 30, y + 9), (60, 55, 50), 1)
        cv2.rectangle(panel, (140, y + 1), (140 + int((ancho - 170) * min(1, val)), y + 9), (90, 200, 90), -1)
        y += 15

    cv2.putText(panel, "FLUJO QUIRURGICO:", (20, 236), FONT, 0.46, (200, 200, 200), 1)
    y = 262
    for i, (nombre, estado_str, color) in enumerate(estados):
        cv2.putText(panel, estado_str, (28, y), FONT, 0.42, color, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{scores[i]:.2f}", (ancho - 50, y), FONT, 0.36, (130, 130, 130), 1, cv2.LINE_AA)
        cv2.putText(panel, nombre, (28, y + 16), FONT, 0.37, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (16, y - 12), (20, y + 20), color, -1)
        y += 40

    cv2.line(panel, (20, alto - 46), (ancho - 20, alto - 46), (75, 65, 60), 1)
    cv2.putText(panel, "Verde: activa / completada", (20, alto - 29), FONT, 0.36, COLOR_VERDE, 1)
    cv2.putText(panel, "Amarillo: pasada no detectada", (20, alto - 12), FONT, 0.36, COLOR_AMARILLO, 1)
    return panel


# ==============================================================================
# PASADA 1: PIPELINE PRODUCTOR-CONSUMIDOR SINCRONIZADO
# ==============================================================================
def extraer(video, csv_path, fps_analisis, modelo, imgsz, mejora, ver=False, tiles=False, debug=False,
            usar_pose=True, modelo_pose="yolov8s-pose.pt", usar_seg=True, modelo_seg="yolov8s-seg.pt"):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duracion_frame_s = 1.0 / fps

    # Cola de frames a analizar por la IA
    cola_ia = queue.Queue(maxsize=1)
    # Cola de fotogramas sincronizados listos para mostrar en pantalla
    cola_display = queue.Queue(maxsize=2)
    
    stop_event = threading.Event()
    init_event = threading.Event()
    lock_conf = threading.Lock()

    conf = {
        "mejora": mejora,
        "tiles": tiles,
        "seg": usar_seg,
        "debug": debug,
    }
    columnas = ["frame", "t", "corte"] + CAMPOS_EV + ["n_personas"]

    # --------------------------------------------------------------------------
    # HILO DE INFERENCIA (Procesa cada frame antes de enviarlo al visor)
    # --------------------------------------------------------------------------
    def worker_ia():
        try:
            det = Detector(modelo, imgsz, mejora, tiles, usar_pose, modelo_pose,
                           usar_seg, modelo_seg, fps_analisis)
            gestor = GestorFases(fps_analisis)
            temporal = EvidenciaTemporal(fps_analisis)
            init_event.set()
        except Exception as e:
            print(f"[!] Error inicializando modelos: {e}")
            init_event.set()
            return

        prev_hist = None
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columnas)
            w.writeheader()

            while not stop_event.is_set():
                try:
                    item = cola_ia.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    cola_ia.task_done()
                    break

                frame_proc, idx_ia, t_ia = item

                hist = histograma_hsv(frame_proc)
                corte = int(prev_hist is not None and
                            cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < UMBRALES["corte_correlacion"])
                prev_hist = hist
                if corte:
                    det.reset()
                    temporal.reset()

                with lock_conf:
                    det.mejora = conf["mejora"]
                    usar_t = conf["tiles"]
                    usar_s = conf["seg"]

                ev, dib = det.inferir(frame_proc, usar_tiles=usar_t, usar_seg=usar_s)
                fila = {"frame": idx_ia, "t": round(t_ia, 3), "corte": corte, "n_personas": ev["n_personas"]}
                fila.update({k: round(float(ev[k]), 4) for k in CAMPOS_EV})
                w.writerow(fila)
                f.flush()

                ev_prom = temporal.actualizar(ev)
                scores = puntuar_fases({k: np.array([v]) for k, v in ev_prom.items()}, 1)[0]
                gestor.actualizar(scores, bool(corte))
                estados = gestor.obtener_estados()

                # El frame y su mascara exacta se entregan juntos a la cola del visor
                if ver:
                    cola_display.put((frame_proc, t_ia, dib, ev_prom, estados, scores))

                cola_ia.task_done()

    hilo_ia = threading.Thread(target=worker_ia, daemon=True)
    hilo_ia.start()
    init_event.wait()

    # --------------------------------------------------------------------------
    # RENDERIZADO VISUAL SINCRONIZADO
    # --------------------------------------------------------------------------
    def pintar(frame_vis, t, dibujo, ev_prom, estados, scores):
        ancho_v = 960
        alto_v = int(frame_vis.shape[0] * ancho_v / frame_vis.shape[1])
        esc = ancho_v / frame_vis.shape[1]
        img = cv2.resize(frame_vis, (ancho_v, alto_v), interpolation=cv2.INTER_AREA)
        alto_t = max(alto_v, 540)
        if alto_t > alto_v:
            img = cv2.copyMakeBorder(img, 0, alto_t - alto_v, 0, 0, cv2.BORDER_CONSTANT, value=(33, 26, 22))

        with lock_conf:
            modo_mejora = conf["mejora"]
            modo_tiles = conf["tiles"]
            modo_seg = conf["seg"]
            modo_debug = conf["debug"]

        # CAPA 1: Mascaras sobre el frame exacto
        if modo_seg:
            overlay = img.copy()
            hay_mascaras = False
            for d in dibujo:
                if d["ok"] and d.get("poly") is not None:
                    poly_scaled = (d["poly"] * esc).astype(np.int32)
                    if len(poly_scaled) >= 3:
                        cv2.fillPoly(overlay, [poly_scaled], d["color"])
                        cv2.polylines(img, [poly_scaled], isClosed=True, color=d["color"], thickness=1, lineType=cv2.LINE_AA)
                        hay_mascaras = True
            if hay_mascaras:
                cv2.addWeighted(overlay, 0.40, img, 0.60, 0, dst=img)

        # CAPA 2: Cajas y etiquetas
        for d in dibujo:
            if d["ok"]:
                dibujar_det(img, d["box"], d["texto"], d["color"], esc, 2)
            elif modo_debug:
                dibujar_det(img, d["box"], d["texto"], d["color"], esc, 1)
            if modo_debug and d.get("pts"):
                for px, py in d["pts"]:
                    cv2.circle(img, (int(px * esc), int(py * esc)), 3, (0, 255, 255), -1)

        cv2.putText(img, f"MEJORA:{modo_mejora}  DEBUG:{'on' if modo_debug else 'off'}  "
                         f"TILES:{'on' if modo_tiles else 'off'}  SEG:{'on' if modo_seg else 'off'}  [c][d][t][s] [espacio] [q]",
                    (8, alto_v - 8), FONT, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        panel = renderizar_panel(alto_t, 380, t, ev_prom, estados, scores)
        cv2.imshow("Monitor Quirurgico EP - Sincronizado", np.hstack((img, panel)))

    def tecla(k):
        if k == ord("q"):
            return True
        with lock_conf:
            if k == ord("c"):
                conf["mejora"] = {"off": "on", "on": "auto", "auto": "off"}[conf["mejora"]]
            elif k == ord("d"):
                conf["debug"] = not conf["debug"]
            elif k == ord("t"):
                conf["tiles"] = not conf["tiles"]
            elif k == ord("s"):
                conf["seg"] = not conf["seg"]
        return False

    # Bucle del reproductor principal
    idx = 0
    k = 0
    paso = max(1, int(round(fps / fps_analisis)))

    while True:
        t_inicio = time.perf_counter()
        ok, raw = cap.read()
        if not ok:
            break

        idx += 1
        # Solo procesamos fotogramas que coincidan con la frecuencia de analisis para no sobrecargar
        if idx % paso != 0 and ver:
            continue

        frame = limitar_lado(raw)
        t = idx / fps

        # Enviamos fotograma a la IA
        cola_ia.put((frame, idx, t), block=True)

        if ver:
            # Esperamos que el fotograma sea procesado para mostrarlo sincronizado
            frame_disp, t_disp, dib, ev_prom, estados, scores = cola_display.get()
            pintar(frame_disp, t_disp, dib, ev_prom, estados, scores)
            cola_display.task_done()

            dt = time.perf_counter() - t_inicio
            espera_ms = max(1, int(round((duracion_frame_s * paso - dt) * 1000)))
            k = cv2.waitKey(espera_ms) & 0xFF
            if k == ord(" "):
                while True:
                    k = cv2.waitKey(0) & 0xFF
                    if k == ord(" "):
                        break
                    if tecla(k):
                        idx = None
                        break
                    pintar(frame_disp, t_disp, dib, ev_prom, estados, scores)
                if idx is None:
                    break
            elif tecla(k):
                break

    # Cierre
    stop_event.set()
    cola_ia.put(None)
    hilo_ia.join()
    cap.release()
    if ver:
        cv2.destroyAllWindows()
    print(f"\n[+] Proceso completado -> Evidencias guardadas en {csv_path}")


# ==============================================================================
# PASADA 2: SEGMENTACION DE FASES (HMM / VITERBI)
# ==============================================================================
def cargar_csv(path):
    with open(path, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    if not filas:
        raise SystemExit("CSV vacio")
    return {k: np.array([float(r[k]) for r in filas]) for k in filas[0].keys()}


def suavizar(x, n):
    if n <= 1:
        return x
    pad = n // 2
    xp = np.pad(x, (pad, n - 1 - pad), mode="edge")
    return np.convolve(xp, np.ones(n) / n, mode="valid")


def puntuar_fases(d, n_suav):
    U = UMBRALES
    g = {k: suavizar(d[k], n_suav) for k in CAMPOS_EV}
    ep = np.maximum(g["navegador"], g["poligrafo"])
    pant = np.maximum(ep, g["fluoro"])
    pac = g["paciente"]
    est = np.maximum(g["bata"], g["guantes"])
    abl = ep * np.clip(g["carto_rojo"] / U["rojo_ablacion"], 0, 1)

    s = np.zeros((len(ep), 6))
    s[:, 0] = np.maximum(g["brazos_al_aire"], est) * (1 - pac)
    s[:, 1] = pac * (1 - est) * (1 - pant)
    s[:, 2] = pac * est * (1 - pant)
    s[:, 3] = pant * (1 - abl)
    s[:, 4] = np.maximum(abl, 0.15 * ep * pac)
    s[:, 5] = pac * est * (1 - pant)
    return s


RETROCESOS = [(4, 3)]


def viterbi(logp, cortes, pen_avance=-6.0, pen_atras=-8.0, max_salto=2):
    T, K = logp.shape
    dp = np.full((T, K), -np.inf)
    bp = np.zeros((T, K), dtype=int)
    for k in range(K):
        dp[0, k] = logp[0, k] + 0.5 * pen_avance * k
    for t in range(1, T):
        m = 0.3 if cortes[t] else 1.0
        for k in range(K):
            cands = [(dp[t - 1, k], k)]
            for n in range(1, max_salto + 1):
                if k - n >= 0:
                    cands.append((dp[t - 1, k - n] + m * pen_avance * n, k - n))
            for desde, hacia in RETROCESOS:
                if hacia == k:
                    cands.append((dp[t - 1, desde] + m * pen_atras, desde))
            score, prev = max(cands, key=lambda c: c[0])
            dp[t, k] = score + logp[t, k]
            bp[t, k] = prev
    path = [int(np.argmax(dp[-1]))]
    for t in range(T - 1, 0, -1):
        path.append(int(bp[t, path[-1]]))
    return np.array(path[::-1])


def segmentos_desde_path(ts, path):
    segs, ini = [], 0
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != path[ini]:
            fin_t = ts[i] if i < len(ts) else ts[-1]
            segs.append((int(path[ini]), float(ts[ini]), float(fin_t)))
            ini = i
    return segs


def fmt(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def limpiar_path(ts, path, min_s):
    path = path.copy()
    while True:
        segs = segmentos_desde_path(ts, path)
        cortos = [(i, s) for i, s in enumerate(segs) if len(segs) > 1 and s[2] - s[1] < min_s]
        if not cortos:
            return path
        i, (k, a, b) = min(cortos, key=lambda x: x[1][2] - x[1][1])
        destino = segs[i - 1][0] if i > 0 else segs[i + 1][0]
        path[(ts >= a) & (ts < b) if i < len(segs) - 1 else (ts >= a)] = destino


def segmentar(csv_path, out_path, fps_analisis):
    d = cargar_csv(csv_path)
    ts = d["t"]
    n_suav = max(1, int(round(UMBRALES["ventana_suavizado_s"] * fps_analisis)))
    s = puntuar_fases(d, n_suav)
    path = viterbi(np.log(s + 0.05), d["corte"])
    path = limpiar_path(ts, path, UMBRALES["min_segmento_s"])
    segs = segmentos_desde_path(ts, path)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fase", "nombre", "inicio_s", "fin_s"])
        for k, a, b in segs:
            w.writerow([k + 1, FASES[k], round(a, 2), round(b, 2)])

    print("\n=== SEGMENTACION DE FASES ===")
    for k, a, b in segs:
        print(f"  {fmt(a)} - {fmt(b)}  {FASES[k]}")
    for k in range(len(FASES)):
        if k not in {k_ for k_, _, _ in segs}:
            print(f"  [no detectada] {FASES[k]}")
    print(f"[+] Segmentos -> {out_path}")
    return ts, path


# ==============================================================================
# VALIDACION Y RENDER
# ==============================================================================
def evaluar(gt_path, ts, path):
    gt = np.full(len(ts), -1)
    with open(gt_path, encoding="utf-8") as f:
        filas = [(int(r["fase"]) - 1, float(r["inicio_s"]), float(r["fin_s"])) for r in csv.DictReader(f)]
    for k, a, b in filas:
        gt[(ts >= a) & (ts < b)] = k
    ok = gt >= 0
    if not ok.any():
        print("[!] Ground truth sin solape con el video")
        return
    K = len(FASES)
    conf = np.zeros((K, K), dtype=int)
    for g_, p_ in zip(gt[ok], path[ok]):
        conf[g_, p_] += 1
    print(f"\n=== VALIDACION === precision por muestra: {100 * np.mean(gt[ok] == path[ok]):.1f}%")
    print("matriz de confusion (filas = real, columnas = predicho):")
    print(conf)
    for k, a, _ in filas:
        pred = ts[path == k]
        err = f"{pred[0] - a:+.1f} s" if len(pred) else "fase no detectada"
        print(f"  inicio fase {k + 1}: real {fmt(a)} | error {err}")


def render(video, ts, path, out_path, ancho=960):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    alto = int(h0 * ancho / w0)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))
    dur = ts[-1] if len(ts) else 1.0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        j = max(0, int(np.searchsorted(ts, t, side="right")) - 1)
        k = int(path[min(len(path) - 1, j)])
        frame = cv2.resize(frame, (ancho, alto))
        cv2.rectangle(frame, (0, 0), (ancho, 34), (20, 24, 30), -1)
        cv2.putText(frame, f"{fmt(t)}  {FASES[k]}", (10, 23), FONT, 0.55, COLOR_FASE[k], 1, cv2.LINE_AA)
        for i in range(len(ts) - 1):
            x1, x2 = int(ancho * ts[i] / dur), int(ancho * ts[i + 1] / dur)
            cv2.rectangle(frame, (x1, alto - 14), (max(x2, x1 + 1), alto), COLOR_FASE[int(path[i])], -1)
        xc = int(ancho * min(t, dur) / dur)
        cv2.line(frame, (xc, alto - 18), (xc, alto), (255, 255, 255), 2)
        out.write(frame)
        idx += 1
    cap.release()
    out.release()
    print(f"[+] Video con timeline -> {out_path}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="Hitos de electrofisiologia con segmentacion sincronizada")
    ap.add_argument("--video", required=True)
    ap.add_argument("--csv", help="CSV de evidencias (por defecto <video>_evidencias.csv)")
    ap.add_argument("--fps", type=float, default=6.0, help="frecuencia de analisis/visualizacion")
    ap.add_argument("--modelo", default="yolov8m-worldv2.pt")
    ap.add_argument("--imgsz", type=int, default=640, help="tamano de inferencia optimizado (ej. 640 u 800)")
    ap.add_argument("--mejora", choices=["off", "on", "auto"], default="auto")
    ap.add_argument("--no-pose", action="store_true", help="desactivar pose")
    ap.add_argument("--modelo-pose", default="yolov8s-pose.pt")
    ap.add_argument("--no-seg", action="store_true", help="desactivar segmentacion")
    ap.add_argument("--modelo-seg", default="yolov8s-seg.pt",
                    help="modelo de segmentacion (yolov8s-seg.pt, FastSAM-s.pt, etc.)")
    ap.add_argument("--ver", action="store_true", help="mostrar visor sincronizado")
    ap.add_argument("--debug", action="store_true", help="mostrar metricas de depuracion")
    ap.add_argument("--tiles", action="store_true", help="tiles activados")
    ap.add_argument("--solo-segmentar", action="store_true")
    ap.add_argument("--gt", help="CSV de ground truth")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    base = os.path.splitext(args.video)[0]
    csv_ev = args.csv or f"{base}_evidencias.csv"
    csv_seg = f"{base}_segmentos.csv"

    if not args.solo_segmentar:
        extraer(args.video, csv_ev, args.fps, args.modelo, args.imgsz, args.mejora,
                ver=args.ver, tiles=args.tiles, debug=args.debug,
                usar_pose=not args.no_pose, modelo_pose=args.modelo_pose,
                usar_seg=not args.no_seg, modelo_seg=args.modelo_seg)
    ts, path = segmentar(csv_ev, csv_seg, args.fps)
    if args.gt:
        evaluar(args.gt, ts, path)
    if args.render:
        render(args.video, ts, path, f"{base}_fases.mp4")


if __name__ == "__main__":
    main()