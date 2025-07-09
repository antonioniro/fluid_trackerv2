import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os

class FluidTracker:
    def __init__(self, video_path, output_path=None):
        self.cap = cv2.VideoCapture(video_path)
        self.output_path = output_path
        self.surface_history = deque(maxlen=50)  # Memorizza ultime 50 superfici
        self.motion_vectors = []
        
        # Parametri per il rilevamento della superficie
        self.canny_low = 50
        self.canny_high = 150
        self.roi_top = 0.2  # ROI per la superficie (20% dall'alto)
        self.roi_bottom = 0.8  # fino all'80% dell'immagine
        
        # Parametri per optical flow
        self.lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Parametri per il rilevamento di feature
        self.feature_params = dict(maxCorners=200,  # Aumentato da 100 a 200
                                  qualityLevel=0.2,  # Ridotto da 0.3 a 0.2 per più punti
                                  minDistance=5,    # Ridotto da 7 a 5 per più punti
                                  blockSize=7)
        
        # Parametri per la selezione manuale
        self.selected_roi = None
        self.roi_points = []
        self.selecting = False
        self.roi_selected = False
        
        # Parametri per la distribuzione dei punti
        self.min_points_distance = 10  # Distanza minima tra i punti
        self.target_points = 100       # Numero target di punti da tracciare
    
    def select_roi(self, frame):
        """Permette all'utente di selezionare manualmente la regione di interesse"""
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.roi_points.append((x, y))
                self.selecting = True
            elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
                temp_frame = frame.copy()
                if len(self.roi_points) > 0:
                    cv2.line(temp_frame, self.roi_points[-1], (x, y), (0, 255, 0), 2)
                cv2.imshow('Select ROI', temp_frame)
            elif event == cv2.EVENT_LBUTTONUP:
                self.selecting = False
                if len(self.roi_points) > 0:
                    self.roi_points.append((x, y))
                    # Calcola il ROI rettangolare
                    x_coords = [p[0] for p in self.roi_points]
                    y_coords = [p[1] for p in self.roi_points]
                    self.selected_roi = (
                        min(x_coords), min(y_coords),
                        max(x_coords) - min(x_coords),
                        max(y_coords) - min(y_coords)
                    )
                    self.roi_selected = True
                    cv2.destroyWindow('Select ROI')

        cv2.namedWindow('Select ROI')
        cv2.setMouseCallback('Select ROI', mouse_callback)
        
        print("Seleziona la regione di interesse:")
        print("1. Clicca e trascina per disegnare un rettangolo")
        print("2. Premi 'r' per ricominciare")
        print("3. Premi 'c' per confermare la selezione")
        
        while True:
            temp_frame = frame.copy()
            if len(self.roi_points) > 0:
                cv2.rectangle(temp_frame, 
                            (self.roi_points[0][0], self.roi_points[0][1]),
                            (self.roi_points[-1][0], self.roi_points[-1][1]),
                            (0, 255, 0), 2)
            cv2.imshow('Select ROI', temp_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):  # Reset
                self.roi_points = []
                self.selected_roi = None
                self.roi_selected = False
            elif key == ord('c'):  # Confirm
                if self.roi_selected:
                    break
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return True

    def distribute_points_along_surface(self, surface_points, width):
        """Distribuisce uniformemente i punti lungo la superficie"""
        if not surface_points:
            return []
        
        # Converti in array numpy per elaborazione
        points = np.array(surface_points)
        
        # Ordina i punti per x
        points = points[points[:, 0].argsort()]
        
        # Calcola la distanza totale lungo la superficie
        total_distance = 0
        distances = []
        for i in range(1, len(points)):
            dist = np.sqrt(np.sum((points[i] - points[i-1])**2))
            total_distance += dist
            distances.append(dist)
        
        # Calcola il numero di punti da mantenere
        num_points = min(self.target_points, len(points))
        target_distance = total_distance / (num_points - 1)
        
        # Seleziona i punti uniformemente distribuiti
        selected_points = [points[0]]
        current_distance = 0
        
        for i in range(1, len(points)):
            current_distance += distances[i-1]
            if current_distance >= target_distance:
                selected_points.append(points[i])
                current_distance = 0
        
        # Assicurati di includere l'ultimo punto
        if len(selected_points) > 0 and not np.array_equal(selected_points[-1], points[-1]):
            selected_points.append(points[-1])
        
        return selected_points

    def detect_surface_line(self, frame):
        """Rileva la linea di superficie del fluido"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        if self.roi_selected and self.selected_roi:
            # Usa la regione selezionata manualmente
            x, y, w_roi, h_roi = self.selected_roi
            roi = gray[y:y+h_roi, x:x+w_roi]
        else:
            # Usa la ROI automatica
            roi_y1 = int(h * self.roi_top)
            roi_y2 = int(h * self.roi_bottom)
            roi = gray[roi_y1:roi_y2, :]
            x, y = 0, roi_y1
        
        # Applica filtro Gaussian per ridurre il rumore
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        
        # Applica threshold adattivo per migliorare il contrasto
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Rilevamento bordi con Canny
        edges = cv2.Canny(thresh, self.canny_low, self.canny_high)
        
        # Trova contorni
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        surface_points = []
        if contours:
            # Trova il contorno più lungo (probabilmente la superficie)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Estrai punti della superficie
            for point in largest_contour:
                px, py = point[0]
                surface_points.append((px + x, py + y))  # Aggiungi offset ROI
            
            # Distribuisci i punti uniformemente lungo la superficie
            surface_points = self.distribute_points_along_surface(surface_points, w)
        
        return surface_points, edges

    def track_motion(self, prev_gray, curr_gray, prev_points):
        """Traccia il movimento usando optical flow"""
        if len(prev_points) == 0:
            return [], []
        
        # Calcola optical flow
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None, **self.lk_params)
        
        # Seleziona punti validi
        good_new = next_points[status == 1]
        good_old = prev_points[status == 1]
        
        return good_new, good_old
    
    def visualize_surface_and_motion(self, frame, surface_points, motion_vectors, frame_num):
        """Visualizza superficie e movimento"""
        vis_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Disegna la superficie
        if surface_points:
            # Converti in array numpy per elaborazione
            points = np.array(surface_points, dtype=np.int32)
            
            # Disegna linea di superficie
            if len(points) > 1:
                cv2.polylines(vis_frame, [points], False, (0, 255, 0), 2)
            
            # Evidenzia punti della superficie
            for point in points:
                cv2.circle(vis_frame, tuple(point), 2, (0, 255, 255), -1)  # Ridotto il raggio da 3 a 2
        
        # Disegna vettori di movimento
        for i, (new, old) in enumerate(motion_vectors):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            
            # Disegna linea di movimento
            cv2.line(vis_frame, (a, b), (c, d), (255, 0, 0), 1)  # Ridotto lo spessore da 2 a 1
            # Disegna punto corrente
            cv2.circle(vis_frame, (a, b), 2, (0, 0, 255), -1)  # Ridotto il raggio da 3 a 2
        
        # Aggiungi informazioni di testo
        cv2.putText(vis_frame, f'Frame: {frame_num}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, f'Surface Points: {len(surface_points)}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, f'Motion Vectors: {len(motion_vectors)}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame
    
    def analyze_wave_motion(self):
        """Analizza il movimento delle onde nel video"""
        if not self.cap.isOpened():
            print("Errore: impossibile aprire il video")
            return
        
        # Leggi primo frame
        ret, frame = self.cap.read()
        if not ret:
            print("Errore: impossibile leggere il primo frame")
            return
        
        # Permetti la selezione manuale della ROI
        if not self.select_roi(frame):
            print("Selezione ROI annullata")
            return
        
        # Inizializza variabili
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_num = 0
        
        # Setup per salvare il video se richiesto
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))
        
        # Rileva superficie iniziale
        surface_points, _ = self.detect_surface_line(frame)
        prev_points = None
        
        if surface_points:
            # Converti i punti della superficie in formato per optical flow
            prev_points = np.array(surface_points, dtype=np.float32).reshape(-1, 1, 2)
        
        print("Inizio analisi del video...")
        print("Premi 'q' per uscire, 's' per salvare il frame corrente")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_num += 1
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Rileva superficie
            surface_points, edges = self.detect_surface_line(frame)
            self.surface_history.append(surface_points)
            
            # Traccia movimento
            motion_vectors = []
            if prev_points is not None and len(prev_points) > 0:
                good_new, good_old = self.track_motion(prev_gray, curr_gray, prev_points)
                motion_vectors = list(zip(good_new, good_old))
                
                # Aggiorna punti per il prossimo frame
                if len(good_new) > 0:
                    prev_points = good_new.reshape(-1, 1, 2)
                else:
                    # Se perdiamo i punti, rileva nuovi punti dalla superficie
                    if surface_points:
                        prev_points = np.array(surface_points, dtype=np.float32).reshape(-1, 1, 2)
                    else:
                        prev_points = None
            
            # Visualizza risultati
            vis_frame = self.visualize_surface_and_motion(frame, surface_points, motion_vectors, frame_num)
            
            # Mostra frame con analisi
            cv2.imshow('Fluid Motion Analysis', vis_frame)
            cv2.imshow('Edge Detection', edges)
            
            # Salva video se richiesto
            if self.output_path:
                out.write(vis_frame)
            
            # Controlli tastiera
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'frame_{frame_num}.jpg', vis_frame)
                print(f"Frame {frame_num} salvato")
            elif key == ord('p'):  # Pausa
                cv2.waitKey(0)
            
            # Aggiorna per prossimo frame
            prev_gray = curr_gray.copy()
            
            # Mostra progresso ogni 30 frames
            if frame_num % 30 == 0:
                print(f"Processato frame {frame_num}")
        
        # Cleanup
        self.cap.release()
        if self.output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Analisi completata. Processati {frame_num} frames.")
    
    def plot_surface_analysis(self):
        """Crea grafici di analisi della superficie"""
        if not self.surface_history:
            print("Nessun dato della superficie disponibile")
            return
        
        # Analizza variazioni della superficie nel tempo
        surface_heights = []
        for surfaces in self.surface_history:
            if surfaces:
                heights = [point[1] for point in surfaces]
                avg_height = np.mean(heights)
                surface_heights.append(avg_height)
            else:
                surface_heights.append(0)
        
        # Crea grafici
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(surface_heights)
        plt.title('Altezza Media della Superficie nel Tempo')
        plt.xlabel('Frame')
        plt.ylabel('Altezza (pixel)')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        if len(surface_heights) > 1:
            surface_diff = np.diff(surface_heights)
            plt.plot(surface_diff)
            plt.title('Variazione dell\'Altezza della Superficie')
            plt.xlabel('Frame')
            plt.ylabel('Differenza (pixel)')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Esempio di utilizzo
def main():
    # Sostituisci con il percorso del tuo video
    video_path = "colorante.mp4"  # Cambia questo percorso
    output_path = "analisi_fluido_output.mp4"  # Opzionale: per salvare il video analizzato
    
    if not os.path.exists(video_path):
        print(f"ATTENZIONE: Il file {video_path} non esiste.")
        print("Modifica il percorso del video nella funzione main()")
        return
    
    # Crea tracker
    tracker = FluidTracker(video_path, output_path)
    
    # Esegui analisi
    tracker.analyze_wave_motion()
    
    # Mostra grafici di analisi
    tracker.plot_surface_analysis()

if __name__ == "__main__":
    main()