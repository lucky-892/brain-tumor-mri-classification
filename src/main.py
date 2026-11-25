import tkinter as tk
from tkinter import filedialog, messagebox
import os
import threading

from core_utils import load_model, get_features, class_names
from module_gradcam import run_gradcam_single
from module_clustering import run_kmeans, run_agglomerative, run_gmm, run_dbscan_forced, run_dbscan_auto

MODEL_PATH = "resnet18_brain_tumor_model.pth"

class BrainTumorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ИС Анализа МРТ")
        self.root.geometry("600x700")
        self.root.configure(bg="#f0f2f5")
        
        self.model = None
        self.features = None
        self.paths = None
        self.selected_directory = ""
        self.image_files_map = {} # Отображаемое имя -> полный путь

        self.setup_ui()
        self.status_var.set("Загрузка модели ResNet18...")
        threading.Thread(target=self.init_model, daemon=True).start()

    def init_model(self):
        try:
            if not os.path.exists(MODEL_PATH):
                self.status_var.set("❌ Ошибка: модель не найдена!")
                return
            self.model = load_model(MODEL_PATH)
            self.status_var.set("✅ Модель загружена. Готов к работе!")
            self.btn_select_dir.config(state="normal")
        except Exception as e:
            self.status_var.set(f"❌ Ошибка загрузки: {e}")

    def setup_ui(self):
        title = tk.Label(self.root, text="Анализ опухолей головного мозга", font=("Helvetica", 16, "bold"), bg="#f0f2f5")
        title.pack(pady=10)

        # --- Секция: Выбор данных ---
        frame_data = tk.LabelFrame(self.root, text=" Выбор данных ", bg="#f0f2f5", font=("Helvetica", 10, "bold"), padx=10, pady=10)
        frame_data.pack(fill="x", padx=20, pady=5)
        
        self.btn_select_dir = tk.Button(frame_data, text="Выбрать папку (Например, 'testing')", command=self.select_directory, state="disabled", bg="#FF9800", fg="white")
        self.btn_select_dir.pack(fill="x", pady=5)

        self.lbl_dir = tk.Label(frame_data, text="Папка не выбрана", bg="#f0f2f5", fg="#555")
        self.lbl_dir.pack()

        # Список файлов
        list_frame = tk.Frame(frame_data)
        list_frame.pack(fill="both", expand=True, pady=5)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        self.listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=6)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

        # --- Секция: Grad-CAM ---
        frame1 = tk.LabelFrame(self.root, text=" 1. Анализ выбранного снимка (Grad-CAM) ", bg="#f0f2f5", font=("Helvetica", 10, "bold"), padx=10, pady=10)
        frame1.pack(fill="x", padx=20, pady=5)
        
        self.btn_gradcam = tk.Button(frame1, text="Показать Grad-CAM для выбранного файла", command=self.launch_gradcam, state="disabled", bg="#4CAF50", fg="white", height=2)
        self.btn_gradcam.pack(fill="x")

        # --- Секция: Кластеризация ---
        frame2 = tk.LabelFrame(self.root, text=" 2. Кластеризация данных всей папки ", bg="#f0f2f5", font=("Helvetica", 10, "bold"), padx=10, pady=10)
        frame2.pack(fill="x", padx=20, pady=5)

        self.btn_load_feats = tk.Button(frame2, text="[Шаг 1] Извлечь признаки всей папки", command=self.prepare_features, state="disabled", bg="#2196F3", fg="white")
        self.btn_load_feats.pack(fill="x", pady=(0, 10))

        self.btn_kmeans = tk.Button(frame2, text="K-Means (Принудительно 4 кластера)", command=lambda: self.run_clustering("kmeans"), state="disabled")
        self.btn_kmeans.pack(fill="x", pady=2)
        
        self.btn_agg = tk.Button(frame2, text="Agglomerative (Принудительно 4 кластера)", command=lambda: self.run_clustering("agg"), state="disabled")
        self.btn_agg.pack(fill="x", pady=2)
        
        self.btn_gmm = tk.Button(frame2, text="GMM (Принудительно 4 кластера)", command=lambda: self.run_clustering("gmm"), state="disabled")
        self.btn_gmm.pack(fill="x", pady=2)
        
        self.btn_dbscan_forced = tk.Button(frame2, text="DBSCAN (Обертка: Принудительный поиск 4 кластеров)", command=lambda: self.run_clustering("dbscan_forced"), state="disabled", bg="#e8f5e9")
        self.btn_dbscan_forced.pack(fill="x", pady=2)

        self.btn_dbscan_auto = tk.Button(frame2, text="DBSCAN (Определяет сам естественное кол-во)", command=lambda: self.run_clustering("dbscan_auto"), state="disabled", fg="purple", bg="#f3e5f5")
        self.btn_dbscan_auto.pack(fill="x", pady=2)

        self.status_var = tk.StringVar()
        self.status_var.set("Инициализация...")
        status_label = tk.Label(self.root, textvariable=self.status_var, bg="#f0f2f5", fg="#555")
        status_label.pack(side="bottom", pady=10)

    def select_directory(self):
        folder_path = filedialog.askdirectory(title="Выберите папку с классами (testing/training)")
        if not folder_path: return
        
        self.selected_directory = folder_path
        self.lbl_dir.config(text=f"Выбрано: {os.path.basename(folder_path)}")
        self.listbox.delete(0, tk.END)
        self.image_files_map.clear()
        
        # Сканируем папку
        for cls in class_names:
            cls_path = os.path.join(folder_path, cls)
            if os.path.isdir(cls_path):
                for fname in os.listdir(cls_path):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        display_name = f"{cls} / {fname}"
                        full_path = os.path.join(cls_path, fname)
                        self.image_files_map[display_name] = (full_path, cls)
                        self.listbox.insert(tk.END, display_name)
        
        if self.image_files_map:
            self.status_var.set(f"Загружено {len(self.image_files_map)} файлов. Выберите файл в списке.")
            self.btn_gradcam.config(state="normal")
            self.btn_load_feats.config(state="normal")
        else:
            self.status_var.set("❌ В выбранной папке не найдены нужные подпапки (glioma, meningioma и т.д.)")

    def launch_gradcam(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("Внимание", "Сначала выберите файл из списка!")
            return
        
        display_name = self.listbox.get(selection[0])
        full_path, true_class = self.image_files_map[display_name]
        
        self.status_var.set("Запуск дашборда Grad-CAM...")
        run_gradcam_single(self.model, full_path, true_class)
        self.status_var.set("Готово!")

    def prepare_features(self):
        self.btn_load_feats.config(state="disabled", text="Извлечение признаков... Ждите")
        self.status_var.set("Нейросеть обрабатывает весь датасет...")
        self.root.update()
        
        def worker():
            self.features, self.paths = get_features(self.model, self.selected_directory)
            self.root.after(0, self.on_features_loaded)
            
        threading.Thread(target=worker, daemon=True).start()

    def on_features_loaded(self):
        self.btn_load_feats.config(text="✅ Данные готовы (Кэшировано)")
        self.status_var.set("Данные готовы к кластеризации!")
        self.btn_kmeans.config(state="normal")
        self.btn_agg.config(state="normal")
        self.btn_gmm.config(state="normal")
        self.btn_dbscan_forced.config(state="normal")
        self.btn_dbscan_auto.config(state="normal")

    def run_clustering(self, method):
        if self.features is None: return
        self.status_var.set(f"Выполняется кластеризация...")
        self.root.update()
        
        try:
            if method == "kmeans": run_kmeans(self.features, self.paths)
            elif method == "agg": run_agglomerative(self.features, self.paths)
            elif method == "gmm": run_gmm(self.features, self.paths)
            elif method == "dbscan_forced": run_dbscan_forced(self.features, self.paths, target_clusters=4) # Жестко задаем 4
            elif method == "dbscan_auto": run_dbscan_auto(self.features, self.paths) # Ничего не задаем
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при кластеризации:\n{e}")
            
        self.status_var.set("✅ Готово! Закройте графики для продолжения.")

if __name__ == "__main__":
    root = tk.Tk()
    app = BrainTumorApp(root)
    root.mainloop()