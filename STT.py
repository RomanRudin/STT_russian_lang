import os
import argparse
import tkinter as tk
from tkinter import filedialog
import soundfile as sf
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from models.pretrained_model import RuBertPunctuator, build_tokenizer
from utils.inference import PunctuationRestorer
from utils.stt_pipeline import STTPunctuationPipeline

# =====================================================================
# ГЛОБАЛЬНЫЕ НАСТРОЙКИ
# =====================================================================
WHISPER_MODEL_PATH = "./final_model"  # Путь к папке с твоим Whisper
RUBERT_WEIGHTS_PATH = "./best_rubert_base.pt"     # Путь к весам ruBERT
RUBERT_VARIANT = "base"                         # Вариант ruBERT ("tiny" или "base")

USE_LORA_WHISPER = False
WHISPER_BASE_MODEL = "openai/whisper-small"     
# =====================================================================


class AudioTextPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Инициализация моделей на {self.device}...")

        # 1. Инициализация Whisper
        self.whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_PATH)
        
        if USE_LORA_WHISPER:
            from peft import PeftModel
            base_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_BASE_MODEL).to(self.device)
            self.whisper = PeftModel.from_pretrained(base_model, WHISPER_MODEL_PATH).to(self.device)
        else:
            self.whisper = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_PATH).to(self.device)
        
        self.whisper.eval()

        # 2. Инициализация ruBERT
        self.rubert = RuBertPunctuator(variant=RUBERT_VARIANT).to(self.device)
        self.rubert_tokenizer = build_tokenizer(variant=RUBERT_VARIANT)
        
        if os.path.exists(RUBERT_WEIGHTS_PATH):
            self.rubert.load_state_dict(torch.load(RUBERT_WEIGHTS_PATH, map_location=self.device))
        else:
            print(f"ВНИМАНИЕ: Файл весов ruBERT '{RUBERT_WEIGHTS_PATH}' не найден!")
            
        self.rubert.eval()

        # 3. Сборка пайплайна пунктуации
        self.restorer = PunctuationRestorer(
            model=self.rubert,
            backend="rubert",
            tokenizer=self.rubert_tokenizer,
            device=self.device
        )
        self.punct_pipeline = STTPunctuationPipeline(self.restorer)

        print("Модели успешно загружены!")

    def process_audio(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=16000)
            duration = librosa.get_duration(y=audio, sr=sr)
        except Exception as e:
            return f"❌ Ошибка чтения аудио: {e}", ""

        if duration < 0.5 or duration > 30.0:
            return f"⚠️ Ошибка: Длительность файла ({duration:.1f} сек) вне диапазона 0.5 - 30.0 сек.", ""

        try:
            inputs = self.whisper_processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
            with torch.no_grad():
                predicted_ids = self.whisper.generate(inputs.input_features, language="ru")
            
            transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

            if not transcription:
                return "Whisper не распознал речь (пустой результат).", ""

            # Прогоняем сырой текст через модель пунктуации
            final_text = self.punct_pipeline(transcription)
            
            return transcription, final_text

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return "💥 КРИТИЧЕСКАЯ ОШИБКА: Видеокарте не хватило памяти!", ""
            return f"❌ Ошибка при генерации: {e}", ""


def run_gui(pipeline):
    root = tk.Tk()
    root.title("Audio to Text Pipeline (Whisper + ruBERT)")
    root.geometry("750x600")

    def select_file():
        file_path = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        text_box.delete("1.0", tk.END)
        text_box.insert(tk.END, "Обработка файла, пожалуйста, подождите...\n")
        root.update() 

        whisper_res, rubert_res = pipeline.process_audio(file_path)
        
        text_box.delete("1.0", tk.END)
        
        # Если вернулась ошибка (второе значение пустое)
        if not rubert_res:
            text_box.insert(tk.END, whisper_res)
        else:
            text_box.insert(tk.END, "=== 1. ВЫВОД WHISPER (Сырой текст) ===\n", "header")
            text_box.insert(tk.END, f"{whisper_res}\n\n")
            text_box.insert(tk.END, "=== 2. ВЫВОД ruBERT (Пунктуация и регистр) ===\n", "header")
            text_box.insert(tk.END, f"{rubert_res}\n")

    btn_open = tk.Button(root, text="Выбрать аудиофайл", command=select_file, font=("Arial", 11, "bold"), padx=10, pady=5)
    btn_open.pack(pady=15)

    text_box = tk.Text(root, wrap=tk.WORD, font=("Arial", 12), padx=12, pady=12)
    # Выделяем заголовки жирным шрифтом в GUI
    text_box.tag_configure("header", font=("Arial", 12, "bold"))
    text_box.pack(expand=True, fill=tk.BOTH, padx=20, pady=(0, 20))

    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Путь к аудиофайлу для CLI")
    args = parser.parse_args()

    pipe = AudioTextPipeline()

    if args.file:
        if not os.path.exists(args.file):
            print(f"Файл не найден: {args.file}")
        else:
            whisper_res, rubert_res = pipe.process_audio(args.file)
            
            if not rubert_res:
                print(whisper_res)
            else:
                print("\n=== 1. ВЫВОД WHISPER (Сырой текст) ===")
                print(whisper_res)
                print("\n=== 2. ВЫВОД ruBERT (Пунктуация и регистр) ===")
                print(rubert_res)
                print("==============================================\n")
    else:
        run_gui(pipe)