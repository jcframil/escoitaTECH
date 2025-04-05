import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import utils.utils
from utils.utils import check_folder

#Mostrar los MFCCS en el espectrograma
class Plots:

    # Dividir en ventanas
    frame_lenght = 2048
    hop_length = 512

    # Umbral para distinguir alta y baja energía
    # 50% de la máxima energía
    threshold = 0.5

    def spectrogram(self, mfcc, sr, name_file):
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(mfcc, x_axis="time", y_axis="mel", sr=sr, cmap="viridis")
        plt.colorbar(label="MFCC Coefficients")
        plt.title("MFCCs " + name_file.upper())
        plt.tight_layout()
        plt.grid()
        path = utils.utils.check_folder(name_file)
        plt.savefig(path / f"{name_file}_spectogram.jpg", dpi=300)
        plt.show()

    def calculate_energy(self, audio, sr, name_file):
        #Calcular energía
        energy=np.array([np.sum(np.abs(audio[i:i+self.frame_lenght])**2)
                            for i in range(0,len(audio),self.hop_length)
                     ])

        # Normalizar energía
        energy_normalized = energy / np.max(energy)
        return energy_normalized

    def plot_energy(self, audio, sr, name_file):

        plt.figure(figsize=(12,4))
        plt.plot(self.calculate_energy( audio, sr, name_file), label="Energía normalizada")
        plt.axhline(y=self.threshold, color="red", linestyle="--", label="Umbral")
        plt.legend()
        plt.xlabel("Frames")
        plt.ylabel("Energía Normalizada")
        plt.title(f"Detección de Alta y Baja Energía {name_file}")
        plt.grid()
        path = utils.utils.check_folder(name_file)
        plt.savefig(path / f"{name_file}_energy.jpg", dpi=300)
        plt.show()

    def plot_high_low_energy(self, audio, sr, name_file):

        energy_normalized = self.calculate_energy(audio, sr, name_file)

        #Alta energía
        plt.figure(figsize=(12,4))
        plt.plot(energy_normalized, label="Energía (normalizada)")
        plt.axhline(y=self.threshold, color="red", linestyle="--", label="Umbral")
        plt.fill_between(range(len(energy_normalized)),0, energy_normalized, where=energy_normalized>self.threshold, color="red", alpha=0.5,label="Alta energía")
        plt.legend()
        plt.xlabel("Frames (Ventanas de tiempo)")
        plt.ylabel("Energía Normalizada")
        plt.title(f"Análisis de Energía: Alta Energía {name_file}")
        plt.grid()
        path = utils.utils.check_folder(name_file)
        plt.savefig(path / f"{name_file}_high_energy.jpg", dpi=300)
        plt.show()

        #Baja energía
        plt.figure(figsize=(12,4))
        plt.plot(energy_normalized, label="Energía (normalizada)")
        plt.axhline(y=self.threshold, color="red", linestyle="--", label="Umbral")
        plt.fill_between(range(len(energy_normalized)),0, energy_normalized, where=energy_normalized<self.threshold, color="orange", alpha=0.5, label="Baja energía")
        plt.legend()
        plt.xlabel("Frames (Ventanas de tiempo)")
        plt.ylabel("Energía Normalizada")
        plt.title(f"Análisis de Energía: Baja Energía {name_file}")
        plt.grid()
        path = utils.utils.check_folder(name_file)
        plt.savefig(path / f"{name_file}_low_energy.jpg", dpi=300)
        plt.show()