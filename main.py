from utils.utils import load_route
from sound_footprint.footprint import Plots
import librosa

def main():
    for i, (route, name_file) in enumerate(load_route().items()):

        # Cargar audio
        audio, sr = librosa.load(route, sr=44100)

        # Calcular the MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

        #Instanciar la clase y llamar metodo
        plot = Plots()
        plot.spectrogram(mfcc, sr, name_file)
        plot.plot_energy(audio, sr, name_file)
        plot.plot_high_low_energy(audio,sr, name_file)




if __name__ == "__main__":
    main()