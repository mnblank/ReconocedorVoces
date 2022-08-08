import winsound
import wave
import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import math
import librosa
import librosa.display
from fastdtw import fastdtw



def calcularDistancia(i1, i2):
    return np.sum((i1-i2)**2)


def normalizar(spec):
    spec = spec + np.abs(np.min(spec))
    return spec/np.max(spec)

def moverTiempo(spec):
    inicioAudio = [];
    for i in range(20,spec.shape[0]):
        for j in range(spec.shape[1]):
            if(spec[i][j] != 0.0):
                inicioAudio.append(j)
                break
            
    inicioAudio = np.min(inicioAudio)
    nuevoSpec = np.zeros((spec.shape[0], 64))
    for i in range(spec.shape[0]):
        y = 0;
        for j in range(inicioAudio, spec.shape[1]):
            nuevoSpec[i][y] = spec[i][j]
            y = y+1
            
    return nuevoSpec
    

# Grabacion del audio

chunk = 512 # grabar paquetes de 512 muestras
type_format = pyaudio.paInt16 # resolucion de 16 bits
channels = 2 # numero de canales a trabajar
fs = 8000 # velocidad de muestreo
tiempo = 1.5 # tiempo de grabacion en segundos
nombre = 'sonido.wav' #nombre del archivo

obj_audio = pyaudio.PyAudio() # objeto de tipo audio
input('Haga click en cualquier tecla para pedir su producto (Agua, Cafe o Jugo):')
print('Inicio del pedido:')
stream = obj_audio.open(format = type_format, channels = channels, rate = fs,
                        frames_per_buffer = chunk, input = True)

tramas = []
sonido = []

for i in range(0, int(fs/chunk*tiempo)):
    datos = stream.read(chunk)
    tramas.append(datos)
    sonido.append(np.frombuffer(datos,dtype = np.int16))
stream.stop_stream()
stream.close()
obj_audio.terminate()
print('Fin del pedido.')

escuchar = wave.open(nombre, 'wb')
escuchar.setnchannels(channels)
escuchar.setsampwidth(obj_audio.get_sample_size(type_format))
escuchar.setframerate(fs)
escuchar.writeframes(b''.join(tramas))
escuchar.close()

winsound.PlaySound(nombre, winsound.SND_FILENAME|winsound.SND_ASYNC)

# Se carga el audio grabado
#sonidoL, rate = librosa.load('sonido.wav')
sonidoL, rate = librosa.load('pruebaDiego.wav')
#sonidoL, rate = librosa.load('pruebaAbi.wav')
#sonidoL, rate = librosa.load('pruebaAlex.wav')

# Se cargan los audios pre-grabados de cada persona
aguaDiego, rateA = librosa.load('agua_Diego.wav')
cafeDiego, rateC = librosa.load('cafe_Diego.wav')
jugoDiego, rateJ = librosa.load('jugo_Diego.wav')

aguaAbi, rateA = librosa.load('agua_Abi.wav')
cafeAbi, rateC = librosa.load('cafe_Abi.wav')
jugoAbi, rateJ = librosa.load('jugo_Abi.wav')

aguaAlex, rateA = librosa.load('Agua_Alex.wav')
cafeAlex, rateC = librosa.load('Cafe_Alex.wav')
jugoAlex, rateJ = librosa.load('Jugo_Alex.wav')


# Se crean los espectrometros de Mel

sonidoSpec = librosa.feature.melspectrogram(sonidoL)

aguaSpecDiego = librosa.feature.melspectrogram(aguaDiego)
cafeSpecDiego = librosa.feature.melspectrogram(cafeDiego)
jugoSpecDiego = librosa.feature.melspectrogram(jugoDiego)

aguaSpecAbi = librosa.feature.melspectrogram(aguaAbi)
cafeSpecAbi = librosa.feature.melspectrogram(cafeAbi)
jugoSpecAbi = librosa.feature.melspectrogram(jugoAbi)

aguaSpecAlex = librosa.feature.melspectrogram(aguaAlex)
cafeSpecAlex = librosa.feature.melspectrogram(cafeAlex)
jugoSpecAlex = librosa.feature.melspectrogram(jugoAlex)



# Se convierten a decibeles y se normalizan

s = normalizar(librosa.amplitude_to_db(sonidoSpec))

aguaDbDiego = normalizar(librosa.amplitude_to_db(aguaSpecDiego))
cafeDbDiego = normalizar(librosa.amplitude_to_db(cafeSpecDiego))
jugoDbDiego = normalizar(librosa.amplitude_to_db(jugoSpecDiego))

aguaDbAbi = normalizar(librosa.amplitude_to_db(aguaSpecAbi))
cafeDbAbi = normalizar(librosa.amplitude_to_db(cafeSpecAbi))
jugoDbAbi = normalizar(librosa.amplitude_to_db(jugoSpecAbi))

aguaDbAlex = normalizar(librosa.amplitude_to_db(aguaSpecAlex))
cafeDbAlex = normalizar(librosa.amplitude_to_db(cafeSpecAlex))
jugoDbAlex = normalizar(librosa.amplitude_to_db(jugoSpecAlex))


# Se elimina el espacio vacio entre el inicio de la grabacion y el audio

s = moverTiempo(s)

aguaDbDiego = moverTiempo(aguaDbDiego)
cafeDbDiego = moverTiempo(cafeDbDiego)
jugoDbDiego = moverTiempo(jugoDbDiego)

aguaDbAbi = moverTiempo(aguaDbAbi)
cafeDbAbi = moverTiempo(cafeDbAbi)
jugoDbAbi = moverTiempo(jugoDbAbi)

aguaDbAlex = moverTiempo(aguaDbAlex)
cafeDbAlex = moverTiempo(cafeDbAlex)
jugoDbAlex = moverTiempo(jugoDbAlex)

# Graficas de espectrogramas de mel

# Para Diego

fig, ax = plt.subplots(1,4, figsize = (20,10))
ax[0].set(title = 'Mel Spectrogram of sonido')
i = librosa.display.specshow(s, ax=ax[0])
ax[1].set(title = 'Mel Spectrogram of agua')
librosa.display.specshow(aguaDbDiego, ax=ax[1])
ax[2].set(title = 'Mel Spectrogram of cafe')
librosa.display.specshow(cafeDbDiego, ax=ax[2])
ax[3].set(title = 'Mel Spectrogram of jugo')
librosa.display.specshow(jugoDbDiego, ax=ax[3])
plt.colorbar(i)

# Para Abi

fig, ax = plt.subplots(1,4, figsize = (20,10))
ax[0].set(title = 'Mel Spectrogram of sonido')
i = librosa.display.specshow(s, ax=ax[0])
ax[1].set(title = 'Mel Spectrogram of agua')
librosa.display.specshow(aguaDbAlex, ax=ax[1])
ax[2].set(title = 'Mel Spectrogram of cafe')
librosa.display.specshow(cafeDbAlex, ax=ax[2])
ax[3].set(title = 'Mel Spectrogram of jugo')
librosa.display.specshow(jugoDbAlex, ax=ax[3])
plt.colorbar(i)

# Para Alex

fig, ax = plt.subplots(1,4, figsize = (20,10))
ax[0].set(title = 'Mel Spectrogram of sonido')
i = librosa.display.specshow(s, ax=ax[0])
ax[1].set(title = 'Mel Spectrogram of agua')
librosa.display.specshow(aguaDbAbi, ax=ax[1])
ax[2].set(title = 'Mel Spectrogram of cafe')
librosa.display.specshow(cafeDbAbi, ax=ax[2])
ax[3].set(title = 'Mel Spectrogram of jugo')
librosa.display.specshow(jugoDbAbi, ax=ax[3])
plt.colorbar(i)


#print('Distancias Diego')
#print(calcularDistancia(s,aguaDbDiego))
#print(calcularDistancia(s,cafeDbDiego))
#print(calcularDistancia(s,jugoDbDiego))
##print('Distancias Ahyde')
##print(calcularDistancia(s,aguaDbAhyde))
##print(calcularDistancia(s,cafeDbAhyde))
##print(calcularDistancia(s,jugoDbAhyde))
#print('Distancias Abi')
#print(calcularDistancia(s,aguaDbAbi))
#print(calcularDistancia(s,cafeDbAbi))
#print(calcularDistancia(s,jugoDbAbi))
#print('Distancias Alex')
#print(calcularDistancia(s,aguaDbAlex))
#print(calcularDistancia(s,cafeDbAlex))
#print(calcularDistancia(s,jugoDbAlex))


# Calculo de Distancias

distancias = []

# Diego
distancias.append(calcularDistancia(s,aguaDbDiego))
distancias.append(calcularDistancia(s,cafeDbDiego))
distancias.append(calcularDistancia(s,jugoDbDiego))

# Abi
distancias.append(calcularDistancia(s,aguaDbAbi))
distancias.append(calcularDistancia(s,cafeDbAbi))
distancias.append(calcularDistancia(s,jugoDbAbi))

# Alex
distancias.append(calcularDistancia(s,aguaDbAlex))
distancias.append(calcularDistancia(s,cafeDbAlex))
distancias.append(calcularDistancia(s,jugoDbAlex))


# Se toma la posicion a la que 
minimo = np.argmin(distancias)

producto = ''
comprador = ''

# Se toma el Nombre del comprador de distancias

if minimo >= 0 and minimo < 3:
    comprador = 'Diego'
elif minimo >= 3 and minimo < 6:
    comprador = 'Abi'
else:
    comprador = 'Alex'
    
# Igual con el producto    
    
if minimo == 0 or minimo == 3 or minimo == 6:
    producto = 'Agua'
elif minimo == 1 or minimo == 4 or minimo == 7:
    producto = 'Cafe'
else:
    producto = 'Jugo'
     
    
# Se construye el mensaje de salida
mensaje = 'Buen dia ' + comprador + '. En seguida recibiras tu ' + producto + '.'
print(mensaje)



















