n_z = 25                # Die Anzahl der Dimensionen in der Repräsentationsschicht
batch_size = 64         # Die Anzahl der Bilder pro Minibatcheinheit

input_shape = (64,64)   # Momentan sind die Bilder auf 64x64 festgelegt mit diesen Wert kann man das ädnern (es muss allerdings der Datensatz angepasst werden)


stuttering = 10         # Gibt das Verhältnis an mit dem ein Frame verworfen wird
DATASET_NAME = "Dataset/1to"+str(stuttering)+"ShuffledDataset.hdf5"
DATASET_DB_NAME = "training"

RUNS = 101          # Gibt die Anzahl der Trainingsepochen an
OFFSET = 0          # Erlaubt es den Einstiegspunkt zu setzen, falls das Modell nachträglich weiter trainiert werden soll (So bleiben Resultate erhalten und nichst wird überschrieben)


INSPECT_DATASET = False

LEARNING_RATE = 0.0003



# Tensorboard unterstützt nur eine gewissen Anzahl an Datensätzen wenn wein Spritesheet verwendet wird
MAX_NUMBERS_EMBEDDING = 16384

# Die einzelnen intervalle in denen die Ergebnisse dokumentiert werden sollen
VIDEO_INTERVALL = 50
CHECKPOINT_INTERVALL = 20
LATENT_INTERVALL = 1
RECONSTRUCTION_INTERVALL = 1
