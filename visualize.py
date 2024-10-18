import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

def numpy_to_video(np_data, save_path):
    # np_data: numpy array of predictions body movements
    # save_path: where the mp4 video will be saved
    num_frames = np_data.shape[0]
    num_joints = np_data.shape[1]

    body_connections = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(4,8),(8,9),(9,10),(0,11),(0,16),(11,12),(12,13),(13,14),(14,15),(16,17),(17,18),(18,19),(19,20)] 

    # Swap axis to visualize correctly
    tmp = np_data.copy()
    np_data[:, :, 1] = tmp[:,:, 2]
    np_data[:, :, 2] = tmp[:,:, 1]

    # Crea una figura e un asse 3D
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Set camera height
    ax.view_init(elev=10.)

    min_x, max_x = min(np_data[:,:,0].flatten()), max(np_data[:,:,0].flatten())
    min_y, max_y = min(np_data[:,:,1].flatten()), max(np_data[:,:,1].flatten())
    min_z, max_z = min(np_data[:,:,2].flatten()), max(np_data[:,:,2].flatten())

    # Inizializzazione della funzione di trama 
    def init():
        pass

    # Funzione di animazione, qui devi mettere il codice per aggiornare il tuo tracciato
    def update(num_frame):
        ax.cla()  # pulisce l'attuale ax
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)
        # print(f"Frame {num_frame} - x[0]: {data[num_frame,0,0]}")
        ax.scatter(np_data[num_frame,:,0], np_data[num_frame,:,1], np_data[num_frame,:,2], marker="x")

        for c1, c2 in body_connections:
            plt.plot([np_data[num_frame,c1,0], np_data[num_frame,c2,0]], [np_data[num_frame,c1,1], np_data[num_frame,c2,1]], [np_data[num_frame,c1,2], np_data[num_frame,c2,2]])
        
        for i in range(num_joints):
            ax.text(np_data[num_frame,i,0], np_data[num_frame,i,1], np_data[num_frame,i,2], str(i))

    # Crea l'animazione utilizzando le funzioni di inizializzazione e aggiornamento
    ani = animation.FuncAnimation(fig, update, frames=range(num_frames), init_func=init, blit=False)

    plt.show()

    # Salva l'animazione come file mp4, bisogna avere ffmepg installato
    print(f"Saving video in {save_path}")
    ani.save(save_path, writer='ffmpeg')
    print("Done")


def main():
    base_dir = os.getcwd()
    # Carica il file .npy
    file_name = "03902_motion"
    file_path = f"tesi/LSTM_skeleton/kit_numpy/test/{file_name}.npy"
    np_data = np.load(file_path)

    # Chiama la funzione numpy_to_video con i dati caricati e il percorso di salvataggio
    save_path = f"{base_dir}/tesi/LSTM_skeleton/visualizations/{file_name}.mp4"
    numpy_to_video(np_data, save_path)



if __name__ == "__main__":
    main()
