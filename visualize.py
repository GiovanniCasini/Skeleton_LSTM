import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

kitml_body_connections = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(4,8),(8,9),(9,10),(0,11),(0,16),(11,12),(12,13),(13,14),(14,15),(16,17),(17,18),(18,19),(19,20)] 

guoh3djoints_body_connections = [  # no hands
        (0, 3), (3,6),(6, 9),(9,12), (12, 15),
        (9, 13), (13,16), (16,18), (18,20),
        (9, 14), (14,17), (17,19), (19,21),
        (0, 1), (1,4), (4,7), (7,10),
        (0, 2), (2,5), (5,8), (8,11)
    ]


def numpy_to_video(np_data, save_path, connections=True, body_connections="kitml", text=None):
    # np_data: numpy array of predictions body movements
    # save_path: where the mp4 video will be saved
    num_frames = np_data.shape[0]
    num_joints = np_data.shape[1]

    body_connections = kitml_body_connections if body_connections=="kitml" else guoh3djoints_body_connections

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

        if connections:
            for c1, c2 in body_connections:
                plt.plot([np_data[num_frame,c1,0], np_data[num_frame,c2,0]], [np_data[num_frame,c1,1], np_data[num_frame,c2,1]], [np_data[num_frame,c1,2], np_data[num_frame,c2,2]])
        
        for i in range(num_joints):
            ax.text(np_data[num_frame,i,0], np_data[num_frame,i,1], np_data[num_frame,i,2], str(i))

    if text is not None:
        ax.text(x=0.5, y=1.05, z=0, s= text, transform=ax.transAxes, ha='center')
    # Crea l'animazione utilizzando le funzioni di inizializzazione e aggiornamento
    ani = animation.FuncAnimation(fig, update, frames=range(num_frames), init_func=init, blit=False)

    plt.show()

    # Salva l'animazione come file mp4, bisogna avere ffmepg installato
    print(f"Saving video in {save_path}")
    ani.save(save_path, writer='ffmpeg')
    print("Done")


def main():
    # Carica il file .npy
    file_name = "00029_motion"
    file_path = f"{os.getcwd()}/kit_numpy/test/{file_name}.npy"
    np_data = np.load(file_path)

    # Chiama la funzione numpy_to_video con i dati caricati e il percorso di salvataggio
    save_path = f"{os.getcwd()}/visualizations/{file_name}.mp4"
    numpy_to_video(np_data, save_path)



if __name__ == "__main__":
    main()