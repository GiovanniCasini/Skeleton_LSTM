import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

def compare_numpy_to_video(np_data1, np_data2, save_path):    
    num_frames = np_data1.shape[0]
    num_joints = np_data1.shape[1]

    body_connections = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(4,8),(8,9),(9,10),(0,11),(0,16),(11,12),(12,13),(13,14),(14,15),(16,17),(17,18),(18,19),(19,20)] 

    # Swap axis to visualize correctly
    tmp1 = np_data1.copy()
    np_data1[:, :, 1] = tmp1[:,:, 2]
    np_data1[:, :, 2] = tmp1[:,:, 1]

    tmp2 = np_data2.copy()
    np_data2[:, :, 1] = tmp2[:,:, 2]
    np_data2[:, :, 2] = tmp2[:,:, 1]

    # Crea una figura e due assi 3D
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))

    # Set camera height
    ax1.view_init(elev=10.)
    ax2.view_init(elev=10.)

    min_x, max_x = min(np_data1[:,:,0].flatten()), max(np_data1[:,:,0].flatten())
    min_y, max_y = min(np_data1[:,:,1].flatten()), max(np_data1[:,:,1].flatten())
    min_z, max_z = min(np_data1[:,:,2].flatten()), max(np_data1[:,:,2].flatten())

    min_x2, max_x2 = min(np_data2[:,:,0].flatten()), max(np_data2[:,:,0].flatten())
    min_y2, max_y2 = min(np_data2[:,:,1].flatten()), max(np_data2[:,:,1].flatten())
    min_z2, max_z2 = min(np_data2[:,:,2].flatten()), max(np_data2[:,:,2].flatten())

    # Inizializzazione della funzione di trama 
    def init():
        pass

    # Funzione di animazione, qui devi mettere il codice per aggiornare i tuoi tracciati
    def update(num_frame):
        ax1.cla()  # pulisce l'attuale ax
        ax2.cla()
        
        # Ax1 settings for np_data
        ax1.set_xlim(min_x, max_x)
        ax1.set_ylim(min_y, max_y)
        ax1.set_zlim(min_z, max_z)
        
        ax1.scatter(np_data1[num_frame,:,0], np_data1[num_frame,:,1], np_data1[num_frame,:,2], marker="x")
        for c1, c2 in body_connections:
            ax1.plot([np_data1[num_frame,c1,0], np_data1[num_frame,c2,0]], 
                     [np_data1[num_frame,c1,1], np_data1[num_frame,c2,1]], 
                     [np_data1[num_frame,c1,2], np_data1[num_frame,c2,2]])
        for i in range(num_joints):
            ax1.text(np_data1[num_frame,i,0], np_data1[num_frame,i,1], np_data1[num_frame,i,2], str(i))

        # Ax2 settings for np_data2
        ax2.set_xlim(min_x2, max_x2)
        ax2.set_ylim(min_y2, max_y2)
        ax2.set_zlim(min_z2, max_z2)
        
        ax2.scatter(np_data2[num_frame,:,0], np_data2[num_frame,:,1], np_data2[num_frame,:,2], marker="x")
        for c1, c2 in body_connections:
            ax2.plot([np_data2[num_frame,c1,0], np_data2[num_frame,c2,0]], 
                     [np_data2[num_frame,c1,1], np_data2[num_frame,c2,1]], 
                     [np_data2[num_frame,c1,2], np_data2[num_frame,c2,2]])
        for i in range(num_joints):
            ax2.text(np_data2[num_frame,i,0], np_data2[num_frame,i,1], np_data2[num_frame,i,2], str(i))

    # Crea l'animazione utilizzando le funzioni di inizializzazione e aggiornamento
    ani = animation.FuncAnimation(fig, update, frames=range(num_frames), init_func=init, blit=False)

    plt.show()

    # Salva l'animazione come file mp4, bisogna avere ffmepg installato
    print(f"Saving video in {save_path}")
    ani.save(save_path, writer='ffmpeg')
    print("Done")


def main():
    base_dir = os.getcwd()
    # Carica i file .npy
    file_name1 = "00182_motion"
    file_name2 = "00234_motion"
    file_path1 = f"tesi/LSTM_skeleton/kit_numpy/test/{file_name1}.npy"
    file_path2 = f"tesi/LSTM_skeleton/kit_numpy/test/{file_name2}.npy"
    
    np_data1 = np.load(file_path1)
    np_data2 = np.load(file_path2)

    # Chiama la funzione numpy_to_video con i dati caricati e il percorso di salvataggio
    save_path = f"{base_dir}/tesi/LSTM_skeleton/visualizations/{file_name1}_{file_name2}.mp4"
    compare_numpy_to_video(np_data1, np_data2, save_path)


if __name__ == "__main__":
    main()
