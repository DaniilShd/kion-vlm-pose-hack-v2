import numpy as np

def yolo_to_stgcn(npy_file, max_persons=2, num_frames=100):
    
    data = np.load(npy_file, allow_pickle=True)

    T = min(len(data), num_frames)
    V = 17
    C = 3
    M = max_persons
    N = 1

    skeleton = np.zeros((N, M, T, V, C), dtype=np.float32)

    for t in range(T):

        frame = data[t]
        kps = frame["keypoints"]   # (persons,17,2)

        persons = min(kps.shape[0], M)

        for m in range(persons):

            xy = kps[m]

            skeleton[0, m, t, :, 0:2] = xy
            skeleton[0, m, t, :, 2] = 1.0   # confidence

    return skeleton