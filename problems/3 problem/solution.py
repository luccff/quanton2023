import numpy as np
from PIL import Image
from dimod import BinaryQuadraticModel
from dimod.reference.samplers import SimulatedAnnealingSampler

def load_image(path):
    image = Image.open(path).convert('L')
    data = np.array(image)
    return data

def create_magnetic_fields(data):
    mean_intensity = np.mean(data)
    alpha = 1
    h = {}
    for idx, value in np.ndenumerate(data):
        h[idx] = -alpha * (value - mean_intensity) / 255
    return h

def create_interaction_graph(data):
    beta = 2
    sigma = 10
    rows, cols = data.shape
    J = {}
    for i in range(rows):
        for j in range(cols):
            current_pixel = data[i, j]
            if i < rows - 1:
                neighbor_pixel = data[i + 1, j]
                diff = abs(int(current_pixel) - int(neighbor_pixel))
                weight = -beta * np.exp(- (diff ** 2) / (2 * sigma ** 2))
                J[((i, j), (i + 1, j))] = weight
            if j < cols - 1:
                neighbor_pixel = data[i, j + 1]
                diff = abs(int(current_pixel) - int(neighbor_pixel))
                weight = -beta * np.exp(- (diff ** 2) / (2 * sigma ** 2))
                J[((i, j), (i, j + 1))] = weight
    return J

def create_bqm(h, J):
    bqm = BinaryQuadraticModel.empty(vartype='SPIN')
    bqm.add_linear_from(h)
    bqm.add_quadratic_from(J)
    return bqm

def segment_image(input_path, output_path):
    print("Загрузка и обработка изображения...")
    data = load_image(input_path)
    h = create_magnetic_fields(data)
    J = create_interaction_graph(data)
    bqm = create_bqm(h, J)

    print("Выполнение симуляции квантового отжига...")
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=10, num_sweeps=100)
    sample = sampleset.first.sample

    print("Формирование сегментированного изображения...")
    segmented = np.zeros(data.shape, dtype=np.uint8)
    for idx in np.ndindex(data.shape):
        segmented[idx] = 0 if sample[idx] == -1 else 255

    segmented_image = Image.fromarray(segmented)
    segmented_image.save(output_path)
    print(f"Сегментированное изображение сохранено как '{output_path}'.")


input_image_path = '4.png'
output_image_path = '4_segmented.png'
segment_image(input_image_path, output_image_path)
