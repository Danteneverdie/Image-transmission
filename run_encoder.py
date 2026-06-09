# run_encoder.py
import os
import sys
import pickle
import cv2
import numpy as np
from bitarray import bitarray

# 导入网络模块
from network import send_file
from utils import DWT, quantize, low_subband_predict, raster_scan, get_initial_threshold, ezw_scan_2d
from huffman import huffman_encode

sys.setrecursionlimit(20000)


def imageEncoder(orgImageFileName, quantizationStepSize):
    img = cv2.imread(orgImageFileName, cv2.IMREAD_GRAYSCALE)
    arr = np.array(img, dtype=np.float64)
    height, width = arr.shape
    factor = np.sqrt(2.0)

    H_0 = np.array([-0.125, 0.25, 0.75, 0.25, -0.125], dtype=np.float64) * factor
    H_1 = np.array([-0.5, 1.0, -0.5], dtype=np.float64) * factor

    coeffs = DWT(arr, 5, H_0=H_0, H_1=H_1)
    q_coeffs = quantize(coeffs, quantizationStepSize)

    h, w = height // (2 ** 5), width // (2 ** 5)
    LL5 = q_coeffs[0:h, 0:w]
    residual = low_subband_predict(LL5)

    ll_sequence = raster_scan(residual)
    initial_T = get_initial_threshold(q_coeffs, img_size=512)
    high_freq_sequence = ezw_scan_2d(q_coeffs, initial_T, img_size=512, num_levels=5)

    combined_symbols = ll_sequence + high_freq_sequence
    bitstream, codebook = huffman_encode(combined_symbols)
    bitlength = len(bitstream)

    payload = {
        'codebook': codebook,
        'initial_T': initial_T
    }
    payload_pickle = pickle.dumps(payload)
    len_payload = len(payload_pickle)

    data_bits = bitarray(endian='big')
    data_bits.extend(bitstream)
    data_bytes = data_bits.tobytes()

    header = len_payload.to_bytes(4, 'big') + bitlength.to_bytes(4, 'big')
    ba_bytes = header + payload_pickle + data_bytes

    with open("image.bit", "wb") as f:
        f.write(ba_bytes)

    bitrate = bitlength / (height * width)
    return bitrate


def main():

    RECEIVER_IP = '127.0.0.1'
    PORT = 65432
    # orgImageFileName = 'image/lena.tif'
    orgImageFileName = 'image/car.png'
    # orgImageFileName = 'image/town.png'
    q_values = [8, 16, 32, 64, 128]

    for q in q_values:
        print(f"\n--- processing step size q = {q} ---")

        # 1. 编码
        print("Encoding...")
        bitrate = imageEncoder(orgImageFileName, q)
        print(f"Bitrate: {bitrate:.4f} bits/pixel")

        # 2. 网络物理发送
        print(f"Sending 'image.bit' to {RECEIVER_IP}:{PORT} ...")
        success = send_file('image.bit', host=RECEIVER_IP, port=PORT)

        if success:
            print(f"Done processing level q = {q}.")
        else:
            print(f"Failed transmission at level q = {q}. Stopped.")
            break

if __name__ == "__main__":
    main()