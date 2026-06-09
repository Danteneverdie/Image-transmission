# run_decoder.py
import os
import sys
import math
import pickle
import socket
import cv2
import numpy as np
from bitarray import bitarray
import matplotlib.pyplot as plt  # 导入绘图库

# 导入网络模块中的同步接收函数
from network import receive_file
from utils import IDWT, inverse_quantize, ezw_inverse_scan_2d, inverse_raster_scan, low_subband_reconstruct
from huffman import huffman_decode

sys.setrecursionlimit(20000)

def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def imageDecoder(bitstreamFileName, quantizationStepSize, orgImageFileName):
    # 1. 物理读取
    with open(bitstreamFileName, "rb") as f:
        ba_bytes = f.read()

    # 2. 解析 8 字节头部
    len_payload = int.from_bytes(ba_bytes[0:4], 'big')
    len_data_bits = int.from_bytes(ba_bytes[4:8], 'big')

    # 3. 反序列化
    payload_pickle = ba_bytes[8: 8 + len_payload]
    payload = pickle.loads(payload_pickle)
    codebook = payload['codebook']
    initial_T = payload['initial_T']

    # 4. 码流截断
    data_bytes = ba_bytes[8 + len_payload:]
    data_bits = bitarray(endian='big')
    data_bits.frombytes(data_bytes)
    bitstream = data_bits.to01()[:len_data_bits]

    # 5. 哈夫曼解码
    decoded_combined_symbols = huffman_decode(bitstream, codebook)

    # 6. 分离高低频
    decoded_ll_symbols = decoded_combined_symbols[:256]
    decoded_high_symbols = decoded_combined_symbols[256:]

    # 7. 重建低频
    reconstructed_predicted_ll = inverse_raster_scan(decoded_ll_symbols, shape=(16, 16))
    reconstructed_quantized_ll = low_subband_reconstruct(reconstructed_predicted_ll)

    # 8. 重建高频
    reconstructed_high_freq = ezw_inverse_scan_2d(decoded_high_symbols, initial_T, img_size=512, num_levels=5)

    # 9. 拼接重组
    decoded_q_coeffs = reconstructed_high_freq.copy()
    decoded_q_coeffs[0:16, 0:16] = reconstructed_quantized_ll

    # 10. 逆量化
    dequantized_coeffs = inverse_quantize(decoded_q_coeffs, quantizationStepSize)

    # 11. 逆小波变换 (IDWT)
    factor = np.sqrt(2.0)
    G_0 = np.array([0.5, 1.0, 0.5], dtype=np.float64) / factor
    G_1 = np.array([-0.125, -0.25, 0.75, -0.25, -0.125], dtype=np.float64) / factor
    reconstructed_img = IDWT(dequantized_coeffs, num=5, G_0=G_0, G_1=G_1)

    # 保存图像
    base_name = os.path.splitext(os.path.basename(orgImageFileName))[0]
    os.makedirs(base_name, exist_ok=True)
    save_filename = os.path.join(base_name, f"{base_name}_{quantizationStepSize}.png")
    save_img = np.clip(reconstructed_img, 0, 255).astype(np.uint8)
    cv2.imwrite(save_filename, save_img)
    print(f"Reconstructed image saved as: {save_filename}")

    # 读取原始图像用于指标计算
    org_img = cv2.imread(orgImageFileName, cv2.IMREAD_GRAYSCALE)
    if org_img is None:
        raise FileNotFoundError(f"Original comparison image '{orgImageFileName}' not found.")

    # 计算 PSNR
    org_img_float = np.array(org_img, dtype=np.float64)
    psnr = calculate_psnr(org_img_float, reconstructed_img)

    # 计算 Bitrate (比特率 = 总比特数 / 图像总像素)
    height, width = org_img.shape
    bitrate = len_data_bits / (height * width)

    print("Done decoding.", flush=True)
    return psnr, bitrate


def main():
    LISTEN_IP = '0.0.0.0'
    PORT = 65432
    # orgImageFileName = 'image/lena.tif'
    orgImageFileName = 'image/car.png'
    # orgImageFileName = 'image/town.png'
    q_values = [8, 16, 32, 64, 128]

    # 用于存放各级别数据的列表，绘制 R-D 曲线时使用
    bitrates = []
    psnrs = []
    successful_qs = []

    # 初始化监听 Socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((LISTEN_IP, PORT))
        s.listen(5)

        for q in q_values:
            print(f"\n--- 正在等待步长 q = {q} 的码流文件 ---")

            received_filename = 'received_image.bit'
            if os.path.exists(received_filename):
                os.remove(received_filename)

            # 定义一个内部闭包回调函数
            def decoder_callback(save_path):
                try:
                    # 解码并返回 PSNR 和 Bitrate
                    psnr, bitrate = imageDecoder(save_path, q, orgImageFileName)
                    print(f"Successfully reconstructed. PSNR: {psnr:.2f} dB, Bitrate: {bitrate:.4f} bpp")

                    # 记录计算数据
                    psnrs.append(psnr)
                    bitrates.append(bitrate)
                    successful_qs.append(q)
                    return True
                except Exception as e:
                    print(f"Failed decoding this level: {e}")
                    return False

            # 调用网络同步接收函数
            received_path = receive_file(s, save_dir='.', process_callback=decoder_callback)

            if not received_path:
                print("An error occurred during network transmission. Stopped.")
                break

    # 绘制并保存 R-D (率失真) 曲线
    if len(bitrates) > 0:
        print("\nGenerating Rate-Distortion (R-D) Curve...")
        plt.figure(figsize=(8, 6))

        # 绘制折线图，横轴为 Bitrate，纵轴为 PSNR
        plt.plot(bitrates, psnrs, marker='o', linestyle='-', color='r', linewidth=2, label="R-D Curve")

        # 标注每个点对应的量化步长 q
        for i, q in enumerate(successful_qs):
            plt.annotate(f"q={q}", (bitrates[i], psnrs[i]), textcoords="offset points", xytext=(0, 10), ha='center',
                         fontsize=9)

        plt.title('Rate-Distortion (R-D) Curve', fontsize=14)
        plt.xlabel('Rate (bits/pixel, bpp)', fontsize=12)
        plt.ylabel('PSNR (dB)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc="lower right")

        # 保存R-D曲线
        base_name = os.path.splitext(os.path.basename(orgImageFileName))[0]
        os.makedirs(base_name, exist_ok=True)
        save_curve_path = os.path.join(base_name, 'RD_Curve.png')
        plt.savefig(save_curve_path, dpi=300)
        print("R-D curve successfully saved as 'RD_Curve.png'")

        # 在屏幕上弹出显示曲线图
        plt.show()
    else:
        print("No transmission data collected. Skipped plotting R-D curve.")


if __name__ == "__main__":
    main()