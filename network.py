# network.py
import socket
import struct
import os


def send_file(filename, host='127.0.0.1', port=65432):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return False

    filesize = os.path.getsize(filename)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))

        # 1. 发送文件名长度和文件名
        encoded_filename = os.path.basename(filename).encode()
        s.sendall(struct.pack('!I', len(encoded_filename)))
        s.sendall(encoded_filename)

        # 2. 发送文件大小
        s.sendall(struct.pack('!Q', filesize))

        # 3. 发送文件内容
        with open(filename, 'rb') as f:
            while True:
                data = f.read(4096)
                if not data:
                    break
                s.sendall(data)

        print(f"Sent {filename} ({filesize} bytes) successfully. Waiting for receiver to process...")

        # 4. 阻塞等待接收端处理完成后的反馈 (ACK)
        ack = s.recv(4)
        if ack == b"DONE":
            print("Received ACK 'DONE' from receiver. Ready for next step.")
            return True
        else:
            print("Error: Failed to receive valid ACK from receiver.")
            return False


def receive_file(listening_socket, save_dir='.', process_callback=None):
    conn, addr = listening_socket.accept()
    with conn:
        print(f"Connected by {addr}")

        # 1. 读入文件名
        raw_len = conn.recv(4)
        if not raw_len:
            return None
        filename_len = struct.unpack('!I', raw_len)[0]
        filename = conn.recv(filename_len).decode()

        if not filename.lower().endswith('.bit'):
            print(f"Security Alert: Blocked invalid file type '{filename}'. Only '.bit' is allowed.")
            conn.sendall(b"FAIL")
            return None

        # 2. 读入文件大小
        raw_size = conn.recv(8)
        filesize = struct.unpack('!Q', raw_size)[0]

        # 3. 接收并保存文件
        save_path = os.path.join(save_dir, 'received_' + filename)
        received_size = 0
        with open(save_path, 'wb') as f:
            while received_size < filesize:
                data = conn.recv(min(4096, filesize - received_size))
                if not data:
                    break
                f.write(data)
                received_size += len(data)

        print(f"Received {filename} and saved to {save_path} ({received_size} bytes)")

        # 4. 调用外部处理机制 (解码等)，并在连接关闭前向发送端回复信号
        if process_callback:
            success = process_callback(save_path)
            if success:
                conn.sendall(b"DONE")
            else:
                conn.sendall(b"FAIL")
        else:
            conn.sendall(b"DONE")

        return save_path