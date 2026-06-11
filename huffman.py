import heapq
from collections import Counter

# 哈夫曼树节点定义
class Node:
    def __init__(self, char, freq):
        self.char = char  # 符号（如 'EZT', 'IZ', (3,6), 0 等）
        self.freq = freq  # 出现频率
        self.left = None
        self.right = None

    # 定义比较规则
    def __lt__(self, other):
        return self.freq < other.freq


# 构建哈夫曼树
def generate_code(node, current_code, codes):
    if node is None:
        return
    if node.char is not None:
        codes[node.char] = current_code
        return
    generate_code(node.left, current_code + "0", codes)
    generate_code(node.right, current_code + "1", codes)

# 构建哈夫曼编码
def huffman_encode(data_list):
    frequencies = Counter(data_list)

    #构建最小堆
    heap = []
    for char, freq in frequencies.items():
        heapq.heappush(heap, Node(char, freq))

    # 处理只有一个唯一符号的边界情况
    if len(heap) == 1:
        node = heapq.heappop(heap)
        root = Node(None, node.freq)
        root.left = node
        heapq.heappush(heap, root)

    # 循环合并节点，构建哈夫曼树
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        parent = Node(None, node1.freq + node2.freq)
        parent.left = node1
        parent.right = node2
        heapq.heappush(heap, parent)

    # 树根
    root = heap[0]

    # 生成密码本
    codes = {}
    generate_code(root, "", codes)
    # 将原始数据流翻译为二进制 01 串
    bitstream = "".join(codes[char] for char in data_list)

    return bitstream, codes

# 解码哈夫曼编码
def huffman_decode(bitstream, codebook):
    if not bitstream or not codebook:
        return []

    # 建立 "码字 -> 符号" 的映射
    reverse_codebook = {v: k for k, v in codebook.items()}

    decoded_list = []
    current_code = ""

    # 逐位扫描码流进行匹配
    for bit in bitstream:
        current_code += bit
        if current_code in reverse_codebook:
            decoded_list.append(reverse_codebook[current_code])
            current_code = ""  # 匹配成功，清空缓存，准备匹配下一个符号

    return decoded_list