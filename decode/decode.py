import regex as re
import json
import os


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

class Tokenizer:
    def __init__(self, assert_dir: str):

        with open(os.path.join(assert_dir, "vocab.json"), encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.byte_encoder = bytes_to_unicode()
        # for byte_value, char in self.byte_encoder.items():
        #     print(f"Byte value: {byte_value}, Character: {char}, Unicode value: {ord(char)}")

        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        sorted_items = sorted(self.byte_decoder.items(), key=lambda item: item[0])

        # Print sorted items
        for char, byte in sorted_items:
            print(f"Character: {ord(char)}, Byte: {(byte)}")

    def decode(self, indices):
        text = "".join([self.decoder.get(index) for index in indices])
        text = ""
        for order, index in enumerate(indices):
            result = self.decoder.get(index)
            print(f"Index {order}: {result}")
            text += result
        # print(text)
        # for c in text:
        #     print(f'Character: {ord(c)}, Byte Decoder Value: {self.byte_decoder[c]}')

        return bytearray(self.byte_decoder[c] for c in text).decode("utf-8")

def test_tokenizer():
    t = Tokenizer(os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets"))
    indices = [464,11398,10905,447,247,82,9552,318,3562,284,564,251,36410,447,251,6697,286,262,2266,10905,447,247,82,4069,13,14657,88,1682,6370,284,651,503,287,2166,286,6319,12,5124,13,770,318,13013,416,4634,262,2496,604,19867,4058,286,6319,12,5124,447,247,82,1459,4067,287,262,4571,326,6319,12,5124,318,16574,13,1881,6631,284,428,318,618,6319,12,5124,318,11300,510,13,14444,284,281,30343,5434,287,262,2438,11,262,17952,3407,257,1364,11677,4961,284,262,2938,510,11677,13,198,464,779,286,6319,12,5124,447,247,82,2938,2292,3578,262,2137,284,14561,262,16408,2438,329,14657,88,13,1649,428,2438]
    print(t.decode(indices))

if __name__ == "__main__":
    test_tokenizer()
