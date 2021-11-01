from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
if __name__ == '__main__':
    token_ids = tokenizer.encode('tôi đang tìm một nơi lưu trú có tầm giá rẻ , chắc thuộc loại khách sạn')
    print(token_ids)
    print(tokenizer.decode(token_ids))
