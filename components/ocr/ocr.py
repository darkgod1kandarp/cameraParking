from components.ocr.config import img_transform, parseq  # noq
from PIL import Image
def OcrThroughArrya(imgArray):
        # imgArray = torch.from_numpy(imgArray)
        print('wqef')
        imgArray = Image.fromarray(imgArray)
        print('qwr')
        imgArray = img_transform(imgArray).unsqueeze(0)

        logits = parseq(imgArray)
        logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

        # Greedy decoding
        pred = logits.softmax(-1)
        label, confidence = parseq.tokenizer.decode(pred)
        return label[0]