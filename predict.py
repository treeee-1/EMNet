
from PIL import Image
from deeplab import DeeplabV3

if __name__ == "__main__":

    deeplab = DeeplabV3()

    while True:
        img = input('Input image filename:')
        canny = input('Input image filename:')
        try:
            image = Image.open(img)
            canny2 = Image.open(canny)

        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = deeplab.detect_image(image,canny2)
            r_image.show()
