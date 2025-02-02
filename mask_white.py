from PIL import Image


def convert_non_black_to_white(image_path, output_path=r"C:\Users\micha\Downloads\autonomous-drone-final-project-master\autonomous-drone-final-project-master\masked_pic.jpg"):
    img = Image.open(image_path)
    # Convert to grayscale
    img = img.convert('L')
    img = img.point(lambda p: 255 if p > 2 else 0)
    img.save(output_path)
    return output_path

image_path = r"C:\Users\micha\Downloads\autonomous-drone-final-project-master\autonomous-drone-final-project-master\pic.jpg"
outputPath = convert_non_black_to_white(image_path)