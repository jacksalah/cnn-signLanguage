from PIL import Image
import os

directory = "E:\\4th year\\1st term\\Project\\prototype\\CNN\\dataset_canny\\test_set\\A"

for file_name in os.listdir(directory):
    print("Processing %s" % file_name)
    image = Image.open(os.path.join(directory, file_name))

    x, y = image.size
    new_dimensions = (150, 150)
    output = image.resize(new_dimensions, Image.ANTIALIAS)
    output_file_name = os.path.join(directory, file_name)
    output.save(output_file_name, "png", quality=95)

print("All done")
