# Library
from PIL import Image, ImageDraw


# Crop center image
def crop_center(image, size=(400, 400)):
    width, height = image.size
    target_width, target_height = size

    # Determining this Center Image
    left = (width - min(width, height)) // 2
    top = (height - min(width, height)) // 2
    right = left + min(width, height)
    bottom = top + min(width, height)

    # Cropping center image
    image = image.crop((left, top, right, bottom))

    # Resize
    image = image.resize(size, Image.LANCZOS)
    return image

# Making the image circle
def make_circle(image, size=(400, 400)):
    # Crop center and resize
    img = crop_center(image, size).convert("RGBA")
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size[0], size[1]), fill=255)

    # Make circle image
    result = Image.new("RGBA", size, (0, 0, 0, 0))
    result.paste(img, (0, 0), mask)
    return result