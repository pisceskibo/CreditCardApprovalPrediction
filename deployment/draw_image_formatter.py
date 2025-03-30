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

# Display the Score of Credit Card
def option_circle_score_credit_card(score_credit):
    # Display the circle plot
    options = {
        "series": [
            {
                "type": "gauge",
                "startAngle": 90,
                "endAngle": -270,
                "pointer": {"show": False},  # Ẩn kim đồng hồ
                "progress": {
                    "show": True,
                    "overlap": False,
                    "roundCap": True,
                    "clip": False
                },
                "axisLine": {
                    "lineStyle": {
                        "width": 10,
                        "color": [[1, "#FF4081"]]  # Màu hồng đậm
                    }
                },
                "axisTick": {"show": False},
                "splitLine": {"show": False},
                "axisLabel": {"show": False},
                "detail": {
                    "valueAnimation": True,
                    "formatter": "{value}",
                    "color": "#FFD700",  # Màu vàng
                    "fontSize": 40,
                    "fontWeight": "bold",
                    "offsetCenter": [0, "0%"]  # Căn giữa theo cả trục X và Y
                },
                "data": [{"value": score_credit}]
            }
        ]
    }
    return options