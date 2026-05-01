import os

def count_images(path):
    total = 0
    for root, dirs, files in os.walk(path):
        total += len(files)
    return total

print("Train:", count_images("data/train"))
print("Val:", count_images("data/val"))