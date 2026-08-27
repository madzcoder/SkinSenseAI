from picamzero import Camera
from time import sleep

# Initialize the Camera Module 3
cam = Camera()

print("Opening camera preview...")
cam.start_preview()

# Keep preview open for 5 seconds
sleep(5)

print("Taking a picture...")
cam.take_photo("test_image.jpg")

cam.stop_preview()
print("Done! Image saved as test_image.jpg")
