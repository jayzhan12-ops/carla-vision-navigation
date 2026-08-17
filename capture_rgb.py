import random
from pathlib import Path
from queue import Queue

import carla


client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()

vehicle = None
rgb_camera = None

# The callback will place received images into this queue.
image_queue = Queue()

try:
    blueprint_library = world.get_blueprint_library()

    # -------------------------
    # Spawn the ego vehicle
    # -------------------------
    vehicle_blueprint = blueprint_library.find(
        "vehicle.tesla.model3"
    )

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(
            vehicle_blueprint,
            spawn_point,
        )

        if vehicle is not None:
            break

    if vehicle is None:
        raise RuntimeError("Could not spawn the ego vehicle")

    print("Vehicle spawned:", vehicle.type_id)

    # -------------------------
    # Configure the RGB camera
    # -------------------------
    camera_blueprint = blueprint_library.find(
        "sensor.camera.rgb"
    )

    camera_blueprint.set_attribute("image_size_x", "320")
    camera_blueprint.set_attribute("image_size_y", "180")
    camera_blueprint.set_attribute("fov", "90")
    camera_blueprint.set_attribute("sensor_tick", "0.1")

    # This transform is relative to the vehicle.
    camera_transform = carla.Transform(
        carla.Location(x=1.5, z=2.0),
        carla.Rotation(pitch=-5.0),
    )

    rgb_camera = world.spawn_actor(
        camera_blueprint,
        camera_transform,
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )

    print("RGB camera attached")

    # Start receiving images.
    rgb_camera.listen(image_queue.put)

    print("Waiting for one RGB frame...")

    # Wait for a camera image, with a maximum wait of 10 seconds.
    image = image_queue.get(timeout=10.0)

    output_directory = Path("data/rgb")
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / "first_frame.png"

    image.save_to_disk(str(output_path.resolve()))

    print("Frame received:", image.frame)
    print("Image size:", image.width, "x", image.height)
    print("Image saved to:", output_path.resolve())

    input("Open the image and inspect it. Press Enter to clean up...")

finally:
    if rgb_camera is not None:
        rgb_camera.stop()
        rgb_camera.destroy()
        print("RGB camera destroyed")

    if vehicle is not None:
        vehicle.destroy()
        print("Vehicle destroyed")