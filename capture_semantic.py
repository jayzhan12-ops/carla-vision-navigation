import random
from pathlib import Path
from queue import Queue

import carla


client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()

vehicle = None
semantic_camera = None
image_queue = Queue()

try:
    blueprint_library = world.get_blueprint_library()

    # Spawn vehicle
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
        raise RuntimeError("Could not spawn the vehicle")

    print("Vehicle spawned:", vehicle.type_id)

    # Configure semantic camera
    camera_blueprint = blueprint_library.find(
        "sensor.camera.semantic_segmentation"
    )

    camera_blueprint.set_attribute("image_size_x", "320")
    camera_blueprint.set_attribute("image_size_y", "180")
    camera_blueprint.set_attribute("fov", "90")
    camera_blueprint.set_attribute("sensor_tick", "0.1")

    camera_transform = carla.Transform(
        carla.Location(x=1.5, z=2.0),
        carla.Rotation(pitch=-5.0),
    )

    semantic_camera = world.spawn_actor(
        camera_blueprint,
        camera_transform,
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )

    print("Semantic camera attached")

    semantic_camera.listen(image_queue.put)

    print("Waiting for semantic frame...")
    image = image_queue.get(timeout=10.0)

    raw_directory = Path("data/semantic_raw")
    visual_directory = Path("data/semantic_visual")

    raw_directory.mkdir(parents=True, exist_ok=True)
    visual_directory.mkdir(parents=True, exist_ok=True)

    raw_path = raw_directory / "first_frame.png"
    visual_path = visual_directory / "first_frame.png"

    # Save class IDs for future model training.
    image.save_to_disk(
        str(raw_path.resolve()),
        carla.ColorConverter.Raw,
    )

    # Save coloured labels for visual inspection.
    image.save_to_disk(
        str(visual_path.resolve()),
        carla.ColorConverter.CityScapesPalette,
    )

    print("Frame:", image.frame)
    print("Resolution:", image.width, "x", image.height)
    print("Raw labels saved to:", raw_path.resolve())
    print("Visual labels saved to:", visual_path.resolve())

    input("Inspect the images, then press Enter to clean up...")

finally:
    if semantic_camera is not None:
        semantic_camera.stop()
        semantic_camera.destroy()
        print("Semantic camera destroyed")

    if vehicle is not None:
        vehicle.destroy()
        print("Vehicle destroyed")