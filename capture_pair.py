#spawn vehicle, attach RGB and semantic cameras with the same settings at the same position, captures images and save.

import queue
import random
from pathlib import Path

import carla


# -------------------- Configuration --------------------

IMAGE_WIDTH = "320"
IMAGE_HEIGHT = "180"
FOV = "90"
TIMESTEP = 0.1

output_dirs = {
    "rgb": Path("data/paired/rgb"),
    "raw": Path("data/paired/semantic_raw"),
    "visual": Path("data/paired/semantic_visual"),
}

for directory in output_dirs.values():
    directory.mkdir(parents=True, exist_ok=True)


def configure_camera(blueprint):
    """Give both cameras identical settings."""
    blueprint.set_attribute("image_size_x", IMAGE_WIDTH)
    blueprint.set_attribute("image_size_y", IMAGE_HEIGHT)
    blueprint.set_attribute("fov", FOV)
    blueprint.set_attribute("sensor_tick", str(TIMESTEP))
    return blueprint


# -------------------- Connect to CARLA --------------------

client = carla.Client("localhost", 2000)
client.set_timeout(15.0)

world = client.get_world()
original_settings = world.get_settings()
blueprints = world.get_blueprint_library()

actors = []
rgb_queue = queue.Queue()
semantic_queue = queue.Queue()

try:
    # Synchronous mode guarantees controlled simulation frames.
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = TIMESTEP
    world.apply_settings(settings)

    # Find a free spawn point.
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    vehicle_bp = blueprints.find("vehicle.tesla.model3")
    vehicle = None

    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            break

    if vehicle is None:
        raise RuntimeError("No free vehicle spawn point found.")

    actors.append(vehicle)

    # Create identically configured cameras.
    rgb_bp = configure_camera(
        blueprints.find("sensor.camera.rgb")
    )
    semantic_bp = configure_camera(
        blueprints.find("sensor.camera.semantic_segmentation")
    )

    camera_transform = carla.Transform(
        carla.Location(x=1.5, z=2.0),
        carla.Rotation(pitch=-5.0),
    )

    rgb_camera = world.spawn_actor(
        rgb_bp,
        camera_transform,
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )

    semantic_camera = world.spawn_actor(
        semantic_bp,
        camera_transform,
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )

    actors.extend([rgb_camera, semantic_camera])

    rgb_camera.listen(rgb_queue.put)
    semantic_camera.listen(semantic_queue.put)

    # Advance exactly one simulation step.
    world.tick()

    rgb_image = rgb_queue.get(timeout=10.0)
    semantic_image = semantic_queue.get(timeout=10.0)

    # The pair is valid only if both frame IDs match.
    if rgb_image.frame != semantic_image.frame:
        raise RuntimeError(
            f"Frame mismatch: RGB={rgb_image.frame}, "
            f"Semantic={semantic_image.frame}"
        )

    frame_id = rgb_image.frame
    filename = f"{frame_id:06d}.png"

    rgb_image.save_to_disk(
        str(output_dirs["rgb"] / filename)
    )

    semantic_image.save_to_disk(
        str(output_dirs["raw"] / filename),
        carla.ColorConverter.Raw,
    )

    semantic_image.save_to_disk(
        str(output_dirs["visual"] / filename),
        carla.ColorConverter.CityScapesPalette,
    )

    print(f"Synchronized frame captured: {frame_id}")
    print(f"Resolution: {rgb_image.width} x {rgb_image.height}")
    print(f"Saved under: {Path('data/paired').resolve()}")

    input("Inspect the images, then press Enter to clean up...")

finally:
    # Cameras must stop listening before destruction.
    for actor in actors[1:]:
        actor.stop()

    # Destroy cameras first, then the vehicle.
    for actor in reversed(actors):
        actor.destroy()

    # Prevent later scripts from getting stuck in synchronous mode.
    world.apply_settings(original_settings)
    print("Cleanup complete.")