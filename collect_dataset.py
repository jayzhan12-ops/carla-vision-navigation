import queue
import random
from pathlib import Path

import carla


# -------------------- Configuration --------------------

IMAGE_WIDTH = "320"
IMAGE_HEIGHT = "180"
FOV = "90"

TIMESTEP = 0.1
SAVE_EVERY_N_FRAMES = 5
WARMUP_FRAMES = 20
NUM_SAMPLES = 20

TM_PORT = 8000
RANDOM_SEED = 42

RUN_NAME = "town10_clear_01"
RUN_DIR = Path("data/dataset") / RUN_NAME

OUTPUT_DIRS = {
    "rgb": RUN_DIR / "rgb",
    "raw": RUN_DIR / "semantic_raw",
    "visual": RUN_DIR / "semantic_visual",
}


def configure_camera(blueprint):
    """Apply identical settings to both cameras."""
    blueprint.set_attribute("image_size_x", IMAGE_WIDTH)
    blueprint.set_attribute("image_size_y", IMAGE_HEIGHT)
    blueprint.set_attribute("fov", FOV)
    blueprint.set_attribute("sensor_tick", str(TIMESTEP))
    return blueprint


def create_output_directories():
    """Create a new dataset run without overwriting old data."""
    if RUN_DIR.exists() and any(RUN_DIR.rglob("*.png")):
        raise RuntimeError(
            f"{RUN_DIR} already contains images. "
            "Change RUN_NAME before collecting again."
        )

    for directory in OUTPUT_DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)


# -------------------- Connect to CARLA --------------------

create_output_directories()

client = carla.Client("localhost", 2000)
client.set_timeout(20.0)

world = client.get_world()
original_settings = world.get_settings()
blueprints = world.get_blueprint_library()

traffic_manager = None
vehicle = None
rgb_camera = None
semantic_camera = None

actors = []
rgb_queue = queue.Queue()
semantic_queue = queue.Queue()

saved_samples = 0


try:
    # Synchronize the CARLA world.
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = TIMESTEP
    world.apply_settings(settings)

    # Traffic Manager must also be synchronized.
    traffic_manager = client.get_trafficmanager(TM_PORT)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(RANDOM_SEED)

    # -------------------- Spawn vehicle --------------------

    spawn_points = world.get_map().get_spawn_points()
    random.Random(RANDOM_SEED).shuffle(spawn_points)

    vehicle_bp = blueprints.find("vehicle.tesla.model3")

    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

        if vehicle is not None:
            break

    if vehicle is None:
        raise RuntimeError("Could not find a free vehicle spawn point.")

    actors.append(vehicle)

    vehicle.set_autopilot(True, TM_PORT)

    # -------------------- Spawn cameras --------------------

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
    actors.append(rgb_camera)

    semantic_camera = world.spawn_actor(
        semantic_bp,
        camera_transform,
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )
    actors.append(semantic_camera)

    rgb_camera.listen(rgb_queue.put)
    semantic_camera.listen(semantic_queue.put)

    print(f"Map: {world.get_map().name}")
    print(f"Vehicle: {vehicle.type_id}")
    print(f"Collecting {NUM_SAMPLES} synchronized samples...")
    print("Press Ctrl+C to stop early.\n")

    # -------------------- Collection loop --------------------

    received_pairs = 0

    while saved_samples < NUM_SAMPLES:
        world.tick()

        rgb_image = rgb_queue.get(timeout=20.0)
        semantic_image = semantic_queue.get(timeout=20.0)

        # Never save mismatched RGB and semantic frames.
        if rgb_image.frame != semantic_image.frame:
            raise RuntimeError(
                f"Frame mismatch: RGB={rgb_image.frame}, "
                f"Semantic={semantic_image.frame}"
            )

        received_pairs += 1

        # Allow autopilot to start moving before saving data.
        if received_pairs <= WARMUP_FRAMES:
            continue

        # Avoid saving nearly identical consecutive frames.
        if (
            received_pairs - WARMUP_FRAMES
        ) % SAVE_EVERY_N_FRAMES != 0:
            continue

        frame_id = rgb_image.frame
        filename = f"{frame_id:06d}.png"

        rgb_image.save_to_disk(
            str(OUTPUT_DIRS["rgb"] / filename)
        )

        semantic_image.save_to_disk(
            str(OUTPUT_DIRS["raw"] / filename),
            carla.ColorConverter.Raw,
        )

        semantic_image.save_to_disk(
            str(OUTPUT_DIRS["visual"] / filename),
            carla.ColorConverter.CityScapesPalette,
        )

        saved_samples += 1

        print(
            f"[{saved_samples:03d}/{NUM_SAMPLES}] "
            f"Saved synchronized frame {frame_id}"
        )

    print("\nDataset collection completed.")
    print(f"Saved {saved_samples} synchronized samples.")
    print(f"Location: {RUN_DIR.resolve()}")

except KeyboardInterrupt:
    print(f"\nStopped early. Saved {saved_samples} samples.")

finally:
    # Disable autopilot before destroying the vehicle.
    if vehicle is not None:
        vehicle.set_autopilot(False, TM_PORT)

    # Stop camera callbacks.
    for sensor in [rgb_camera, semantic_camera]:
        if sensor is not None:
            sensor.stop()

    # Destroy cameras first, followed by the vehicle.
    for actor in reversed(actors):
        actor.destroy()

    # Return Traffic Manager and CARLA to asynchronous mode.
    if traffic_manager is not None:
        traffic_manager.set_synchronous_mode(False)

    world.apply_settings(original_settings)

    print("Actors destroyed and CARLA settings restored.")