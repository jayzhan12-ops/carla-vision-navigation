import queue
import random
import time
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

SENSOR_WAIT_SECONDS = 1.0
MAX_SYNC_TICKS = 30

# Use a new name because run 01 contains partial data.
RUN_NAME = "town10_clear_02"
RUN_DIR = Path("data/dataset") / RUN_NAME

OUTPUT_DIRS = {
    "rgb": RUN_DIR / "rgb",
    "raw": RUN_DIR / "semantic_raw",
    "visual": RUN_DIR / "semantic_visual",
}


# -------------------- Helper functions --------------------

def configure_camera(blueprint):
    """Give both cameras identical settings."""
    blueprint.set_attribute("image_size_x", IMAGE_WIDTH)
    blueprint.set_attribute("image_size_y", IMAGE_HEIGHT)
    blueprint.set_attribute("fov", FOV)

    # 0.0 means capture on every world tick.
    # The world runs at 10 Hz, so the cameras also run at 10 Hz.
    blueprint.set_attribute("sensor_tick", "0.0")

    return blueprint


def create_output_directories():
    """Create a new run without overwriting existing images."""
    if RUN_DIR.exists() and any(RUN_DIR.rglob("*.png")):
        raise RuntimeError(
            f"{RUN_DIR} already contains images. "
            "Change RUN_NAME before collecting again."
        )

    for directory in OUTPUT_DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)


def follow_vehicle(spectator, vehicle):
    """Move CARLA's spectator camera behind the vehicle."""
    vehicle_transform = vehicle.get_transform()
    forward = vehicle_transform.get_forward_vector()

    spectator_location = carla.Location(
        x=vehicle_transform.location.x - 6.0 * forward.x,
        y=vehicle_transform.location.y - 6.0 * forward.y,
        z=vehicle_transform.location.z + 3.0,
    )

    spectator_rotation = carla.Rotation(
        pitch=-15.0,
        yaw=vehicle_transform.rotation.yaw,
        roll=0.0,
    )

    spectator.set_transform(
        carla.Transform(
            spectator_location,
            spectator_rotation,
        )
    )


def store_sensor_event(pending_frames, sensor_name, image):
    """Store an image under its CARLA frame ID."""
    frame_data = pending_frames.setdefault(image.frame, {})
    frame_data[sensor_name] = image


def pop_matching_pair(pending_frames):
    """Return the oldest frame containing both camera images."""
    matching_frames = sorted(
        frame_id
        for frame_id, frame_data in pending_frames.items()
        if "rgb" in frame_data and "semantic" in frame_data
    )

    if not matching_frames:
        return None

    frame_id = matching_frames[0]
    frame_data = pending_frames[frame_id]

    rgb_image = frame_data["rgb"]
    semantic_image = frame_data["semantic"]

    # Remove the matched frame and any older incomplete frames.
    old_frames = [
        old_frame
        for old_frame in pending_frames
        if old_frame <= frame_id
    ]

    for old_frame in old_frames:
        del pending_frames[old_frame]

    return rgb_image, semantic_image


def drain_sensor_queue(sensor_queue, pending_frames):
    """Move all currently available sensor messages into the buffer."""
    while True:
        try:
            sensor_name, image = sensor_queue.get_nowait()
        except queue.Empty:
            break

        store_sensor_event(
            pending_frames,
            sensor_name,
            image,
        )


def get_matching_pair(
    world,
    sensor_queue,
    pending_frames,
    spectator,
    vehicle,
):
    """
    Advance CARLA until RGB and semantic images with the same
    frame ID are available.
    """
    for _ in range(MAX_SYNC_TICKS):
        # First process anything already waiting in the queue.
        drain_sensor_queue(sensor_queue, pending_frames)

        matching_pair = pop_matching_pair(pending_frames)

        if matching_pair is not None:
            return matching_pair

        # No complete pair yet, so advance the simulation.
        world.tick()
        follow_vehicle(spectator, vehicle)

        # Give GPU-camera callbacks time to deliver their data.
        deadline = time.monotonic() + SENSOR_WAIT_SECONDS

        while time.monotonic() < deadline:
            remaining_time = deadline - time.monotonic()

            try:
                sensor_name, image = sensor_queue.get(
                    timeout=remaining_time
                )
            except queue.Empty:
                break

            store_sensor_event(
                pending_frames,
                sensor_name,
                image,
            )

            matching_pair = pop_matching_pair(pending_frames)

            if matching_pair is not None:
                return matching_pair

    raise RuntimeError(
        "Could not obtain a synchronized RGB/semantic pair "
        f"after {MAX_SYNC_TICKS} simulation ticks."
    )


# -------------------- Setup --------------------

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

# Both cameras send tagged images into one queue.
sensor_queue = queue.Queue()

# Stores images temporarily by CARLA frame ID.
pending_frames = {}

saved_samples = 0


try:
    # -------------------- Synchronization --------------------

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = TIMESTEP
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(TM_PORT)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(RANDOM_SEED)

    # -------------------- Vehicle --------------------

    spawn_points = world.get_map().get_spawn_points()
    random.Random(RANDOM_SEED).shuffle(spawn_points)

    vehicle_bp = blueprints.find("vehicle.tesla.model3")

    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(
            vehicle_bp,
            spawn_point,
        )

        if vehicle is not None:
            break

    if vehicle is None:
        raise RuntimeError(
            "Could not find a free vehicle spawn point."
        )

    actors.append(vehicle)

    vehicle.set_autopilot(True, TM_PORT)

    spectator = world.get_spectator()
    follow_vehicle(spectator, vehicle)

    # -------------------- Cameras --------------------

    rgb_bp = configure_camera(
        blueprints.find("sensor.camera.rgb")
    )

    semantic_bp = configure_camera(
        blueprints.find(
            "sensor.camera.semantic_segmentation"
        )
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

    rgb_camera.listen(
        lambda image: sensor_queue.put(("rgb", image))
    )

    semantic_camera.listen(
        lambda image: sensor_queue.put(("semantic", image))
    )

    # -------------------- Collection --------------------

    print(f"Map: {world.get_map().name}")
    print(f"Vehicle: {vehicle.type_id}")
    print("Waiting for synchronized camera data...")
    print(f"Collecting {NUM_SAMPLES} samples...")
    print("Press Ctrl+C to stop early.\n")

    received_pairs = 0
    next_save_frame = None

    while saved_samples < NUM_SAMPLES:
        rgb_image, semantic_image = get_matching_pair(
            world,
            sensor_queue,
            pending_frames,
            spectator,
            vehicle,
        )

        # This should always be true because the buffer matched them.
        assert rgb_image.frame == semantic_image.frame

        frame_id = rgb_image.frame
        received_pairs += 1

        # Let autopilot and sensors stabilize before saving.
        if received_pairs <= WARMUP_FRAMES:
            if received_pairs == WARMUP_FRAMES:
                next_save_frame = (
                    frame_id + SAVE_EVERY_N_FRAMES
                )
                print("Warmup complete. Saving samples...\n")

            continue

        # Require at least five CARLA frames between saved samples.
        if frame_id < next_save_frame:
            continue

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
        next_save_frame = (
            frame_id + SAVE_EVERY_N_FRAMES
        )

        print(
            f"[{saved_samples:03d}/{NUM_SAMPLES}] "
            f"Saved synchronized frame {frame_id}"
        )

    print("\nDataset collection completed.")
    print(f"Saved samples: {saved_samples}")
    print(f"Dataset: {RUN_DIR.resolve()}")

except KeyboardInterrupt:
    print(
        f"\nCollection stopped early. "
        f"Saved {saved_samples} samples."
    )

finally:
    # -------------------- Cleanup --------------------

    if vehicle is not None:
        try:
            vehicle.set_autopilot(False, TM_PORT)
        except RuntimeError:
            pass

    for sensor in [rgb_camera, semantic_camera]:
        if sensor is not None:
            try:
                sensor.stop()
            except RuntimeError:
                pass

    for actor in reversed(actors):
        try:
            actor.destroy()
        except RuntimeError:
            pass

    if traffic_manager is not None:
        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass

    try:
        world.apply_settings(original_settings)
    except RuntimeError:
        pass

    print("Actors destroyed and CARLA settings restored.")