import argparse
import csv
import json
import math
import queue
import random
import re
import time
from pathlib import Path

import carla
import numpy as np
from PIL import Image


# Stable collection settings. These stay fixed across experiments so that only
# town, weather, route/run, split, sample count, and seed change.
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180
FOV = 90
TIMESTEP = 0.1               # 10 simulation ticks per second
SAVE_EVERY_N_FRAMES = 5      # Save at 2 Hz
WARMUP_FRAMES = 20           # Two simulated seconds at 10 Hz

TM_PORT = 8000
SENSOR_WAIT_SECONDS = 1.0
MAX_SYNC_TICKS = 30
RPC_TIMEOUT_SECONDS = 60.0
WORLD_READY_TIMEOUT_SECONDS = 180.0

DATASET_ROOT = Path("data/dataset")
ROAD_CLASS_IDS = (1, 24)     # CARLA 0.9.16: Road and RoadLine

SPLITS = ("train", "validation", "test")
WEATHER_PRESETS = (
    "ClearNoon",
    "CloudyNoon",
    "WetNoon",
    "WetCloudyNoon",
    "SoftRainNoon",
    "MidRainyNoon",
    "HardRainNoon",
    "ClearSunset",
    "CloudySunset",
    "WetSunset",
    "WetCloudySunset",
    "SoftRainSunset",
    "MidRainSunset",
    "HardRainSunset",
)

METADATA_FIELDS = (
    "sample_index",
    "frame_id",
    "timestamp_seconds",
    "split",
    "run_name",
    "town",
    "weather",
    "seed",
    "rgb_file",
    "semantic_file",
    "mask_file",
    "camera_x",
    "camera_y",
    "camera_z",
    "camera_pitch",
    "camera_yaw",
    "camera_roll",
    "vehicle_x",
    "vehicle_y",
    "vehicle_z",
    "vehicle_pitch",
    "vehicle_yaw",
    "vehicle_roll",
    "speed_kmh",
)


def positive_integer(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def safe_run_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError(
            "use only letters, numbers, underscores, and hyphens"
        )
    return value


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Collect synchronized CARLA RGB images, semantic labels, "
            "binary road masks, and per-frame metadata."
        )
    )
    parser.add_argument(
        "--town",
        required=True,
        help="CARLA town, such as Town01 or Town10HD_Opt.",
    )
    parser.add_argument(
        "--weather",
        required=True,
        choices=WEATHER_PRESETS,
        help="CARLA weather preset.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        type=safe_run_name,
        help="Unique name for this complete driving run.",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=SPLITS,
        help="Dataset split assigned to the entire run.",
    )
    parser.add_argument(
        "--samples",
        required=True,
        type=positive_integer,
        help="Number of synchronized samples to save.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for spawn and Traffic Manager choices (default: 42).",
    )
    return parser.parse_args()


def build_output_paths(arguments):
    run_directory = DATASET_ROOT / arguments.split / arguments.run_name
    directories = {
        "rgb": run_directory / "rgb",
        "semantic": run_directory / "semantic_raw",
        "mask": run_directory / "road_masks",
    }
    return run_directory, directories


def prepare_output_directories(run_directory, directories):
    if run_directory.exists() and any(
        path.is_file() for path in run_directory.rglob("*")
    ):
        raise FileExistsError(
            f"{run_directory} already contains files. "
            "Use a new --run-name; existing runs are never overwritten."
        )

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)


def resolve_map_name(client, requested_town):
    available_maps = {
        Path(map_path).name: map_path for map_path in client.get_available_maps()
    }
    if requested_town not in available_maps:
        choices = ", ".join(sorted(available_maps))
        raise ValueError(
            f"Town '{requested_town}' is not installed. Available maps: {choices}"
        )
    return available_maps[requested_town]


def wait_for_world_ready(client, timeout=WORLD_READY_TIMEOUT_SECONDS):
    """Poll CARLA with short RPC calls until its world is responsive."""
    deadline = time.monotonic() + timeout
    last_error = None
    client.set_timeout(5.0)

    while time.monotonic() < deadline:
        try:
            world = client.get_world()
            world.get_map()
            world.get_settings()
            client.set_timeout(RPC_TIMEOUT_SECONDS)
            return world
        except RuntimeError as error:
            last_error = error
            time.sleep(1.0)

    client.set_timeout(RPC_TIMEOUT_SECONDS)
    raise RuntimeError(
        f"CARLA did not become ready within {timeout:.0f} seconds."
    ) from last_error


def configure_camera(blueprint):
    blueprint.set_attribute("image_size_x", str(IMAGE_WIDTH))
    blueprint.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    blueprint.set_attribute("fov", str(FOV))
    blueprint.set_attribute("sensor_tick", "0.0")
    return blueprint


def follow_vehicle(spectator, vehicle):
    vehicle_transform = vehicle.get_transform()
    forward = vehicle_transform.get_forward_vector()
    location = carla.Location(
        x=vehicle_transform.location.x - 6.0 * forward.x,
        y=vehicle_transform.location.y - 6.0 * forward.y,
        z=vehicle_transform.location.z + 3.0,
    )
    rotation = carla.Rotation(
        pitch=-15.0,
        yaw=vehicle_transform.rotation.yaw,
    )
    spectator.set_transform(carla.Transform(location, rotation))


def store_sensor_event(pending_frames, sensor_name, image):
    pending_frames.setdefault(image.frame, {})[sensor_name] = image


def pop_matching_pair(pending_frames):
    matching_frames = [
        frame_id
        for frame_id, frame_data in pending_frames.items()
        if "rgb" in frame_data and "semantic" in frame_data
    ]
    if not matching_frames:
        return None

    frame_id = min(matching_frames)
    frame_data = pending_frames[frame_id]

    # A newer complete pair is safe to use. Older incomplete frames cannot be
    # paired with it and are removed rather than corrupting the dataset.
    for old_frame in [frame for frame in pending_frames if frame <= frame_id]:
        del pending_frames[old_frame]

    return frame_data["rgb"], frame_data["semantic"]


def drain_sensor_queue(sensor_queue, pending_frames):
    while True:
        try:
            sensor_name, image = sensor_queue.get_nowait()
        except queue.Empty:
            return
        store_sensor_event(pending_frames, sensor_name, image)


def get_matching_pair(world, sensor_queue, pending_frames, spectator, vehicle):
    for _ in range(MAX_SYNC_TICKS):
        drain_sensor_queue(sensor_queue, pending_frames)
        pair = pop_matching_pair(pending_frames)
        if pair is not None:
            return pair

        world.tick()
        follow_vehicle(spectator, vehicle)

        deadline = time.monotonic() + SENSOR_WAIT_SECONDS
        while time.monotonic() < deadline:
            try:
                sensor_name, image = sensor_queue.get(
                    timeout=deadline - time.monotonic()
                )
            except queue.Empty:
                break

            store_sensor_event(pending_frames, sensor_name, image)
            pair = pop_matching_pair(pending_frames)
            if pair is not None:
                return pair

    raise RuntimeError(
        "Could not obtain a synchronized RGB/semantic pair after "
        f"{MAX_SYNC_TICKS} simulation ticks."
    )


def save_road_mask(semantic_image, output_path):
    bgra = np.frombuffer(semantic_image.raw_data, dtype=np.uint8).reshape(
        semantic_image.height, semantic_image.width, 4
    )
    labels = bgra[:, :, 2]  # Semantic class ID is stored in the red channel.
    mask = ((labels == ROAD_CLASS_IDS[0]) | (labels == ROAD_CLASS_IDS[1]))
    Image.fromarray(mask.astype(np.uint8) * 255).save(output_path)


def speed_kmh(vehicle):
    velocity = vehicle.get_velocity()
    return 3.6 * math.sqrt(
        velocity.x**2 + velocity.y**2 + velocity.z**2
    )


def transform_fields(prefix, transform):
    return {
        f"{prefix}_x": round(transform.location.x, 3),
        f"{prefix}_y": round(transform.location.y, 3),
        f"{prefix}_z": round(transform.location.z, 3),
        f"{prefix}_pitch": round(transform.rotation.pitch, 3),
        f"{prefix}_yaw": round(transform.rotation.yaw, 3),
        f"{prefix}_roll": round(transform.rotation.roll, 3),
    }


def write_run_config(path, arguments, client, actual_town):
    save_period = TIMESTEP * SAVE_EVERY_N_FRAMES
    config = {
        "carla_client_version": client.get_client_version(),
        "carla_server_version": client.get_server_version(),
        "town": actual_town,
        "weather": arguments.weather,
        "split": arguments.split,
        "run_name": arguments.run_name,
        "seed": arguments.seed,
        "requested_samples": arguments.samples,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "fov_degrees": FOV,
        "simulation_hz": 1.0 / TIMESTEP,
        "saved_sample_hz": 1.0 / save_period,
        "warmup_frames": WARMUP_FRAMES,
        "road_class_ids": list(ROAD_CLASS_IDS),
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def verify_run(directories, expected_samples):
    file_sets = {
        name: {path.name for path in directory.glob("*.png")}
        for name, directory in directories.items()
    }
    counts = {name: len(files) for name, files in file_sets.items()}

    if any(count != expected_samples for count in counts.values()):
        raise RuntimeError(
            f"Dataset integrity check failed. Expected {expected_samples}; "
            f"found {counts}."
        )
    if not (file_sets["rgb"] == file_sets["semantic"] == file_sets["mask"]):
        raise RuntimeError("Dataset integrity check failed: filenames differ.")


def main():
    arguments = parse_arguments()
    run_directory, directories = build_output_paths(arguments)

    # Refuse accidental overwrite before spending time loading a map.
    if run_directory.exists() and any(
        path.is_file() for path in run_directory.rglob("*")
    ):
        raise FileExistsError(
            f"{run_directory} already contains files. Use a new --run-name."
        )

    client = carla.Client("localhost", 2000)
    client.set_timeout(RPC_TIMEOUT_SECONDS)

    world = None
    original_settings = None
    original_weather = None
    traffic_manager = None
    vehicle = None
    rgb_camera = None
    semantic_camera = None
    metadata_file = None
    actors = []
    saved_samples = 0

    sensor_queue = queue.Queue()
    pending_frames = {}

    try:
        current_world = wait_for_world_ready(client)
        map_name = resolve_map_name(client, arguments.town)
        current_town = current_world.get_map().name.split("/")[-1]

        if current_town == arguments.town:
            print(f"Using already loaded town: {current_town}")
            world = current_world
        else:
            print(f"Loading town: {arguments.town}")
            client.set_timeout(WORLD_READY_TIMEOUT_SECONDS)
            client.load_world(map_name)
            world = wait_for_world_ready(client)

        original_settings = world.get_settings()
        original_weather = world.get_weather()
        world.set_weather(getattr(carla.WeatherParameters, arguments.weather))

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = TIMESTEP
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager(TM_PORT)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(arguments.seed)

        prepare_output_directories(run_directory, directories)
        actual_town = world.get_map().name.split("/")[-1]
        write_run_config(
            run_directory / "run_config.json",
            arguments,
            client,
            actual_town,
        )

        blueprints = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        random.Random(arguments.seed).shuffle(spawn_points)

        vehicle_blueprint = blueprints.find("vehicle.tesla.model3")
        for spawn_point in spawn_points:
            vehicle = world.try_spawn_actor(vehicle_blueprint, spawn_point)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError("Could not find a free vehicle spawn point.")

        actors.append(vehicle)
        vehicle.set_autopilot(True, TM_PORT)
        spectator = world.get_spectator()
        follow_vehicle(spectator, vehicle)

        rgb_blueprint = configure_camera(
            blueprints.find("sensor.camera.rgb")
        )
        semantic_blueprint = configure_camera(
            blueprints.find("sensor.camera.semantic_segmentation")
        )
        camera_mount = carla.Transform(
            carla.Location(x=1.5, z=2.0),
            carla.Rotation(pitch=-5.0),
        )

        rgb_camera = world.spawn_actor(
            rgb_blueprint,
            camera_mount,
            attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )
        semantic_camera = world.spawn_actor(
            semantic_blueprint,
            camera_mount,
            attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )
        actors.extend((rgb_camera, semantic_camera))

        rgb_camera.listen(
            lambda image: sensor_queue.put(("rgb", image))
        )
        semantic_camera.listen(
            lambda image: sensor_queue.put(("semantic", image))
        )

        metadata_file = (run_directory / "metadata.csv").open(
            "w", newline="", encoding="utf-8"
        )
        metadata_writer = csv.DictWriter(
            metadata_file, fieldnames=METADATA_FIELDS
        )
        metadata_writer.writeheader()
        metadata_file.flush()

        print(f"Map: {actual_town}")
        print(f"Weather: {arguments.weather}")
        print(f"Split: {arguments.split}")
        print(f"Run: {arguments.run_name}")
        print(f"Seed: {arguments.seed}")
        print(f"Vehicle: {vehicle.type_id}")
        print(f"Collecting {arguments.samples} synchronized samples...")
        print("Press Ctrl+C to stop early.\n")

        received_pairs = 0
        next_save_frame = None

        while saved_samples < arguments.samples:
            rgb_image, semantic_image = get_matching_pair(
                world,
                sensor_queue,
                pending_frames,
                spectator,
                vehicle,
            )
            if rgb_image.frame != semantic_image.frame:
                raise RuntimeError("Internal synchronization error.")

            frame_id = rgb_image.frame
            received_pairs += 1

            if received_pairs <= WARMUP_FRAMES:
                if received_pairs == WARMUP_FRAMES:
                    next_save_frame = frame_id + SAVE_EVERY_N_FRAMES
                    print("Warmup complete. Saving samples...\n")
                continue

            if next_save_frame is None or frame_id < next_save_frame:
                continue

            filename = f"{frame_id:06d}.png"
            rgb_path = directories["rgb"] / filename
            semantic_path = directories["semantic"] / filename
            mask_path = directories["mask"] / filename

            rgb_image.save_to_disk(str(rgb_path))
            semantic_image.save_to_disk(
                str(semantic_path), carla.ColorConverter.Raw
            )
            save_road_mask(semantic_image, mask_path)

            saved_samples += 1
            vehicle_transform = vehicle.get_transform()

            row = {
                "sample_index": saved_samples,
                "frame_id": frame_id,
                "timestamp_seconds": round(rgb_image.timestamp, 3),
                "split": arguments.split,
                "run_name": arguments.run_name,
                "town": actual_town,
                "weather": arguments.weather,
                "seed": arguments.seed,
                "rgb_file": Path("rgb", filename).as_posix(),
                "semantic_file": Path("semantic_raw", filename).as_posix(),
                "mask_file": Path("road_masks", filename).as_posix(),
                "speed_kmh": round(speed_kmh(vehicle), 3),
            }
            row.update(transform_fields("camera", rgb_image.transform))
            row.update(transform_fields("vehicle", vehicle_transform))
            metadata_writer.writerow(row)
            metadata_file.flush()

            next_save_frame = frame_id + SAVE_EVERY_N_FRAMES
            print(
                f"[{saved_samples:03d}/{arguments.samples}] "
                f"Saved synchronized frame {frame_id}"
            )

        metadata_file.close()
        metadata_file = None
        verify_run(directories, arguments.samples)

        print("\nDataset collection completed and verified.")
        print(f"Saved samples: {saved_samples}")
        print(f"Dataset: {run_directory.resolve()}")

    except KeyboardInterrupt:
        print(f"\nCollection stopped early. Saved {saved_samples} samples.")

    finally:
        if metadata_file is not None:
            metadata_file.close()

        if vehicle is not None:
            try:
                vehicle.set_autopilot(False, TM_PORT)
            except RuntimeError:
                pass

        for sensor in (rgb_camera, semantic_camera):
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

        if world is not None:
            if original_weather is not None:
                try:
                    world.set_weather(original_weather)
                except RuntimeError:
                    pass
            if original_settings is not None:
                try:
                    world.apply_settings(original_settings)
                except RuntimeError:
                    pass

        print("Actors destroyed and CARLA settings restored.")


if __name__ == "__main__":
    main()
