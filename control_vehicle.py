import math
import random
import time

import carla


def calculate_speed_kmh(vehicle):
    velocity = vehicle.get_velocity()

    speed_mps = math.sqrt(
        velocity.x**2
        + velocity.y**2
        + velocity.z**2
    )

    return speed_mps * 3.6


client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()
vehicle = None

try:
    blueprint_library = world.get_blueprint_library()

    # Use a regular passenger vehicle
    vehicle_blueprint = blueprint_library.find("vehicle.tesla.model3")

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

    # Place the spectator above the vehicle
    vehicle_transform = vehicle.get_transform()

    spectator_transform = carla.Transform(
        vehicle_transform.location + carla.Location(z=15),
        carla.Rotation(
            pitch=-90,
            yaw=vehicle_transform.rotation.yaw,
        ),
    )

    world.get_spectator().set_transform(spectator_transform)

    input("Press Enter to move the vehicle...")

    # Move forward
    vehicle.apply_control(
        carla.VehicleControl(
            throttle=0.35,
            steer=0.0,
            brake=0.0,
        )
    )

    time.sleep(3.0)

    print(f"Speed before braking: {calculate_speed_kmh(vehicle):.1f} km/h")

    # Stop the vehicle
    vehicle.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
        )
    )

    time.sleep(2.0)

    print(f"Speed after braking: {calculate_speed_kmh(vehicle):.1f} km/h")

    input("Press Enter to destroy the vehicle...")

finally:
    if vehicle is not None:
        vehicle.apply_control(carla.VehicleControl(brake=1.0))
        vehicle.destroy()
        print("Vehicle destroyed safely")