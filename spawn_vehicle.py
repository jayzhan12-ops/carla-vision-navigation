import random
import carla


client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()

vehicle = None

try:
    # Choose a vehicle blueprint
    vehicle_blueprints = world.get_blueprint_library().filter("vehicle.*")
    vehicle_blueprint = random.choice(vehicle_blueprints)

    # Try available spawn points until one succeeds
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_blueprint, spawn_point)

        if vehicle is not None:
            break

    if vehicle is None:
        raise RuntimeError("Could not find an available spawn point")

    print("Vehicle spawned successfully")
    print("Vehicle:", vehicle.type_id)
    print("Actor ID:", vehicle.id)

    # Move the spectator above the vehicle
    vehicle_transform = vehicle.get_transform()

    spectator_transform = carla.Transform(
        vehicle_transform.location + carla.Location(z=15),
        carla.Rotation(
            pitch=-90,
            yaw=vehicle_transform.rotation.yaw,
        ),
    )

    world.get_spectator().set_transform(spectator_transform)

    input("Look at the CARLA window. Press Enter to destroy the vehicle...")

finally:
    if vehicle is not None:
        vehicle.destroy()
        print("Vehicle destroyed safely")