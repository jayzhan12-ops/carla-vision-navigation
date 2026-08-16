import carla

client = carla.Client('localhost', 2000)
client.set_timeout(15.0)

print("Client API version:", client.get_client_version())
print("Server API version:", client.get_server_version())

world = client.get_world()
print("Current map:", world.get_map().name)
print("Number of actors in world:", len(world.get_actors()))
