import tomllib
import json

with open("./example.toml", "rb") as file:
    content = tomllib.load(file)
    print(json.dumps(content, indent=2))
