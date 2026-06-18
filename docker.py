from dotenv import load_dotenv
from tugboat import binderize, build, create
import os

load_dotenv()

create(
    project=".",
    FROM="python:3.11-bookworm",
    exclude=[
        ".binder/",
        ".venv/",
        "figures/*",
        "!figures/.gitignore",
        ".git/",
        ".gitattributes",
        ".github/",
        ".env"
    ]
)

dock = build(
    image_name="anytime-valid-adaptive-experiments",
    push=True,
    dh_username=os.getenv("DOCKERHUB_USERNAME"),
    dh_password=os.getenv("DOCKERHUB_PASSWORD")
)

binderize()
